"""Matched development backtest for Ontario hourly demand and window peaks."""

import argparse
import csv
from datetime import date, datetime, timedelta
from functools import lru_cache
import io
import json
from pathlib import Path

import numpy as np
from scipy.linalg import lstsq

import baselines
import boosted_demand as existing
from ieso import (EST, HOUR, assumed_available_at, digest, iso, load_reports,
                  market_midnight, parse_date, parse_origin, provenance, write_json)


KEYS = ("origin_utc", "target_start_utc", "issued_at_utc", "available_at_utc", "source")
HOLIDAY_NAMES = ("holiday", "day_before_holiday", "day_after_holiday")
TORONTO = "toronto_c"


def nth_monday(year, month, n):
    first = date(year, month, 1)
    return first + timedelta(days=(-first.weekday()) % 7 + 7 * (n - 1))


@lru_cache(maxsize=32)
def holidays(year):
    # Gregorian computus, used for Good Friday.
    from dateutil.easter import easter

    may24 = date(year, 5, 24)
    canada = date(year, 7, 1)
    return {date(year, 1, 1), nth_monday(year, 2, 3), easter(year) - timedelta(days=2),
            may24 - timedelta(days=may24.weekday()), canada + timedelta(days=canada.weekday() == 6),
            nth_monday(year, 9, 1), nth_monday(year, 10, 2), date(year, 12, 25), date(year, 12, 26)}


def holiday_features(local_day):
    days = [local_day + timedelta(days=offset) for offset in (0, 1, -1)]
    return [int(day in holidays(day.year)) for day in days]


def timestamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("forecast timestamps must have an explicit UTC offset of zero")
    return parsed


def read_forecasts(path, operator=False):
    payload = path.read_bytes()
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline=""), strict=True)
    header = reader.fieldnames or []
    columns = tuple(header[len(KEYS):])
    if (tuple(header[:len(KEYS)]) != KEYS or len(set(header)) != len(header) or
            (columns != ("average_demand_mw",) if operator else
             TORONTO not in columns or any(not name.endswith("_c") for name in columns))):
        raise ValueError(f"invalid forecast columns in {path}")
    rows = {}
    for row in reader:
        if None in row or any(value is None for value in row.values()) or not row["source"].strip():
            raise ValueError(f"incomplete forecast row in {path}:{reader.line_num}")
        origin, target = (parse_origin(row[key]) for key in KEYS[:2])
        issued, available = (timestamp(row[key]) for key in KEYS[2:4])
        values = np.array([float(row[name]) for name in columns])
        key = (origin, target)
        if (key in rows or not issued <= available <= origin or not origin <= target < origin + 24 * HOUR
                or not np.isfinite(values).all() or (operator and (values < 0).any())):
            raise ValueError(f"duplicate, unavailable, or invalid forecast row in {path}:{reader.line_num}")
        rows[key] = values, (row["source"], iso(issued), iso(available))
    if not rows:
        raise ValueError(f"empty forecast series in {path}")
    return columns, rows, {"path": str(path), "sha256": digest(payload)}


def forecast_window(rows, origin, *, operator=False):
    try:
        records = [rows[origin, origin + step * HOUR] for step in range(24)]
    except KeyError as error:
        raise ValueError(f"missing forecast at {error.args[0]}; no scoring rows were trimmed") from error
    values, evidence = zip(*records)
    groups = (evidence[:16], evidence[16:]) if operator else (evidence,)
    if any(len(set(group)) != 1 for group in groups):
        raise ValueError(f"inconsistent forecast issuance at {iso(origin)}")
    return np.stack(values), [dict(zip(KEYS[4:] + KEYS[2:4], group[0])) for group in groups]


def prepare(observations, start, end, weather=None, *, fit_cutoff=None):
    rows, targets, temperatures, sources, origins = [], [], [], [], []
    for offset in range((end - start).days):
        origin = existing.origin_at(start + timedelta(days=offset))
        hours = [origin + step * HOUR for step in range(24)]
        if fit_cutoff is not None and (hours[-1] + HOUR > market_midnight(end) or
                                       assumed_available_at(hours[-1]) > fit_cutoff):
            continue
        base, _ = existing.feature_rows(observations, origin)
        local_days = [datetime.fromisoformat(baselines.calendar_fields(hour)["local_start"]).date()
                      for hour in hours]
        rows.append([row + holiday_features(day) for row, day in zip(base, local_days)])
        targets.append([observations[hour].demand_mw for hour in hours])
        origins.append(origin)
        if weather is not None:
            values, evidence = forecast_window(weather, origin)
            temperatures.append(values)
            sources.append(evidence)
    return np.asarray(rows), np.asarray(targets), origins, np.asarray(temperatures), sources


def weather_features(temperature):
    shape = temperature.shape
    low, high = (np.broadcast_to(reduce(temperature, axis=1, keepdims=True), shape)
                 for reduce in (np.min, np.max))
    return np.concatenate((temperature, low, high), axis=2)


def fit_boosted(training, loads, rows, feature_names):
    import lightgbm as lgb

    model = lgb.train(existing.PARAMETERS, lgb.Dataset(
        training.reshape(-1, len(feature_names)), label=loads.ravel(), feature_name=list(feature_names)),
        num_boost_round=existing.ROUNDS)
    text = model.model_to_string()
    predictions = model.predict(rows.reshape(-1, len(feature_names)), num_threads=1).reshape(rows.shape[:2])
    return predictions, {"model_text": text, "model_sha256": digest(text.encode())}


def spline_design(rows, temperature):
    hour = rows[..., existing.FEATURES.index("local_hour")]
    weekday = rows[..., existing.FEATURES.index("local_weekday")]
    hourly = np.eye(24)[hour.astype(int)]
    weekdays = np.eye(7)[weekday.astype(int)]
    groups = [hourly, weekdays, hourly * (weekday >= 5)[..., None], hourly * rows[..., -3:-2],
              rows[..., [5, 6]], rows[..., 9:10], rows[..., 11:19], rows[..., -3:]]
    if temperature.size:
        # Fixed knots for heating, comfort and cooling, plus daily thermal extrema.
        groups.extend([weather_features(temperature), *(np.maximum(temperature - knot, 0)
                                                        for knot in (-10, 0, 10, 18, 25))])
        groups.extend([hourly * np.maximum(15 - temperature[..., :1], 0),
                       hourly * np.maximum(temperature[..., :1] - 20, 0)])
    return np.concatenate(groups, axis=2).reshape(hour.size, -1)


def fit_spline(train, loads, rows, train_temperature, temperature):
    x, y = spline_design(train, train_temperature), loads.ravel()
    scoring = spline_design(rows, temperature)
    center, spread = x.mean(axis=0), x.std(axis=0)
    spread[spread == 0] = 1
    x = (x - center) / spread
    penalty = np.sqrt(10) * np.eye(x.shape[1])
    weights = lstsq(np.vstack((x, penalty)), np.concatenate((y - y.mean(), np.zeros(len(penalty)))))[0]
    prediction = y.mean() + (scoring - center) / spread @ weights
    return np.maximum(prediction.reshape(rows.shape[:2]), 0)


def peak_risk(predictions, errors, threshold):
    # Carry whole historical error curves forward to preserve dependence across lead hours.
    peaks = np.maximum(predictions[:, None, :] + errors[None, :, :], 0).max(axis=2)
    lower, upper = np.quantile(peaks, (0.05, 0.95), axis=1)
    return lower, upper, (peaks >= threshold).mean(axis=1)


def evaluate_points(actual, predictions, risk, threshold, mask):
    if not mask.any():
        return {"windows": 0}
    y, p = actual[mask], predictions[mask]
    peaks = y.max(axis=1)
    lower, upper, probability = (values[mask] for values in risk)
    return {"windows": int(mask.sum()), "hourly_mae_mw": float(np.abs(p - y).mean()),
            "hourly_bias_mw": float((p - y).mean()), "peak_mae_mw": float(np.abs(p.max(axis=1) - peaks).mean()),
            "peak_hour_mae": float(np.abs(p.argmax(axis=1) - y.argmax(axis=1)).mean()),
            "peak_90_coverage": float(((lower <= peaks) & (peaks <= upper)).mean()),
            "peak_90_mean_width_mw": float((upper - lower).mean()),
            "peak_90_interval_score_mw": float((upper - lower + 20 * np.maximum(lower - peaks, 0)
                                                + 20 * np.maximum(peaks - upper, 0)).mean()),
            "peak_exceedance_brier": float(((probability - (peaks >= threshold)) ** 2).mean()),
            "peak_exceedance_events": int((peaks >= threshold).sum())}


def evaluate(paths, train_start, calibration_start, start, end, threshold, weather_path=None, operator_path=None,
             *, recent_paths=(), train_end=None):
    train_end = train_end or calibration_start
    if (not train_start < train_end <= calibration_start < start < end or
            (recent_paths and (train_end > existing.DEVELOPMENT_END or calibration_start < date(2026, 1, 1))) or
            (not recent_paths and end > existing.DEVELOPMENT_END)):
        raise ValueError("invalid development split; use historical development through 2023 or separate 2026+ scoring data")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("peak threshold must be finite and positive")
    observations, inputs = load_reports(paths)
    recent, recent_inputs = load_reports(recent_paths) if recent_paths else (observations, [])
    weather_columns, weather, weather_input = read_forecasts(weather_path) if weather_path else ((), None, None)
    train, loads, train_origins, train_temperatures, _ = prepare(
        observations, train_start, train_end, weather, fit_cutoff=existing.origin_at(train_end))
    cal, cal_loads, cal_origins, cal_temperatures, cal_weather_sources = prepare(
        recent, calibration_start, start, weather, fit_cutoff=existing.origin_at(start))
    rows, actual, origins, temperatures, weather_sources = prepare(recent, start, end, weather)
    if min(len(train_origins), len(cal_origins)) < existing.MIN_TRAIN_ORIGINS:
        raise ValueError("need 30 eligible origins each for training and calibration")
    expanded = np.concatenate((cal, rows))
    expanded_weather = np.concatenate((cal_temperatures, temperatures))
    width = len(existing.FEATURES)
    control = existing.fit_model(train[..., :width].reshape(-1, width), loads.ravel())
    control_text = control.model_to_string()
    models = {"lightgbm": {"model_text": control_text, "model_sha256": digest(control_text.encode())}}
    predictions = {"lightgbm": np.array(existing.predict(
        control, expanded[..., :width].reshape(-1, width))).reshape(expanded.shape[:2]),
        "daily": expanded[..., existing.FEATURES.index("daily_seasonal_mw")],
        "weekly": expanded[..., existing.FEATURES.index("weekly_seasonal_mw")]}
    calendar_names = existing.FEATURES + HOLIDAY_NAMES
    predictions["lightgbm_holidays"], models["lightgbm_holidays"] = fit_boosted(train, loads, expanded, calendar_names)
    challenger = "lightgbm_holidays"
    if weather is not None:
        names = calendar_names + tuple(f"{kind}_{site}" for kind in ("hour", "low", "high") for site in weather_columns)
        predictions["lightgbm_weather"], models["lightgbm_weather"] = fit_boosted(
            np.concatenate((train, weather_features(train_temperatures)), axis=2), loads,
            np.concatenate((expanded, weather_features(expanded_weather)), axis=2), names)
        challenger = "lightgbm_weather"
    predictions["spline"] = fit_spline(train, loads, expanded, train_temperatures, expanded_weather)
    predictions["blend"] = (predictions[challenger] + predictions["spline"]) / 2
    operator_input, operator_sources = None, []
    if operator_path:
        _, operator, operator_input = read_forecasts(operator_path, operator=True)
        windows, operator_sources = zip(*(forecast_window(operator, origin, operator=True)
                                         for origin in cal_origins + origins))
        predictions["ieso_average_demand"] = np.stack(windows)[..., 0]
    if any(not np.isfinite(p).all() or (p < 0).any() for p in predictions.values()):
        raise ValueError("model returned invalid demand predictions")
    masks = {"all": np.ones(len(origins), dtype=bool)}
    masks["winter"] = np.array([origin.astimezone(EST).month in (12, 1, 2) for origin in origins])
    masks["summer"] = np.array([origin.astimezone(EST).month in (6, 7, 8) for origin in origins])
    if weather is not None:
        toronto = temperatures[..., weather_columns.index(TORONTO)]
        masks["forecast_hot"] = toronto.max(axis=1) >= 28
        masks["forecast_cold"] = toronto.min(axis=1) <= -10
    metrics, forecasts = {}, {}
    for name, values in predictions.items():
        calibration, scoring = np.split(values, [len(cal_origins)])
        errors = cal_loads - calibration
        risk = peak_risk(scoring, errors, threshold)
        metrics[name] = {key: evaluate_points(actual, scoring, risk, threshold, mask) for key, mask in masks.items()}
        forecasts[name] = {"hourly_mw": scoring.tolist(), "peak_lower_mw": risk[0].tolist(),
                           "peak_upper_mw": risk[1].tolist(), "peak_exceedance_probability": risk[2].tolist(),
                           "calibration_errors_mw": errors.tolist()}
    source = provenance()
    for path in (Path(__file__), Path(existing.__file__)):
        source["source_hashes"][f"apps/energy/{path.name}"] = digest(path.read_bytes())
    source["dependencies"] = {name: existing.version(name) for name in
                              ("numpy", "scipy", "lightgbm", "python-dateutil")}
    return {"schema_version": 1, **source, "limitation": baselines.LIMITATION, "phase": "development",
            "training_start": str(train_start), "training_end_exclusive": str(train_end),
            "training_windows": len(train_origins),
            "calibration_start": str(calibration_start), "calibration_windows": len(cal_origins),
            "start": str(start), "end_exclusive": str(end), "threshold_mw": threshold,
            "hourly_targets": actual.size, "demand_inputs": inputs, "recent_demand_inputs": recent_inputs,
            "weather_input": weather_input,
            "operator_input": operator_input, "metrics": metrics, "models": models,
            "origins_utc": list(map(iso, origins)), "calibration_origins_utc": list(map(iso, cal_origins)),
            "actual_hourly_mw": actual.tolist(), "forecasts": forecasts,
            "forecast_sources": {"weather_calibration": cal_weather_sources, "weather_scoring": weather_sources,
                                 "operator_calibration_then_scoring": operator_sources}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, nargs="+", type=Path)
    parser.add_argument("--recent-csv", nargs="+", type=Path, default=())
    parser.add_argument("--train-start", type=parse_date, default=date(2018, 1, 15))
    parser.add_argument("--train-end", type=parse_date, help="exclusive training end, defaults to calibration start")
    parser.add_argument("--calibration-start", type=parse_date, default=date(2022, 1, 1))
    parser.add_argument("--start", type=parse_date, default=date(2023, 1, 1))
    parser.add_argument("--end", type=parse_date, default=existing.DEVELOPMENT_END)
    parser.add_argument("--threshold-mw", type=float, default=22_000)
    parser.add_argument("--weather", type=Path)
    parser.add_argument("--operator", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--retrospective", action="store_true", required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("output exists; choose a new path")
        report = evaluate(args.csv, args.train_start, args.calibration_start, args.start, args.end,
                          args.threshold_mw, args.weather, args.operator, recent_paths=args.recent_csv,
                          train_end=args.train_end)
        write_json(args.output, report)
    except (ValueError, KeyError, OSError, csv.Error) as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps({name: results["all"] for name, results in report["metrics"].items()}, indent=2))


if __name__ == "__main__":
    main()
