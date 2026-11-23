"""Matched retrospective benchmark of archived weather and IESO forecasts."""

import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np

import boosted_demand as existing
from demand_experiment import (HOLIDAY_NAMES, TORONTO, evaluate_points, fit_boosted, fit_spline,
                               peak_risk, prepare, read_forecasts, timestamp, weather_features)
from ieso import HOUR, UTC, digest, iso, load_reports, parse_origin, provenance, write_json


TRAIN_2024 = (date(2024, 3, 15), date(2024, 12, 31))
TRAIN_2026 = (date(2026, 1, 15), date(2026, 6, 1))
CALIBRATION_START = date(2026, 6, 1)
SCORE_START = date(2026, 7, 15)
SCORE_END = date(2026, 11, 20)
THRESHOLD = 22_000
BOOST_SETTINGS = {"small": {"num_leaves": 15, "max_depth": 4, "min_data_in_leaf": 80},
                  "regularized": {"num_leaves": 31, "max_depth": 6, "lambda_l2": 20},
                  "wide": {"num_leaves": 47, "max_depth": 8, "min_data_in_leaf": 30}}


def blocks(observations, start, end, weather, *, purge, known_gaps):
    # Missing archive origins must be named before the comparison. All models share that mask.
    eligible_end = end - timedelta(days=1) if purge else end
    spans, omitted = [], []
    current_start = start
    for step in range((eligible_end - start).days):
        day = start + timedelta(days=step)
        origin = existing.origin_at(day)
        if all((origin, origin + hour * HOUR) in weather for hour in range(24)):
            continue
        if day not in known_gaps:
            raise ValueError(f"weather archive is incomplete on {day}; document the gap before comparing")
        if current_start < day:
            spans.append((current_start, day))
        omitted.append(day.isoformat())
        current_start = day + timedelta(days=1)
    if current_start < eligible_end:
        spans.append((current_start, eligible_end))
    results = [prepare(observations, first, last, weather) for first, last in spans]
    if not results:
        raise ValueError("no complete origins in benchmark period")
    rows, loads, origins, temperatures, sources = zip(*results)
    return (np.concatenate(rows), np.concatenate(loads), sum(origins, []),
            np.concatenate(temperatures), sum(sources, []), omitted)


def with_weather(rows, temperature, columns):
    names = existing.FEATURES + HOLIDAY_NAMES + tuple(
        f"{kind}_{site}" for kind in ("hour", "low", "high") for site in columns)
    return np.concatenate((rows, weather_features(temperature)), axis=2), names


def paired_interval(actual, control, challenger):
    delta = np.stack(((np.abs(control - actual) - np.abs(challenger - actual)).mean(axis=1),
                      np.abs(control.max(axis=1) - actual.max(axis=1)) -
                      np.abs(challenger.max(axis=1) - actual.max(axis=1))), axis=1)
    rng = np.random.default_rng(42)
    count = len(delta)
    block = 7
    if count < 10 * block:
        return None
    starts = rng.integers(count, size=(2000, (count + block - 1) // block))
    indices = ((starts[..., None] + np.arange(block)) % count).reshape(2000, -1)[:, :count]
    lower, upper = np.quantile(delta[indices].mean(axis=1), (0.025, 0.975), axis=0)
    return {"hourly_reduction_mw": [float(lower[0]), float(upper[0])],
            "peak_reduction_mw": [float(lower[1]), float(upper[1])]}


def operator_values(path, origins):
    raw = path.read_bytes()
    archives = json.loads(raw)
    rows, matched = [], []
    for origin in origins:
        day = origin.date().isoformat()
        groups = archives.get(day, [])
        row = {}
        for report in groups:
            if not origin - 24 * HOUR <= timestamp(report["issued"]) <= timestamp(report["last_modified"]) <= origin:
                raise ValueError(f"operator vintage is ineligible at {day}")
            for key, value in report["hours"].items():
                target = parse_origin(key)
                if target in row or not origin <= target < origin + 24 * HOUR or not np.isfinite(value) or value < 0:
                    raise ValueError(f"invalid operator forecast at {key}")
                row[target] = value
        matched.append(len(row) == 24)
        if len(row) == 24:
            rows.append([row[origin + i * HOUR] for i in range(24)])
    return np.asarray(rows), np.asarray(matched), {"sha256": digest(raw), "snapshots": archives}


def point_scores(actual, prediction):
    error = prediction - actual
    return {"windows": len(actual), "hourly_mae_mw": float(np.abs(error).mean()),
            "wape_percent": float(100 * np.abs(error).sum() / actual.sum()),
            "hourly_bias_mw": float(error.mean()),
            "peak_mae_mw": float(np.abs(prediction.max(axis=1) - actual.max(axis=1)).mean()),
            "peak_hour_mae": float(np.abs(prediction.argmax(axis=1) - actual.argmax(axis=1)).mean())}


def select_tuning(train, loads, extended, width_names, spline, default, labels):
    count = len(labels)
    base_score = float(np.abs(default[:count] - labels).mean())
    candidates = [(base_score, "default_50_50", default, None)]
    x = train.reshape(-1, len(width_names))
    score_x = extended.reshape(-1, len(width_names))
    for name, override in BOOST_SETTINGS.items():
        model = lgb.train(existing.PARAMETERS | override, lgb.Dataset(x, label=loads.ravel(),
                          feature_name=list(width_names)), num_boost_round=existing.ROUNDS)
        prediction = model.predict(score_x, num_threads=1).reshape(extended.shape[:2])
        for weight in (0.25, 0.5, 0.75):
            result = weight * prediction + (1 - weight) * spline
            score = float(np.abs(result[:count] - labels).mean())
            candidates.append((score, f"{name}_{weight}", result, model.model_to_string()))
    winner = min(candidates, key=lambda candidate: candidate[0])
    return winner[2], {"selected": winner[1], "model_text": winner[3],
                       "calibration_mae_mw": {name: score for score, name, _, _ in candidates}}


def evaluate(demand_2024, demand_2026, weather_path, operator_path, known_gaps, tune):
    began = time.perf_counter()
    weather_columns, weather, weather_input = read_forecasts(weather_path)
    if weather_columns[0] != TORONTO:
        raise ValueError("Toronto must be the first weather column")
    history, history_inputs = load_reports([demand_2024])
    recent, recent_inputs = load_reports([demand_2026])
    early = blocks(history, *TRAIN_2024, weather, purge=True, known_gaps=known_gaps)
    later = blocks(recent, *TRAIN_2026, weather, purge=True, known_gaps=known_gaps)
    cal = blocks(recent, CALIBRATION_START, SCORE_START, weather, purge=True, known_gaps=known_gaps)
    score = blocks(recent, SCORE_START, SCORE_END, weather, purge=False, known_gaps=known_gaps)
    train, loads, train_temperature = (np.concatenate((early[i], later[i])) for i in (0, 1, 3))
    cal_actual, score_actual = cal[1], score[1]
    rows, temperatures = (np.concatenate((cal[i], score[i])) for i in (0, 3))
    n_cal = len(cal_actual)
    empty = np.array([])
    values = {"daily": rows[..., existing.FEATURES.index("daily_seasonal_mw")],
              "weekly": rows[..., existing.FEATURES.index("weekly_seasonal_mw")]}
    models, splines = {}, {}
    for label, columns in (("calendar", ()), ("toronto", weather_columns[:1]), ("four_city", weather_columns)):
        t = train_temperature[..., :len(columns)] if columns else empty
        a = temperatures[..., :len(columns)] if columns else empty
        fit, names = with_weather(train, t, columns) if columns else (train, existing.FEATURES + HOLIDAY_NAMES)
        features, _ = with_weather(rows, a, columns) if columns else (rows, names)
        lgbm, models[label] = fit_boosted(fit, loads, features, names)
        spline = fit_spline(train, loads, rows, t, a)
        values[f"{label}_lightgbm"], values[f"{label}_blend"] = lgbm, (lgbm + spline) / 2
        splines[label] = spline
    cal_scores = {name: point_scores(cal_actual, p[:n_cal]) for name, p in values.items()}
    weather_best = min(("toronto", "four_city"), key=lambda label: cal_scores[f"{label}_blend"]["hourly_mae_mw"])
    tuning = None
    if tune and cal_scores[f"{weather_best}_blend"]["hourly_mae_mw"] < 0.99 * cal_scores["calendar_blend"]["hourly_mae_mw"]:
        columns = weather_columns[:1] if weather_best == "toronto" else weather_columns
        fit, names = with_weather(train, train_temperature[..., :len(columns)], columns)
        features, _ = with_weather(rows, temperatures[..., :len(columns)], columns)
        values["tuned_blend"], tuning = select_tuning(
            fit, loads, features, names, splines[weather_best], values[f"{weather_best}_blend"], cal_actual)
    if any(not np.isfinite(prediction).all() or (prediction < 0).any() for prediction in values.values()):
        raise ValueError("forecast contains invalid demand values")
    operator, matched, operator_input = operator_values(operator_path, score[2])
    masks = {"all": np.ones(len(score_actual), dtype=bool),
             "hot": score[3][..., 0].max(axis=1) >= 28, "cold": score[3][..., 0].min(axis=1) <= -10}
    metric, outputs, operator_metric, intervals = {}, {}, {}, {}
    for name, prediction in values.items():
        cal_prediction, scoring = np.split(prediction, [n_cal])
        risk = peak_risk(scoring, cal_actual - cal_prediction, THRESHOLD)
        metric[name] = {group: evaluate_points(score_actual, scoring, risk, THRESHOLD, mask)
                        for group, mask in masks.items()}
        for group, mask in masks.items():
            if mask.any():
                metric[name][group].update(point_scores(score_actual[mask], scoring[mask]))
        outputs[name] = {"hourly_mw": scoring.tolist(), "peak_p90": [risk[0].tolist(), risk[1].tolist()],
                         "peak_exceedance_probability": risk[2].tolist()}
        if matched.any():
            operator_metric[name] = point_scores(score_actual[matched], scoring[matched])
        if name.endswith("_blend") and name != "calendar_blend":
            intervals[name] = paired_interval(score_actual, values["calendar_blend"][n_cal:], scoring)
    if matched.any():
        operator_metric["ieso_average_demand"] = point_scores(score_actual[matched], operator)
    metadata = provenance()
    metadata["source_hashes"].update({
        "apps/energy/recent_benchmark.py": digest(Path(__file__).read_bytes()),
        "apps/energy/demand_experiment.py": digest(Path(__file__).with_name("demand_experiment.py").read_bytes()),
        "apps/energy/boosted_demand.py": digest(Path(existing.__file__).read_bytes())})
    metadata["dependencies"] = {name: existing.version(name) for name in ("numpy", "scipy", "lightgbm", "python-dateutil")}
    return {"schema_version": 1, **metadata, "runtime_seconds": time.perf_counter() - began,
            "training_origins": len(loads), "calibration_origins": n_cal, "scored_origins": len(score_actual),
            "omitted_weather": {"training_2024": early[5], "training_2026": later[5], "calibration": cal[5], "scoring": score[5]},
            "threshold_mw": THRESHOLD, "demand_inputs": history_inputs + recent_inputs, "weather": weather_input,
            "operator": operator_input, "matched_operator_origins": [iso(t) for t, yes in zip(score[2], matched) if yes],
            "operator_metrics": operator_metric, "calibration_metrics": cal_scores, "metrics": metric, "intervals": intervals,
            "weather_selected_in_calibration": weather_best, "tuning": tuning, "models": models,
            "origins": list(map(iso, score[2])), "calibration_dates": list(map(iso, cal[2])),
            "weather_vintages": {"training": early[4] + later[4], "calibration": cal[4], "scoring": score[4]},
            "hourly_actual_mw": score_actual.tolist(), "operator_hourly_mw_matched": operator.tolist(), "forecasts": outputs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("demand-2024", "demand-2026", "weather", "operator", "output"):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    parser.add_argument("--known-weather-gap", type=date.fromisoformat, action="append", default=[])
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new output path")
    result = evaluate(args.demand_2024, args.demand_2026, args.weather, args.operator,
                      set(args.known_weather_gap), args.tune)
    write_json(args.output, result)
    keys = ("training_origins", "calibration_origins", "scored_origins", "metrics", "operator_metrics", "intervals")
    print(json.dumps({key: result[key] for key in keys}, indent=2))


if __name__ == "__main__":
    main()
