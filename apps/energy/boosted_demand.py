import argparse
import calendar
import csv
from datetime import date, datetime, timedelta
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
import time

import baselines
from ieso import (EST, HOUR, assumed_available_at, calendar_fields, digest, iso, load_reports,
                  market_midnight, parse_date, provenance, write_json)


FEATURES = ("lead_hour", "market_hour", "local_hour", "local_weekday", "local_month",
            "local_year_day_sin", "local_year_day_cos", "local_weekend", "local_utc_offset_hours",
            "daily_seasonal_mw", "daily_lag_periods", "seasonal_48_mw", "weekly_seasonal_mw", "seasonal_336_mw",
            "recent_day_mean_mw", "recent_day_min_mw", "recent_day_max_mw", "recent_week_mean_mw",
            "recent_day_minus_prior_mean_mw")
PARAMETERS = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6,
              "min_data_in_leaf": 40, "lambda_l2": 5, "num_threads": 1, "device_type": "cpu",
              "seed": 42, "deterministic": True, "force_col_wise": True, "verbosity": -1}
ROUNDS = 200
MIN_TRAIN_ORIGINS = 30
DEVELOPMENT_END = date(2023, 12, 31)


def origin_at(day):
    return market_midnight(day) + 8 * HOUR


def feature_rows(observations, origin):
    if origin != origin_at(origin.astimezone(EST).date()):
        raise ValueError("energy model origins must be at 08:00 fixed EST")
    last = market_midnight(origin.astimezone(EST).date()) - HOUR
    recent_starts = [last - offset * HOUR for offset in reversed(range(168))]
    if any(start not in observations or assumed_available_at(start) > origin for start in recent_starts):
        raise ValueError("missing or unavailable recent demand history; no filling applied")
    recent = [observations[start].demand_mw for start in recent_starts]
    day_mean = math.fsum(recent[-24:]) / 24
    summaries = [day_mean, min(recent[-24:]), max(recent[-24:]), math.fsum(recent) / 168,
                 day_mean - math.fsum(recent[-48:-24]) / 24]
    rows, evidence = [], []
    for step in range(24):
        target = origin + step * HOUR
        fields = calendar_fields(target)
        local = datetime.fromisoformat(fields["local_start"])
        angle = 2 * math.pi * (local.timetuple().tm_yday - 1) / (366 if calendar.isleap(local.year) else 365)
        lags = {period: baselines.seasonal_value(observations, origin, target, period) for period in (24, 48, 168, 336)}
        rows.append([step + 1, target.astimezone(EST).hour, local.hour, local.weekday(), local.month,
                     math.sin(angle), math.cos(angle), int(local.weekday() >= 5), local.utcoffset().total_seconds() / 3600,
                     lags[24]["prediction_mw"], lags[24]["lag_periods"], lags[48]["prediction_mw"],
                     lags[168]["prediction_mw"], lags[336]["prediction_mw"], *summaries])
        evidence.append({"lag_sources_utc": {str(period): lag["source_start_utc"] for period, lag in lags.items()},
                         "max_feature_available_at_utc": max(iso(assumed_available_at(last)),
                             *(lag["assumed_available_at_utc"] for lag in lags.values()))})
    if any(len(row) != len(FEATURES) or not all(math.isfinite(value) for value in row) for row in rows):
        raise ValueError("energy features must be finite and match the feature contract")
    return rows, evidence


def training_data(observations, start, end):
    if start >= end:
        raise ValueError("training start must precede training end")
    cutoff, boundary = origin_at(end), market_midnight(end)
    training, labels, origins, purged = [], [], [], []
    day = start
    latest_available = None
    while day < end:
        origin = origin_at(day)
        targets = [origin + step * HOUR for step in range(24)]
        if any(target + HOUR > boundary or assumed_available_at(target) > cutoff for target in targets):
            purged.append(day.isoformat())
        else:
            if any(target not in observations for target in targets):
                raise ValueError("missing training target; requested training period was not shortened")
            rows, _ = feature_rows(observations, origin)
            training.extend(rows)
            labels.extend(observations[target].demand_mw for target in targets)
            origins.append(origin)
            latest_available = max(assumed_available_at(target) for target in targets)
        day += timedelta(days=1)
    if len(origins) < MIN_TRAIN_ORIGINS:
        raise ValueError(f"require at least {MIN_TRAIN_ORIGINS} complete eligible training origins")
    return training, labels, {"origin_start_est": start.isoformat(), "origin_end_exclusive_est": end.isoformat(),
        "origins": len(origins), "rows": len(labels), "purged_origin_dates": purged,
        "first_origin_utc": iso(origins[0]), "last_origin_utc": iso(origins[-1]),
        "last_target_end_utc": iso(origins[-1] + 24 * HOUR),
        "last_label_assumed_available_at_utc": iso(latest_available), "fit_origin_utc": iso(cutoff),
        "boundary_utc": iso(boundary), "rule": "whole windows; labels end by boundary and are available at fit origin"}


def fit_model(training, labels, *, quantile=None):
    if quantile is not None and (type(quantile) not in (int, float) or not 0 < quantile < 1):
        raise ValueError("quantile must be finite and strictly between zero and one")
    try:
        import lightgbm as lgb
        import numpy as np
    except ImportError as error:
        raise ValueError("install apps/energy/requirements.txt in .venv for the native energy model") from error
    x, y = np.asarray(training, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    if (y.ndim != 1 or x.ndim != 2 or x.shape != (len(y), len(FEATURES)) or
            len(y) < MIN_TRAIN_ORIGINS * 24 or not np.isfinite(x).all() or
            not np.isfinite(y).all() or (y < 0).any()):
        raise ValueError("invalid native energy training matrix or labels")
    parameters = dict(PARAMETERS)
    if quantile is not None:
        parameters.update(objective="quantile", alpha=quantile)
    try:
        model = lgb.train(parameters, lgb.Dataset(x, label=y, feature_name=list(FEATURES)), num_boost_round=ROUNDS)
    except lgb.basic.LightGBMError as error:
        raise ValueError(f"energy model fit failed; no fallback: {error}") from error
    return model


def predict(model, rows):
    import lightgbm as lgb
    import numpy as np

    x = np.asarray(rows, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(FEATURES) or not len(x) or not np.isfinite(x).all():
        raise ValueError("invalid native energy prediction matrix")
    try:
        result = np.asarray(model.predict(x, num_threads=1), dtype=np.float64)
    except lgb.basic.LightGBMError as error:
        raise ValueError(f"energy prediction failed; no fallback: {error}") from error
    if result.shape != (len(x),) or not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("energy model returned invalid or negative predictions; no fallback")
    return result.tolist()


def evaluate(paths, train_start, train_end, start, end):
    began = time.perf_counter()
    if not train_start < train_end <= start < end <= DEVELOPMENT_END:
        raise ValueError("require training start < training end <= validation start < end <= 2023-12-31 (exclusive)")
    observations, inputs = load_reports(paths)
    training, labels, training_details = training_data(observations, train_start, train_end)
    result = baselines.backtest(observations, start, end)
    rows, feature_evidence = [], []
    for window in result["windows"]:
        prepared, evidence = feature_rows(observations, origin_at(parse_date(window["origin_est"][:10])))
        rows.extend(prepared)
        feature_evidence.extend(evidence)
    fit_began = time.perf_counter()
    model = fit_model(training, labels)
    fit_seconds = time.perf_counter() - fit_began
    predicted = predict(model, rows)
    points = [point for window in result["windows"] for point in window["points"]]
    for point, value, evidence in zip(points, predicted, feature_evidence):
        point["models"]["lightgbm"] = {"prediction_mw": value, **evidence}
    for window in result["windows"]:
        window["peak_errors_mw"]["lightgbm"] = max(point["models"]["lightgbm"]["prediction_mw"]
                                                    for point in window["points"]) - window["actual_peak_mw"]
    result["metrics"]["lightgbm"] = baselines.metrics(points, result["windows"], "lightgbm")
    model_text = model.model_to_string()
    source = provenance()
    source["source_hashes"]["apps/energy/boosted_demand.py"] = digest(Path(__file__).read_bytes())
    source["dependencies"] = {name: version(name) for name in ("numpy", "lightgbm", "scipy")}
    return {"schema_version": 1, "status": "ok", "phase": "validation", "mode": "retrospective_scenario",
        "task": "ontario_hourly_demand_lightgbm", "series": "Ontario Demand", "unit": "MW", "horizon_hours": 24,
        "limitation": baselines.LIMITATION, "inputs": inputs, **source,
        "start_est": start.isoformat(), "end_exclusive_est": end.isoformat(), "origin_hour_est": 8,
        "procedure": {"feature_names": FEATURES, "parameters": PARAMETERS, "rounds": ROUNDS,
            "fit_schedule": "once before validation; no tuning, early stopping, calibration or validation labels in fit",
            "target": "hourly MW; direct 24-horizon rows pooled into one regressor",
            "availability": "market day plus one 08:00 EST assumed; seasonal recursion as in Day 7",
            "point_in_time": False, "weather": "excluded", "uncertainty": "not estimated; planned Day 9",
            "peak_definition": "maximum over 24-hour forecast window, not calendar-day peak",
            "minimum_training_origins": MIN_TRAIN_ORIGINS},
        "training": training_details, "model_text": model_text, "model_sha256": digest(model_text.encode()),
        "fit_seconds": fit_seconds, "runtime_seconds": time.perf_counter() - began, **result}


def main():
    parser = argparse.ArgumentParser(description="Fixed CPU LightGBM Ontario demand validation; no final-period scoring.")
    parser.add_argument("--csv", type=Path, nargs="+", required=True)
    parser.add_argument("--train-start", type=parse_date, default=date(2018, 1, 15))
    parser.add_argument("--train-end", type=parse_date, default=date(2022, 1, 1))
    parser.add_argument("--start", type=parse_date, default=date(2022, 1, 1))
    parser.add_argument("--end", type=parse_date, default=DEVELOPMENT_END)
    parser.add_argument("--retrospective", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if not args.retrospective:
            raise ValueError("pass --retrospective to acknowledge revised data and assumed availability")
        if args.output.exists():
            raise ValueError("output exists; use a new report path")
        report = evaluate(args.csv, args.train_start, args.train_end, args.start, args.end)
        write_json(args.output, report)
    except (ValueError, OSError, csv.Error, OverflowError) as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps({key: report[key] for key in ("counts", "metrics", "fit_seconds", "runtime_seconds")}, indent=2))
    print(baselines.LIMITATION, file=sys.stderr)


if __name__ == "__main__":
    main()
