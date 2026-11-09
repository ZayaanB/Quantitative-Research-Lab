import argparse
import csv
from datetime import date, datetime
import json
import math
from pathlib import Path
import sys
import time

import baselines
import boosted_demand as energy
from ieso import assumed_available_at, digest, iso, load_reports, market_midnight, parse_date, parse_origin, write_json


QUANTILES = (0.05, 0.95)
COVERAGE = 0.9
MIN_CALIBRATION_ORIGINS = 30
LIMITATION = ("Chronological inspected validation on revised data with assumed availability. "
              "Hourly observations are dependent: no exchangeability-based, per-hour/season, simultaneous-24h "
              "or live coverage guarantee. The 90% target is marginal, not point-price accuracy.")


def partition(windows, score_start):
    calibration, evaluation, purged = [], [], []
    cutoff, boundary = energy.origin_at(score_start), market_midnight(score_start)
    for window in windows:
        origin = parse_origin(window["origin_utc"])
        if origin >= cutoff:
            evaluation.append(window)
        elif any(parse_origin(point["end_utc"]) > boundary or
                 assumed_available_at(parse_origin(point["start_utc"])) > cutoff for point in window["points"]):
            purged.append(window["origin_est"][:10])
        else:
            calibration.append(window)
    if len(calibration) < MIN_CALIBRATION_ORIGINS or not evaluation:
        raise ValueError("require at least 30 complete past calibration origins and nonempty evaluation")
    return calibration, evaluation, purged


def correction(points, cutoff):
    scores = []
    for point in points:
        available = assumed_available_at(parse_origin(point["start_utc"]))
        if available > cutoff or parse_origin(point["end_utc"]) > cutoff:
            raise ValueError("calibration label is not available at evaluation origin")
        low, high = point["ranges"]["raw"]
        actual = point["actual_mw"]
        if not all(math.isfinite(value) for value in (low, high, actual)) or low < 0 or low > high or actual < 0:
            raise ValueError("invalid calibration demand or bounds")
        scores.append(max(low - actual, actual - high, 0.0))
    rank = math.ceil((len(scores) + 1) * COVERAGE)
    if not scores or rank > len(scores):
        raise ValueError("too few calibration scores for the finite-sample quantile rank")
    return {"rows": len(scores), "rank_one_based": rank, "widening_mw": sorted(scores)[rank - 1],
            "latest_label_assumed_available_at_utc": max(iso(assumed_available_at(parse_origin(p["start_utc"]))) for p in points),
            "frozen_at_utc": iso(cutoff)}


def pinball(actual, predicted, quantile):
    error = actual - predicted
    return max(quantile * error, (quantile - 1) * error)


def range_metrics(points, kind):
    if not points:
        raise ValueError("cannot score an empty interval group")
    hits, widths, lower_losses, upper_losses, interval_scores = [], [], [], [], []
    for point in points:
        low, high = point["ranges"][kind]
        actual = point["actual_mw"]
        if not all(math.isfinite(value) for value in (low, high, actual)) or not 0 <= low <= high or actual < 0:
            raise ValueError("invalid scored range or demand")
        hits.append(low <= actual <= high)
        widths.append(high - low)
        lower_losses.append(pinball(actual, low, QUANTILES[0]))
        upper_losses.append(pinball(actual, high, QUANTILES[1]))
        interval_scores.append(high - low + 2 / (1 - COVERAGE) * max(low - actual, actual - high, 0))
    count = len(points)
    lower, upper = math.fsum(lower_losses) / count, math.fsum(upper_losses) / count
    return {"rows": count, "coverage": sum(hits) / count, "mean_width_mw": math.fsum(widths) / count,
            "pinball_lower_mw": lower, "pinball_upper_mw": upper, "mean_pinball_mw": (lower + upper) / 2,
            "mean_interval_score_mw": math.fsum(interval_scores) / count}


def breakdown(points):
    groups = {"season": {}, "local_hour": {}}
    for point in points:
        local = datetime.fromisoformat(point["local_start"])
        season = ("winter", "spring", "summer", "autumn")[(local.month % 12) // 3]
        for name, key in (("season", season), ("local_hour", f"{local.hour:02d}")):
            groups[name].setdefault(key, []).append(point)
    return {name: {key: {kind: range_metrics(group, kind) for kind in ("raw", "calibrated")}
                   for key, group in sorted(values.items())} for name, values in groups.items()}


def evaluate(paths, train_start, train_end, calibration_start, score_start, end):
    began = time.perf_counter()
    if not train_start < train_end <= calibration_start < score_start < end <= energy.DEVELOPMENT_END:
        raise ValueError("require chronological train/calibration/evaluation periods ending by 2023-12-31 exclusive")
    point_report = energy.evaluate(paths, train_start, train_end, calibration_start, end)
    observations, inputs = load_reports(paths)
    if inputs != point_report["inputs"]:
        raise ValueError("inputs changed during evaluation")
    training, labels, details = energy.training_data(observations, train_start, train_end)
    if details != point_report["training"]:
        raise ValueError("training procedure changed during evaluation")
    calibration, evaluation, purged = partition(point_report["windows"], score_start)
    windows = calibration + evaluation
    rows = []
    for window in windows:
        features, _ = energy.feature_rows(observations, parse_origin(window["origin_utc"]))
        rows.extend(features)
    predictions, models = [], {}
    for quantile in QUANTILES:
        model = energy.fit_model(training, labels, quantile=quantile)
        predictions.append(energy.predict(model, rows))
        model_text = model.model_to_string()
        models[str(quantile)] = {"model_text": model_text, "sha256": digest(model_text.encode())}
    points = [point for window in windows for point in window["points"]]
    if any(len(values) != len(points) for values in predictions):
        raise ValueError("quantile predictions do not match requested targets")
    for point, lower, upper in zip(points, *predictions):
        point["ranges"] = {"raw": [min(lower, upper), max(lower, upper)]}
        point["quantile_crossing"] = lower > upper
    cal_points = [point for window in calibration for point in window["points"]]
    score_points = [point for window in evaluation for point in window["points"]]
    adjustment = correction(cal_points, energy.origin_at(score_start))
    for point in score_points:
        low, high = point["ranges"]["raw"]
        point["ranges"]["calibrated"] = [max(0, low - adjustment["widening_mw"]), high + adjustment["widening_mw"]]
    metadata = {key: point_report[key] for key in ("inputs", "source_revision", "source_hashes", "dependencies", "python",
                "platform", "calendar_zone", "system_zonefile_sha256", "training", "model_text", "model_sha256")}
    metadata["source_hashes"]["apps/energy/uncertainty.py"] = digest(Path(__file__).read_bytes())
    return {"schema_version": 1, "status": "ok", "phase": "validation", "mode": "retrospective_scenario",
        "task": "ontario_hourly_demand_uncertainty", "unit": "MW", "series": "Ontario Demand", "horizon_hours": 24,
        "limitation": LIMITATION + " " + baselines.LIMITATION, **metadata,
        "procedure": {"coverage_target": COVERAGE, "quantiles": QUANTILES, "parameters": energy.PARAMETERS,
            "quantile_objective": "quantile; alpha is each requested quantile", "rounds": energy.ROUNDS,
            "feature_names": energy.FEATURES, "minimum_calibration_origins": MIN_CALIBRATION_ORIGINS,
            "calibration": "one pooled widening-only split CQR correction, frozen before evaluation",
            "rank": "ceil((n+1)*coverage); scores=max(lower-y,y-upper,0)",
            "crossings": "sort bounds and disclose counts", "nonnegative_lower": "max(0, lower-widening)",
            "availability": "market day plus one 08:00 EST assumed", "point_in_time": False,
            "season": "target Toronto calendar month: DJF/MAM/JJA/SON", "hour": "target Toronto interval start hour",
            "weather": "excluded", "refits_or_tuning": "none"},
        "calibration": {"origin_start_est": calibration_start.isoformat(), "origin_end_exclusive_est": score_start.isoformat(),
            "origins": len(calibration), "purged_origin_dates": purged,
            "crossed_quantile_rows": sum(p["quantile_crossing"] for p in cal_points), **adjustment},
        "evaluation": {"origin_start_est": score_start.isoformat(), "origin_end_exclusive_est": end.isoformat(),
            "origins": len(evaluation), "rows": len(score_points),
            "crossed_quantile_rows": sum(p["quantile_crossing"] for p in score_points)},
        "range_metrics": {kind: range_metrics(score_points, kind) for kind in ("raw", "calibrated")},
        "point_metrics": {name: baselines.metrics(score_points, evaluation, name) for name in (*baselines.PERIODS, "lightgbm")},
        "breakdown": breakdown(score_points), "quantile_models": models,
        "calibration_windows": calibration, "windows": evaluation, "runtime_seconds": time.perf_counter() - began}


def main():
    parser = argparse.ArgumentParser(description="Past-only 90% demand ranges: fixed quantiles, calibration, then evaluation.")
    parser.add_argument("--csv", nargs="+", type=Path, required=True)
    parser.add_argument("--train-start", type=parse_date, default=date(2018, 1, 15))
    parser.add_argument("--train-end", type=parse_date, default=date(2022, 1, 1))
    parser.add_argument("--calibration-start", type=parse_date, default=date(2022, 1, 1))
    parser.add_argument("--start", type=parse_date, default=date(2023, 1, 1))
    parser.add_argument("--end", type=parse_date, default=energy.DEVELOPMENT_END)
    parser.add_argument("--retrospective", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if not args.retrospective:
            raise ValueError("pass --retrospective to acknowledge revised data and assumed availability")
        if args.output.exists():
            raise ValueError("output exists; choose a new report path")
        report = evaluate(args.csv, args.train_start, args.train_end, args.calibration_start, args.start, args.end)
        write_json(args.output, report)
    except (ValueError, OSError, csv.Error, OverflowError) as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps({"calibration": report["calibration"], "evaluation": report["evaluation"],
                      "range_metrics": report["range_metrics"], "runtime_seconds": report["runtime_seconds"]}, indent=2))
    print(LIMITATION, file=sys.stderr)


if __name__ == "__main__":
    main()
