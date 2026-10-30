import argparse
import csv
from datetime import timedelta
import json
import math
from pathlib import Path
import sys
import time

from ieso import (EST, HOUR, assumed_available_at, calendar_fields, iso, load_reports,
                  market_midnight, parse_date, parse_origin, provenance, write_json)


PERIODS = {"previous_day": 24, "previous_week": 168}
LIMITATION = ("Retrospective revised-data scenario, not a point-in-time or live forecast. "
              "Assume each market day's values become available at 08:00 EST the next day; "
              "this is a research assumption, not an IESO publication guarantee. Revisions remain in the inputs.")


def seasonal_value(observations, origin, target, period):
    if type(period) is not int or period <= 0:
        raise ValueError("seasonal period must be a positive integer number of hours")
    source, periods = target - period * HOUR, 1
    while assumed_available_at(source) > origin:
        source -= period * HOUR
        periods += 1
    if source not in observations:
        raise ValueError(f"missing available seasonal source {iso(source)}; no source substitution applied")
    return {"prediction_mw": observations[source].demand_mw, "source_start_utc": iso(source),
            "assumed_available_at_utc": iso(assumed_available_at(source)), "lag_periods": periods}


def forecast(observations, origin):
    origin = parse_origin(origin.isoformat())
    points = []
    for step in range(24):
        target = origin + step * HOUR
        points.append({**calendar_fields(target), "lead_hour": step + 1,
                       "models": {name: seasonal_value(observations, origin, target, period)
                                  for name, period in PERIODS.items()}})
    return {"origin_utc": iso(origin), "origin_est": origin.astimezone(EST).isoformat(), "points": points}


def metrics(points, windows, model):
    errors = [point["models"][model]["prediction_mw"] - point["actual_mw"] for point in points]
    absolute = math.fsum(abs(error) for error in errors)
    total = math.fsum(point["actual_mw"] for point in points)
    return {"hourly_mae_mw": absolute / len(errors),
            "hourly_rmse_mw": math.sqrt(math.fsum(error * error for error in errors) / len(errors)),
            "hourly_bias_mw": math.fsum(errors) / len(errors),
            "wape_percent": 100 * absolute / total if total else None,
            "window_peak_mae_mw": math.fsum(abs(window["peak_errors_mw"][model]) for window in windows) / len(windows),
            "recursive_hours": sum(point["models"][model]["lag_periods"] > 1 for point in points)}


def backtest(observations, start, end):
    if start >= end:
        raise ValueError("require start < end; end is exclusive")
    windows, points = [], []
    day = start
    while day < end:
        origin = market_midnight(day) + 8 * HOUR
        window = forecast(observations, origin)
        for point in window["points"]:
            target = parse_origin(point["start_utc"])
            if target not in observations:
                raise ValueError(f"missing target {point['start_utc']}; requested origins were not trimmed")
            point["actual_mw"] = observations[target].demand_mw
        actual_peak = max(point["actual_mw"] for point in window["points"])
        window["actual_peak_mw"] = actual_peak
        window["peak_errors_mw"] = {
            model: max(point["models"][model]["prediction_mw"] for point in window["points"]) - actual_peak
            for model in PERIODS}
        points.extend(window["points"])
        windows.append(window)
        day += timedelta(days=1)
    return {"windows": windows, "counts": {"origins": len(windows), "hourly_targets": len(points)},
            "metrics": {model: metrics(points, windows, model) for model in PERIODS}}


def evaluate(paths, *, start=None, end=None, origin=None):
    began = time.perf_counter()
    if ((origin is not None and (start is not None or end is not None)) or
            (origin is None and (start is None or end is None))):
        raise ValueError("choose either an origin or both start and end")
    observations, inputs = load_reports(paths)
    report = {"schema_version": 1, "status": "ok", "task": "ontario_hourly_demand_baselines",
              "mode": "retrospective_scenario", "limitation": LIMITATION,
              "series": "Ontario Demand", "unit": "MW", "horizon_hours": 24,
              "availability": {"rule": "market_day_plus_one_0800_EST_assumed", "point_in_time": False},
              "procedure": {"period_hours": PERIODS, "unavailable_lags": "recursive seasonal repetition until available",
                            "target_interval": "[origin, origin + 24 hours)",
                            "peak_definition": "maximum of each 24-hour forecast window, not Toronto calendar day",
                            "missing_rows": "fail, never trim or fill", "uncertainty": "not estimated by this baseline runner"},
              "inputs": inputs, **provenance()}
    if origin is not None:
        report.update(run="forecast", **forecast(observations, origin))
    else:
        report.update(run="backtest", start=start.isoformat(), end_exclusive=end.isoformat(),
                      origin_hour_est=8, **backtest(observations, start, end))
    report["runtime_seconds"] = time.perf_counter() - began
    return report


def main():
    parser = argparse.ArgumentParser(description="Retrospective Ontario 24-hour seasonal demand baselines (MW).")
    parser.add_argument("mode", choices=("forecast", "backtest"))
    parser.add_argument("--csv", type=Path, nargs="+", required=True, help="raw IESO files in chronological order")
    parser.add_argument("--origin", help="forecast interval start with UTC offset")
    parser.add_argument("--start", help="first backtest origin date in EST, at 08:00")
    parser.add_argument("--end", help="exclusive backtest origin date in EST")
    parser.add_argument("--retrospective", action="store_true", help="acknowledge revised data and assumed availability")
    parser.add_argument("--output", type=Path, required=True, help="new JSON report path")
    args = parser.parse_args()
    try:
        if not args.retrospective:
            raise ValueError("pass --retrospective to acknowledge revised data and assumed availability")
        if args.output.exists():
            raise ValueError("output exists; use a new report path")
        if args.mode == "forecast":
            if not args.origin or args.start or args.end:
                raise ValueError("forecast requires --origin, without --start/--end")
            report = evaluate(args.csv, origin=parse_origin(args.origin))
        else:
            if args.origin or not args.start or not args.end:
                raise ValueError("backtest requires --start and --end, without --origin")
            report = evaluate(args.csv, start=parse_date(args.start), end=parse_date(args.end))
        write_json(args.output, report)
    except (ValueError, OSError, csv.Error, OverflowError) as error:
        parser.exit(1, f"error: {error}\n")
    summary = {key: report[key] for key in ("mode", "run", "runtime_seconds")}
    if "metrics" in report:
        summary.update(counts=report["counts"], metrics=report["metrics"])
    print(json.dumps(summary, indent=2))
    print(LIMITATION, file=sys.stderr)


if __name__ == "__main__":
    main()
