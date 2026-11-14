import argparse
import csv
from datetime import datetime
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
import time

import baselines
import boosted_demand as energy
import uncertainty
from ieso import UTC, digest, load_reports, parse_date, parse_origin, parse_report, provenance, write_json


SOURCES = ("ieso.py", "baselines.py", "boosted_demand.py", "uncertainty.py", "final_evaluation.py")
PROTOCOL = {"start_est": "2024-01-01", "end_exclusive_est": "2025-12-31", "origin_hour_est": 8,
            "horizon_hours": 24, "primary_model": "lightgbm", "primary_baseline": "previous_day",
            "primary_metric": "hourly_mae_mw", "refit_or_recalibration": False,
            "bootstrap_samples": 2000, "bootstrap_seed": 42, "block_days": [1, 7, 28],
            "interval_method": "paired origin-day MAE reduction; circular blocks; percentile 95%",
            "availability": "market day plus one 08:00 EST assumed", "point_in_time": False}
LIMITATION = ("Retrospective final-period comparison using revised annual reports and assumed availability, not a live test. "
              "No model reselection/refitting/recalibration. Paired bootstrap intervals are dependence sensitivities, "
              "not proof of a universal edge. Local used markers cannot prove a period was never viewed or prevent bypass by copying files.")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw, object_pairs_hook=unique_object), digest(raw)


def runtime():
    info = provenance()
    return {"source_hashes": {f"apps/energy/{name}": digest(Path(__file__).with_name(name).read_bytes()) for name in SOURCES},
            "dependencies": {name: version(name) for name in ("numpy", "lightgbm", "scipy")},
            **{key: info[key] for key in ("python", "platform", "calendar_zone", "system_zonefile_sha256")}}


def validation_artifacts(report):
    current = runtime()
    expected_sources = {key: value for key, value in current["source_hashes"].items() if not key.endswith("/final_evaluation.py")}
    if (type(report.get("schema_version")) is not int or report["schema_version"] != 1 or
            report.get("mode") != "retrospective_scenario" or report.get("unit") != "MW" or
            report.get("series") != "Ontario Demand" or report.get("horizon_hours") != 24 or
            report.get("task") != "ontario_hourly_demand_uncertainty" or report.get("phase") != "validation" or
            report.get("status") != "ok" or report.get("source_hashes") != expected_sources or
            any(report.get(key) != value for key, value in current.items() if key != "source_hashes")):
        raise ValueError("require a current Day 9 validation report matching source/runtime")
    procedure = report["procedure"]
    calibration = report["calibration"]
    if (procedure["parameters"] != energy.PARAMETERS or procedure["rounds"] != energy.ROUNDS or
            procedure["feature_names"] != list(energy.FEATURES) or procedure["quantiles"] != list(uncertainty.QUANTILES) or
            procedure["coverage_target"] != uncertainty.COVERAGE or
            report["training"]["origin_start_est"] != "2018-01-15" or
            report["training"]["origin_end_exclusive_est"] != "2022-01-01" or
            calibration["origin_start_est"] != "2022-01-01" or calibration["origin_end_exclusive_est"] != "2023-01-01" or
            calibration["rows"] != 8736 or calibration["purged_origin_dates"] != ["2022-12-31"] or
            calibration["latest_label_assumed_available_at_utc"] > calibration["frozen_at_utc"] or
            calibration["frozen_at_utc"] != energy.origin_at(parse_date("2023-01-01")).isoformat() or
            report["evaluation"]["origin_start_est"] != "2023-01-01" or
            report["evaluation"]["origin_end_exclusive_est"] != "2023-12-31" or report["evaluation"]["rows"] != 8736):
        raise ValueError("validation does not match the fixed Day 9 procedure/splits")
    widening = calibration["widening_mw"]
    if type(widening) not in (int, float) or not math.isfinite(widening) or widening < 0:
        raise ValueError("invalid frozen calibration correction")
    models = {"point": {"model_text": report["model_text"], "sha256": report["model_sha256"]},
              **report["quantile_models"]}
    if set(models) != {"point", "0.05", "0.95"}:
        raise ValueError("require point, 0.05 and 0.95 models")
    for model in models.values():
        if digest(model["model_text"].encode()) != model["sha256"]:
            raise ValueError("model text/hash mismatch")
    return {"models": models, "widening_mw": widening, "calibration": calibration,
            "training": report["training"], "quantile_procedure": procedure}


def freeze(validation, paths):
    report, validation_sha = read_json(validation)
    artifacts = validation_artifacts(report)
    if len(paths) != 3:
        raise ValueError("provide complete 2023 warmup, 2024 and 2025 CSVs in order")
    _, inputs = load_reports(paths)
    for path, year in zip(paths, (2023, 2024, 2025)):
        parse_report(Path(path).read_bytes(), expected_year=year)
    if inputs[0]["sha256"] not in {entry["sha256"] for entry in report["inputs"]}:
        raise ValueError("2023 warmup must match the validated snapshot")
    return {"schema_version": 1, "task": "frozen_ontario_final", "protocol": PROTOCOL, "runtime": runtime(),
            "validation_path": str(Path(validation).resolve()), "validation_sha256": validation_sha,
            "inputs": inputs, "artifacts": artifacts, "frozen_at_utc": datetime.now(UTC).isoformat(),
            "source_revision": provenance()["source_revision"], "limitation": LIMITATION}


def verify(procedure):
    fields = {"schema_version", "task", "protocol", "runtime", "validation_path", "validation_sha256", "inputs",
              "artifacts", "frozen_at_utc", "source_revision", "limitation"}
    if (not isinstance(procedure, dict) or set(procedure) != fields or type(procedure["schema_version"]) is not int or
            procedure["schema_version"] != 1 or procedure["task"] != "frozen_ontario_final" or procedure["protocol"] != PROTOCOL):
        raise ValueError("invalid or changed frozen final protocol")
    if procedure["runtime"] != runtime():
        raise ValueError("source or runtime changed after freezing")
    validation, actual_sha = read_json(procedure["validation_path"])
    if actual_sha != procedure["validation_sha256"] or validation_artifacts(validation) != procedure["artifacts"]:
        raise ValueError("validation or artifacts changed after freezing")
    observations, inputs = load_reports([entry["path"] for entry in procedure["inputs"]])
    if inputs != procedure["inputs"]:
        raise ValueError("demand snapshots changed after freezing")
    return observations


def load_models(artifacts):
    import lightgbm as lgb

    models = {}
    for name, artifact in artifacts["models"].items():
        try:
            model = lgb.Booster(model_str=artifact["model_text"])
        except lgb.basic.LightGBMError as error:
            raise ValueError(f"invalid frozen native model: {error}") from error
        if model.feature_name() != list(energy.FEATURES):
            raise ValueError("frozen model features do not match the energy contract")
        models[name] = model
    return models


def paired_intervals(windows):
    import numpy as np

    reductions = np.asarray([[sum(abs(p["models"][baseline]["prediction_mw"] - p["actual_mw"]) -
                                  abs(p["models"]["lightgbm"]["prediction_mw"] - p["actual_mw"])
                                  for p in window["points"]) / len(window["points"])
                              for baseline in baselines.PERIODS] for window in windows], dtype=np.float64)
    if reductions.ndim != 2 or len(reductions) == 0 or not np.isfinite(reductions).all():
        raise ValueError("require finite matched origin-window reductions")
    generator = np.random.default_rng(PROTOCOL["bootstrap_seed"])
    result = {name: {"mean_mae_reduction_mw": float(reductions[:, index].mean()), "block_sensitivity": []}
              for index, name in enumerate(baselines.PERIODS)}
    for block in PROTOCOL["block_days"]:
        record = {"block_days": block, "effective_blocks": len(windows) / block,
                  "samples": PROTOCOL["bootstrap_samples"], "interval_95_mw": None}
        if len(windows) / block < 10:
            for name in result:
                result[name]["block_sensitivity"].append({**record, "reason": "fewer than ten effective blocks"})
            continue
        draws = []
        for start in range(0, PROTOCOL["bootstrap_samples"], 100):
            batch = min(100, PROTOCOL["bootstrap_samples"] - start)
            starts = generator.integers(0, len(windows), size=(batch, math.ceil(len(windows) / block)))
            indices = ((starts[:, :, None] + np.arange(block)) % len(windows)).reshape(batch, -1)[:, :len(windows)]
            draws.append(reductions[indices].mean(axis=1))
        bounds = np.quantile(np.concatenate(draws), [0.025, 0.975], axis=0)
        for index, name in enumerate(result):
            result[name]["block_sensitivity"].append({**record, "interval_95_mw": bounds[:, index].tolist()})
    return result


def evaluate_frozen(procedure, observations, models):
    result = baselines.backtest(observations, parse_date(PROTOCOL["start_est"]), parse_date(PROTOCOL["end_exclusive_est"]))
    rows, evidence = [], []
    for window in result["windows"]:
        prepared, sources = energy.feature_rows(observations, parse_origin(window["origin_utc"]))
        rows.extend(prepared)
        evidence.extend(sources)
    predictions = {name: energy.predict(model, rows) for name, model in models.items()}
    points = [point for window in result["windows"] for point in window["points"]]
    widening = procedure["artifacts"]["widening_mw"]
    for index, point in enumerate(points):
        point["models"]["lightgbm"] = {"prediction_mw": predictions["point"][index], **evidence[index]}
        a, b = predictions["0.05"][index], predictions["0.95"][index]
        low, high = min(a, b), max(a, b)
        point["quantile_crossing"] = a > b
        point["ranges"] = {"raw": [low, high], "calibrated": [max(0, low - widening), high + widening]}
    for window in result["windows"]:
        window["peak_errors_mw"]["lightgbm"] = max(p["models"]["lightgbm"]["prediction_mw"] for p in window["points"]) - window["actual_peak_mw"]
    result["metrics"]["lightgbm"] = baselines.metrics(points, result["windows"], "lightgbm")
    return {**result, "range_metrics": {kind: uncertainty.range_metrics(points, kind) for kind in ("raw", "calibrated")},
            "breakdown": uncertainty.breakdown(points), "paired_mae_reductions": paired_intervals(result["windows"]),
            "crossed_quantile_rows": sum(point["quantile_crossing"] for point in points)}


def run_final(procedure_path, output, *, confirmed=False, retrospective=False):
    began = time.perf_counter()
    procedure_path, output = Path(procedure_path), Path(output)
    marker = procedure_path.with_name(procedure_path.name + ".used")
    if not confirmed or not retrospective:
        raise ValueError("require --confirm-untouched-period and --retrospective")
    if output.exists() or marker.exists():
        raise ValueError("output or used marker exists; preserve the previous attempt")
    if not output.parent.is_dir() or output.resolve() in (marker.resolve(), procedure_path.resolve()):
        raise ValueError("choose an existing output directory and a separate report path")
    procedure, procedure_sha = read_json(procedure_path)
    observations = verify(procedure)
    models = load_models(procedure["artifacts"])
    write_json(marker, {"procedure_sha256": procedure_sha, "status": "final_started", "started_at_utc": datetime.now(UTC).isoformat()})
    result = evaluate_frozen(procedure, observations, models)
    verify(procedure)
    if digest(procedure_path.read_bytes()) != procedure_sha:
        raise ValueError("procedure changed during the final attempt")
    report = {"schema_version": 1, "status": "ok", "phase": "holdout", "task": "ontario_demand_final",
              "final_period_consumed": True, "procedure_sha256": procedure_sha, "protocol": PROTOCOL,
              "frozen_procedure_path": str(procedure_path.resolve()), "runtime": procedure["runtime"],
              "inputs": procedure["inputs"], "unit": "MW", "limitation": LIMITATION + " " + uncertainty.LIMITATION,
              "runtime_seconds": time.perf_counter() - began, **result}
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description="Freeze Day 9 artifacts, then make one explicit final energy attempt.")
    commands = parser.add_subparsers(dest="command", required=True)
    frozen = commands.add_parser("freeze")
    frozen.add_argument("--validation", type=Path, required=True)
    frozen.add_argument("--csv", type=Path, nargs="+", required=True)
    frozen.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--procedure", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--confirm-untouched-period", action="store_true")
    run.add_argument("--retrospective", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            if args.output.exists():
                raise ValueError("procedure already exists; choose a new path")
            write_json(args.output, freeze(args.validation, args.csv))
            print(f"Frozen saved models and correction; no final scores computed. Saved {args.output}")
        else:
            report = run_final(args.procedure, args.output, confirmed=args.confirm_untouched_period, retrospective=args.retrospective)
            print(json.dumps({key: report[key] for key in ("counts", "metrics", "range_metrics", "paired_mae_reductions", "runtime_seconds")}, indent=2))
    except (ValueError, KeyError, TypeError, OSError, csv.Error, OverflowError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
