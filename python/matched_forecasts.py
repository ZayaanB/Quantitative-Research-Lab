import argparse
import csv
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time

from benchmark_forecasts import (HORIZONS, bootstrap_sensitivity, iso_date, read_manifest,
                                run_benchmark, sha256, summarize, unique_object)
from ensemble_forecasts import point_metrics
from forecast_quality import inspect_asset


def reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def positive_number(value, context):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{context}: require a positive finite price")
    return value


def read_reference(path, config, digest, histories, column):
    raw = path.read_bytes()
    report = json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)
    if (not isinstance(report, dict) or type(report.get("schema_version")) is not int or report.get("schema_version") != 1 or
            report.get("phase") != "validation" or report.get("manifest_sha256") != digest or
            report.get("config") != config):
        raise ValueError(f"{path.name}: require a validation report for the identical manifest")
    groups = {}
    expected = {(asset["symbol"], horizon) for asset in config["assets"] for horizon in HORIZONS}
    if not isinstance(report.get("results"), list):
        raise ValueError("reference results must be a list")
    for group in report["results"]:
        symbol, horizon = group["symbol"], group["horizon_days"]
        key = (symbol, horizon)
        if type(horizon) is not int or key not in expected or key in groups:
            raise ValueError("unexpected or duplicate reference group")
        points = group["points"]
        if (not isinstance(points, list) or not points or type(group.get("forecasts")) is not int or
                group["forecasts"] != len(points)):
            raise ValueError(f"{symbol} {horizon}d: invalid reference forecast count")
        dates, prices = histories[symbol]
        positions = {day: index for index, day in enumerate(dates)}
        cleaned, indices = [], []
        for point in points:
            origin, target = iso_date(point["origin_date"]), iso_date(point["target_date"])
            if origin not in positions or target not in positions:
                raise ValueError("reference date is absent from validation history")
            index = positions[origin]
            if (index < config["lookback_days"] or positions[target] != index + horizon or
                    origin < config["validation_start"] or target >= config["holdout_start"]):
                raise ValueError("reference violates lookback, horizon or validation boundaries")
            for name, expected_price in (("current_price", prices[index]), ("actual_price", prices[index + horizon])):
                if positive_number(point[name], name) != expected_price:
                    raise ValueError(f"reference {name} does not match hashed history")
            if "latest_price_forecast" in point and positive_number(
                    point["latest_price_forecast"], "latest_price_forecast") != prices[index]:
                raise ValueError("reference latest-price baseline does not match history")
            cutoff = point.get("last_training_target_index", index)
            if type(cutoff) is not int or not 0 <= cutoff <= index:
                raise ValueError("reference training target crosses its origin")
            cleaned.append({"origin_date": origin, "target_date": target,
                            "current_price": prices[index], "actual_price": prices[index + horizon],
                            "latest_price_forecast": prices[index],
                            "forecast_price": positive_number(point[column], column)})
            indices.append(index)
        if indices != list(range(indices[0], len(dates) - horizon, horizon)):
            raise ValueError("reference grid has duplicate, unordered, missing or overlapping rows")
        groups[key] = cleaned
    if set(groups) != expected:
        raise ValueError("reference must include every manifest asset and 1/5/20-session horizon")
    return groups, {"file": path.name, "sha256": sha256(raw), "prediction_column": column,
                    "model": report.get("model"), "dependencies": report.get("dependencies"),
                    "source_hashes": {key: value for key, value in report.items() if "sha256" in key or key == "source_hashes"}}


def evaluate(manifest, references=(), engine=Path("build/mcprice"), column="forecast_price", quality_only=False):
    began = time.perf_counter()
    config, digest = read_manifest(manifest)
    histories, quality = {}, []
    for asset in config["assets"]:
        loaded, inspection = inspect_asset(asset, manifest.parent, config["holdout_start"])
        quality.append(inspection)
        if loaded is not None and inspection["status"] == "ok":
            header, rows, metadata = loaded
            price_index = next(csv.reader([header])).index(metadata["price_column"])
            retained = [(day, float(next(csv.reader([line]))[price_index])) for day, line in rows
                        if day < config["holdout_start"] and day <= config["test_end"]]
            dates, prices = zip(*retained)
            histories[asset["symbol"]] = (dates, prices)
    report = {"schema_version": 1, "phase": "validation", "manifest_sha256": digest, "config": config,
              "status": "blocked" if any(item["status"] == "error" for item in quality) else "ok",
              "quality": quality, "results": [], "python_version": platform.python_version(),
              "platform": platform.platform(),
              "source_hashes": {name: sha256(Path(__file__).with_name(name).read_bytes()) for name in
                  ("matched_forecasts.py", "forecast_quality.py", "benchmark_forecasts.py", "ensemble_forecasts.py")}}
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                                  text=True, capture_output=True, check=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    report["source_revision"] = revision
    if report["status"] == "blocked" or quality_only:
        report["runtime_seconds"] = time.perf_counter() - began
        return report
    if not references:
        raise ValueError("provide at least one reference report or use --quality-only")
    loaded_references = [read_reference(path, config, digest, histories, column) for path in references]
    if len({identity["sha256"] for _, identity in loaded_references}) != len(loaded_references):
        raise ValueError("duplicate reference reports")
    canonical = loaded_references[0][0]
    for groups, _ in loaded_references[1:]:
        for key, points in canonical.items():
            if [(p["origin_date"], p["target_date"]) for p in points] != [
                    (p["origin_date"], p["target_date"]) for p in groups[key]]:
                raise ValueError("reference origin/target grids differ; no rows were silently intersected")
    gbm = run_benchmark(manifest, engine, "validation", dense=True)
    if gbm["manifest_sha256"] != digest:
        raise ValueError("manifest changed during comparison")
    report["engine_sha256"] = gbm["engine_sha256"]
    report["references"] = [identity for _, identity in loaded_references]
    settings = config.get("bootstrap", {"samples": 2000, "seed": 42, "block_sizes": [1, 2, 4]})
    ratios = {name: [] for name in ("gbm", *(f"reference_{index}" for index in range(len(references))))}
    ineligible = set()
    for group in gbm["results"]:
        key = (group["symbol"], group["horizon_days"])
        reference = canonical[key]
        index = {(point["origin_date"], point["target_date"]): point for point in group["points"]}
        matched = []
        for point in reference:
            actual = index.get((point["origin_date"], point["target_date"]))
            if actual is None or any(actual[name] != point[name] for name in ("current_price", "actual_price")):
                raise ValueError("GBM rows do not match reference dates and observed prices")
            matched.append(actual)
        points = [{**point, "gbm_price": actual["forecast_price"],
                   **{f"reference_{i}": groups[key][j]["forecast_price"] for i, (groups, _) in enumerate(loaded_references)}}
                  for j, (point, actual) in enumerate(zip(reference, matched))]
        metrics = {"latest_price": point_metrics(points, "current_price"), "gbm": point_metrics(points, "gbm_price")}
        metrics.update({f"reference_{i}": point_metrics(points, f"reference_{i}") for i in range(len(references))})
        baseline = metrics["latest_price"]["mae"]
        bootstrap = {"gbm": bootstrap_sensitivity(matched, settings)}
        for i, (groups, _) in enumerate(loaded_references):
            bootstrap[f"reference_{i}"] = bootstrap_sensitivity(groups[key], settings)
        for name in ratios:
            error = metrics[name]["mae"]
            if baseline:
                ratio = error / baseline
                if not math.isfinite(ratio):
                    raise ValueError("relative MAE exceeds finite range")
                ratios[name].append(ratio)
            elif error:
                ineligible.add(name)
        report["results"].append({"symbol": key[0], "horizon_days": key[1], "forecasts": len(points),
                                  "dense_gbm_rows_not_scored": len(group["points"]) - len(points),
                                  "metrics": metrics, "always_up_accuracy": statistics.mean(
                                      p["actual_price"] > p["current_price"] for p in points),
                                  "gbm_distribution": summarize(matched), "bootstrap_sensitivity": bootstrap,
                                  "points": points})
    if len(report["results"]) != len(canonical):
        raise ValueError("GBM result groups do not cover the reference groups")
    report["mean_relative_mae"] = {name: statistics.mean(values) if values and name not in ineligible else None
                                   for name, values in ratios.items()}
    report["matching_rule"] = "First reference defines the frozen grid; other references must match exactly; GBM generated densely."
    report["limitations"] = ["Validation only; not model selection or an untouched final test.",
                              "The first reference declares warmup/start; its predictions are not refitted or independently certified.",
                              "Reported training cutoffs are checked when present; hashes do not prove absence of leakage.",
                              "Bootstrap blocks are heuristic and not multiple-comparison corrected.",
                              "No model probabilities/ranges are inferred from point forecasts."]
    report["runtime_seconds"] = time.perf_counter() - began
    return report


def main():
    parser = argparse.ArgumentParser(description="Validate history and compare frozen point reports on exactly matched dates")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", default=[])
    parser.add_argument("--prediction-column", default="forecast_price")
    parser.add_argument("--engine", type=Path, default=Path("build/mcprice"))
    parser.add_argument("--quality-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.quality_only and args.reference:
        parser.error("--quality-only does not accept reference reports")
    try:
        if args.output.exists():
            raise ValueError("output already exists; choose a new report path")
        report = evaluate(args.manifest.resolve(), [path.resolve() for path in args.reference],
                          args.engine.resolve(), args.prediction_column, args.quality_only)
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    except (ValueError, TypeError, KeyError, OSError, UnicodeError, csv.Error, OverflowError,
            StopIteration, subprocess.SubprocessError) as error:
        parser.exit(1, f"Matched comparison failed: {error}\n")
    print(f"Data quality: {report['status']}; groups={len(report['results'])}")
    if "mean_relative_mae" in report:
        print(f"Mean relative MAE (latest price = 1): {report['mean_relative_mae']}")
    if report["status"] == "blocked":
        parser.exit(1, "Data errors blocked scoring; details are saved in the output report.\n")


if __name__ == "__main__":
    main()
