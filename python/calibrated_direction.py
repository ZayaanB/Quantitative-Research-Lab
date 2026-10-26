import argparse
from collections import Counter
import csv
from importlib.metadata import version
import json
import math
from pathlib import Path
import statistics
import time
import warnings

from benchmark_forecasts import HORIZONS, sha256
from direction_forecasts import (BASELINES, BINARY_PARAMETERS, LOGISTIC_PARAMETERS, ROUNDS,
                                classification_metrics, fit_predict, origin_folds,
                                paired_uncertainty, pooled_training_data)
from stock_features import evaluate as prepare_features


CANDIDATES = (
    {"id": "historical_up", "model": "baseline", "overrides": {}},
    {"id": "logistic_c0.1", "model": "logistic", "overrides": {"C": 0.1}},
    {"id": "logistic_c1", "model": "logistic", "overrides": {"C": 1.0}},
    {"id": "lightgbm_l2_1", "model": "lightgbm", "overrides": {"lambda_l2": 1.0}},
    {"id": "lightgbm_l2_10", "model": "lightgbm", "overrides": {"lambda_l2": 10.0}},
)
CALIBRATION_CLIP = 1e-6
MINIMUM_OOF_ROWS = 20
MINIMUM_OOF_DATES = 10
OUTPUT_MODELS = ("selected_raw", "calibrated")


def predict_candidate(histories, requests, fold_start, horizon, lookback, candidate):
    x, y, training = pooled_training_data(histories, fold_start, horizon, lookback)
    if candidate["model"] == "baseline":
        probabilities = [training["by_asset"][symbol]["historical_up_probability"] for _, symbol, _ in requests]
        diagnostics = {"fit_kind": "declared_historical_up_baseline"}
    else:
        model = candidate["model"]
        arguments = {"logistic_parameters" if model == "logistic" else "binary_parameters": candidate["overrides"]}
        predictions, fitted = fit_predict(x, y, [histories[symbol][index]["features"] for _, symbol, index in requests],
                                          model_names=(model,), **arguments)
        probabilities, diagnostics = predictions[model], fitted[model]
    return probabilities, {"candidate_id": candidate["id"], "fold_start": fold_start,
                            "horizon_days": horizon, **training, "model": diagnostics}


def oof_block(histories, year, horizon, lookback, candidate):
    start, end = f"{year:04}-01-01", f"{year + 1:04}-01-01"
    requests = []
    for symbol, points in sorted(histories.items()):
        indices = [index for index in range(len(points) - horizon)
                   if start <= points[index]["origin_date"] < end and points[index + horizon]["origin_date"] < end][::horizon]
        if not indices:
            raise ValueError(f"{symbol} {year} {horizon}d: empty OOF block; no asset was dropped")
        requests.extend((points[index]["origin_date"], symbol, index) for index in indices)
    requests.sort()
    if len(requests) < MINIMUM_OOF_ROWS or len({day for day, _, _ in requests}) < MINIMUM_OOF_DATES:
        raise ValueError("OOF block requires at least 20 rows and 10 distinct origin dates")
    probabilities, fitted = predict_candidate(histories, requests, start, horizon, lookback, candidate)
    rows = [{"symbol": symbol, "origin_date": day,
             "target_date": histories[symbol][index + horizon]["origin_date"],
             "actual_up": int(histories[symbol][index + horizon]["current_price"] > histories[symbol][index]["current_price"]),
             "probability": probability}
            for (day, symbol, index), probability in zip(requests, probabilities)]
    metrics = {symbol: classification_metrics([row["actual_up"] for row in rows if row["symbol"] == symbol],
                                              [row["probability"] for row in rows if row["symbol"] == symbol])
               for symbol in sorted(histories)}
    return {"block_id": f"{candidate['id']}:{horizon}d:{year}", "year": year, "start": start, "end_exclusive": end,
            "candidate_id": candidate["id"], "horizon_days": horizon, "fit": fitted,
            "rows": len(rows), "origin_dates": len({row["origin_date"] for row in rows}),
            "metrics_by_asset": metrics, "points": rows}


def select_candidate(blocks_by_candidate):
    if set(blocks_by_candidate) != {candidate["id"] for candidate in CANDIDATES}:
        raise ValueError("selection requires every predeclared candidate")
    reference = None
    scores = []
    for candidate in CANDIDATES:
        blocks = blocks_by_candidate[candidate["id"]]
        if len(blocks) != 2:
            raise ValueError("selection requires two chronological OOF blocks")
        grid = [(row["symbol"], row["origin_date"], row["target_date"], row["actual_up"])
                for block in blocks for row in block["points"]]
        if reference is not None and grid != reference:
            raise ValueError("candidate OOF grids differ; no silent intersection is allowed")
        reference = grid
        scores.append({"candidate_id": candidate["id"], "selection_brier": statistics.mean(
            metrics["brier_score"] for block in blocks for metrics in block["metrics_by_asset"].values()),
            "block_ids": [block["block_id"] for block in blocks]})
    best = min(range(len(scores)), key=lambda index: scores[index]["selection_brier"])
    return CANDIDATES[best], scores


def calibrate(rows, current):
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    if len(rows) < MINIMUM_OOF_ROWS or len({row["origin_date"] for row in rows}) < MINIMUM_OOF_DATES:
        raise ValueError("calibration requires at least 20 OOF rows and 10 origin dates")
    labels = [row["actual_up"] for row in rows]
    prior = [row["probability"] for row in rows]
    classification_metrics(labels, prior)
    if not current or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in current):
        raise ValueError("calibration requires finite current probabilities in [0, 1]")
    if len(set(labels)) == 1:
        return [float(labels[0])] * len(current), {"fit_kind": "constant_single_class", "probability": float(labels[0])}

    def logits(values):
        clipped = np.clip(np.asarray(values, dtype=np.float64), CALIBRATION_CLIP, 1 - CALIBRATION_CLIP)
        return (np.log(clipped) - np.log1p(-clipped)).reshape(-1, 1)

    try:
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            fitted = LogisticRegression(**LOGISTIC_PARAMETERS).fit(logits(prior), labels)
            probabilities = fitted.predict_proba(logits(current))[:, list(fitted.classes_).index(1)].tolist()
    except (ConvergenceWarning, ValueError, FloatingPointError) as error:
        raise ValueError(f"calibration failed; no fallback applied: {error}") from error
    if len(probabilities) != len(current) or any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError("calibrator returned an invalid probability")
    return probabilities, {"fit_kind": "native_sigmoid_on_logit", "coefficient": float(fitted.coef_[0, 0]),
                            "intercept": float(fitted.intercept_[0]), "iterations": int(fitted.n_iter_[0])}


def evaluate(manifest, context_manifest):
    began = time.perf_counter()
    prepared = prepare_features(manifest, context_manifest)
    config = prepared["config"]
    if config["lookback_days"] < 80:
        raise ValueError("calibrated direction requires lookback_days of at least 80")
    report = {key: value for key, value in prepared.items() if key not in ("results", "runtime_seconds", "phase")}
    report.update(phase="validation", task="binary_direction", training_mode="pooled_selection_calibration",
                  results=[], aggregate_metrics={}, outer_folds=[], oof_blocks=[],
                  procedure={"candidates_in_tie_order": CANDIDATES, "logistic_parameters": LOGISTIC_PARAMETERS,
                             "lightgbm_parameters": BINARY_PARAMETERS, "rounds": ROUNDS,
                             "selection": "lowest Brier, equal asset/year weights, on years Y-3 and Y-2; first candidate wins ties",
                             "calibration": "fixed L2 C=1 sigmoid on clipped logits of chosen candidate OOF predictions in Y-1",
                             "calibration_clip": CALIBRATION_CLIP, "minimum_oof_rows": MINIMUM_OOF_ROWS,
                             "minimum_oof_dates": MINIMUM_OOF_DATES,
                             "training_target_cutoff": "strictly before each inner/outer fold start across every asset",
                             "oof_target_cutoff": "strictly before inner block end; boundary-crossing labels purged",
                             "outer": "same yearly folds and 1/5/20-session origin grids as fixed classifiers",
                             "threshold": 0.5, "deployment": "none; existing defaults unchanged"})
    report["source_hashes"].update({name: sha256(Path(__file__).with_name(name).read_bytes()) for name in
                                   ("calibrated_direction.py", "direction_forecasts.py", "requirements-direction.txt")})
    report["limitations"] = [item for item in prepared["limitations"] if not item.startswith("Features only:")] + [
        "Inspected validation only; this small search does not make 2018-2023 untouched.",
        "Calibration is estimated, not guaranteed; Brier combines calibration and discrimination.",
        "The next-year refit may change the score distribution relative to the prior OOF calibration year.",
        "Raw, calibrated and losing candidate results are preserved; the baseline may win selection.",
        "Pooled rows/assets and overlapping training labels are dependent; aggregate metrics are descriptive.",
        "Sigmoid coefficients may reverse weak rankings; no threshold tuning or automatic model promotion.",
        "Within-stock bootstrap checks are not joint cross-stock or multiple-experiment corrections.",
        "A present-day survivor list is not a point-in-time or survivorship-free universe."]
    if report["status"] == "blocked":
        report["runtime_seconds"] = time.perf_counter() - began
        return report
    report["dependencies"] = {name: version(name) for name in ("scikit-learn", "lightgbm", "numpy", "scipy", "threadpoolctl")}
    histories = {group["symbol"]: group["points"] for group in prepared["results"]}
    schedules = {(symbol, horizon): origin_folds(points, config, horizon)
                 for symbol, points in histories.items() for horizon in HORIZONS}
    blocks, routed = {}, {}

    def block(candidate, year, horizon):
        key = (candidate["id"], year, horizon)
        if key not in blocks:
            blocks[key] = oof_block(histories, year, horizon, config["lookback_days"], candidate)
            report["oof_blocks"].append(blocks[key])
        return blocks[key]

    for horizon in HORIZONS:
        folds = sorted({fold for symbol in histories for fold in schedules[(symbol, horizon)]})
        for fold in folds:
            year = int(fold[:4])
            if year < 4:
                raise ValueError("three preceding calendar years are required for selection/calibration")
            candidate, scores = select_candidate({candidate["id"]: [block(candidate, year - offset, horizon)
                                                    for offset in (3, 2)] for candidate in CANDIDATES})
            calibration = block(candidate, year - 1, horizon)
            requests = sorted((histories[symbol][index]["origin_date"], symbol, index)
                              for symbol in histories for index in schedules[(symbol, horizon)].get(fold, []))
            raw, fitted = predict_candidate(histories, requests, fold, horizon, config["lookback_days"], candidate)
            calibrated, calibrator = calibrate(calibration["points"], raw)
            fit_id = f"selected:{horizon}d:{fold}"
            report["outer_folds"].append({"fit_id": fit_id, "fold_start": fold, "horizon_days": horizon,
                "selected_candidate": candidate["id"], "candidate_scores": scores,
                "selection_end_exclusive": f"{year - 1:04}-01-01", "calibration_block_id": calibration["block_id"],
                "calibration_last_target": max(row["target_date"] for row in calibration["points"]),
                "calibrator": calibrator, "base_fit": fitted})
            for (_, symbol, index), probability, adjusted in zip(requests, raw, calibrated):
                point, target = histories[symbol][index], histories[symbol][index + horizon]
                local = fitted["by_asset"][symbol]
                routed[(symbol, horizon, index)] = {"origin_date": point["origin_date"],
                    "target_date": target["origin_date"], "origin_index": point["origin_index"],
                    "current_price": point["current_price"], "actual_price": target["current_price"],
                    "actual_up": int(target["current_price"] > point["current_price"]),
                    "fold_start": fold, "fit_id": fit_id, "selected_candidate": candidate["id"],
                    "last_training_target": local["last_training_target"],
                    "last_training_target_index": local["last_training_target_index"],
                    "pooled_last_training_target": fitted["last_training_target"],
                    "selected_raw": probability, "calibrated": adjusted, "always_up": 1.0,
                    "latest_price_not_up": 0.0, "historical_up": local["historical_up_probability"]}
    all_points = []
    settings = config.get("bootstrap", {"samples": 2000, "seed": 42, "block_sizes": [1, 2, 4]})
    for group in prepared["results"]:
        symbol = group["symbol"]
        for horizon in HORIZONS:
            points = [routed[(symbol, horizon, index)] for indices in schedules[(symbol, horizon)].values() for index in indices]
            labels = [point["actual_up"] for point in points]
            report["results"].append({"symbol": symbol, "horizon_days": horizon, "forecasts": len(points),
                "metrics": {name: classification_metrics(labels, [point[name] for point in points])
                            for name in (*OUTPUT_MODELS, *BASELINES)},
                "bootstrap_sensitivity": paired_uncertainty(points, settings, models=OUTPUT_MODELS),
                "calibration_sensitivity": paired_uncertainty(points, settings, models=("calibrated",), brier_reference="selected_raw"),
                "points": points})
            all_points.extend(points)
    report["aggregate_metrics"] = {name: classification_metrics([point["actual_up"] for point in all_points],
        [point[name] for point in all_points]) for name in (*OUTPUT_MODELS, *BASELINES)}
    report["selection_counts"] = dict(Counter(fold["selected_candidate"] for fold in report["outer_folds"]))
    report["runtime_seconds"] = time.perf_counter() - began
    return report


def main():
    parser = argparse.ArgumentParser(description="Past-only pooled direction selection and probability calibration")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("output already exists; choose a new path")
        report = evaluate(args.manifest.resolve(), args.context_manifest.resolve())
        encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    except (ValueError, TypeError, KeyError, OSError, UnicodeError, csv.Error, ImportError, OverflowError) as error:
        parser.exit(1, f"Calibrated direction experiment failed: {error}; dependencies: python/requirements-direction.txt\n")
    if report["status"] != "ok":
        parser.exit(1, f"Calibrated direction blocked by data checks; see {args.output}\n")
    for group in report["results"]:
        scores = group["metrics"]
        print(f"{group['symbol']} {group['horizon_days']}d: n={group['forecasts']}, "
              f"raw/calibrated Brier={scores['selected_raw']['brier_score']:.4f}/{scores['calibrated']['brier_score']:.4f}, "
              f"calibrated/always-up accuracy={scores['calibrated']['accuracy']:.3f}/{scores['always_up']['accuracy']:.3f}")
    print(f"Selections: {report['selection_counts']}; validation only, no default changed")


if __name__ == "__main__":
    main()
