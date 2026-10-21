import argparse
from bisect import bisect_left
import csv
from importlib.metadata import version
import json
import math
from pathlib import Path
import statistics
import time
import warnings

from benchmark_forecasts import HORIZONS, sha256
from stock_features import FEATURES, WARMUP, evaluate as prepare_features


MINIMUM_LABELS = 40
LOGISTIC_PARAMETERS = {"C": 1.0, "solver": "lbfgs", "max_iter": 1000, "tol": 1e-8,
                       "random_state": 42}
BINARY_PARAMETERS = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 7,
                     "max_depth": 3, "min_data_in_leaf": 20, "lambda_l2": 1.0,
                     "max_bin": 63, "num_threads": 1, "seed": 42, "deterministic": True,
                     "force_col_wise": True, "device_type": "cpu", "verbosity": -1}
ROUNDS = 80
LOG_EPSILON = 1e-15
MODELS = ("logistic", "lightgbm")
BASELINES = ("always_up", "historical_up", "latest_price_not_up")


def training_data(points, fold_start, horizon, lookback):
    if type(horizon) is not int or horizon <= 0 or type(lookback) is not int or lookback < 80:
        raise ValueError("require positive horizon and lookback of at least 80 sessions")
    cutoff = bisect_left([point["origin_date"] for point in points], fold_start)
    indices = range(max(0, cutoff - lookback), cutoff - horizon)
    if len(indices) < MINIMUM_LABELS:
        raise ValueError("insufficient pre-fold history for 40 matured training labels")
    x = [points[index]["features"] for index in indices]
    y = [int(points[index + horizon]["current_price"] > points[index]["current_price"])
         for index in indices]
    return x, y, {"training_rows": len(y), "first_training_origin": points[indices[0]]["origin_date"],
                  "last_training_origin": points[indices[-1]]["origin_date"],
                  "last_training_target": points[indices[-1] + horizon]["origin_date"],
                  "last_training_target_index": points[indices[-1] + horizon]["origin_index"],
                  "historical_up_probability": statistics.mean(y)}


def fit_predict(x, y, current):
    import lightgbm as lgb
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    train, target, observed = np.asarray(x, dtype=np.float64), np.asarray(y), np.asarray(current, dtype=np.float64)
    if (train.ndim != 2 or train.shape[1] != len(FEATURES) or observed.ndim != 2 or
            observed.shape[1] != len(FEATURES) or not len(observed) or len(target) != len(train) or
            target.ndim != 1 or len(target) < MINIMUM_LABELS or not np.isin(target, [0, 1]).all() or
            not np.isfinite(train).all() or not np.isfinite(observed).all()):
        raise ValueError("invalid direction training or prediction matrix")
    if len(set(y)) == 1:
        return ({name: [float(y[0])] * len(observed) for name in MODELS},
                {name: {"fit_kind": "constant_single_class", "constant_probability": float(y[0])}
                 for name in MODELS})
    try:
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            scaler = StandardScaler().fit(train)
            logistic = LogisticRegression(**LOGISTIC_PARAMETERS).fit(scaler.transform(train), target)
            linear = logistic.predict_proba(scaler.transform(observed))[:, list(logistic.classes_).index(1)]
            dataset = lgb.Dataset(train, label=target, feature_name=list(FEATURES))
            boosted = lgb.train(dict(BINARY_PARAMETERS), dataset, num_boost_round=ROUNDS)
            binary = boosted.predict(observed, num_threads=1)
    except (ConvergenceWarning, lgb.basic.LightGBMError, ValueError, FloatingPointError) as error:
        raise ValueError(f"direction fit/prediction failed, no fallback applied: {error}") from error
    probabilities = {"logistic": linear.tolist(), "lightgbm": binary.tolist()}
    if any(not math.isfinite(value) or not 0 <= value <= 1
           for values in probabilities.values() for value in values):
        raise ValueError("direction model returned an invalid probability")
    return probabilities, {"logistic": {"fit_kind": "native", "iterations": int(logistic.n_iter_[0]),
                                        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist()},
                           "lightgbm": {"fit_kind": "native", "trees": boosted.current_iteration()}}


def classification_metrics(labels, probabilities):
    if (not labels or len(labels) != len(probabilities) or any(type(value) is not int or value not in (0, 1)
            for value in labels) or any(type(value) not in (float, int) or not math.isfinite(value) or
                                      not 0 <= value <= 1 for value in probabilities)):
        raise ValueError("metrics require nonempty binary labels and finite probabilities in [0, 1]")
    predicted = [int(probability >= 0.5) for probability in probabilities]
    tp = sum(actual == 1 and forecast == 1 for actual, forecast in zip(labels, predicted))
    tn = sum(actual == 0 and forecast == 0 for actual, forecast in zip(labels, predicted))
    up, not_up = sum(labels), len(labels) - sum(labels)
    bins = []
    for index in range(5):
        retained = [(actual, probability) for actual, probability in zip(labels, probabilities)
                    if min(int(probability * 5), 4) == index]
        bins.append({"lower": index / 5, "upper": (index + 1) / 5, "count": len(retained),
                     "mean_probability": statistics.mean(value for _, value in retained) if retained else None,
                     "observed_up_rate": statistics.mean(actual for actual, _ in retained) if retained else None})
    return {"samples": len(labels), "accuracy": (tp + tn) / len(labels),
            "balanced_accuracy": (tp / up + tn / not_up) / 2 if up and not_up else None,
            "brier_score": statistics.mean((probability - actual) ** 2
                                            for actual, probability in zip(labels, probabilities)),
            "log_loss": -statistics.mean(math.log(max(LOG_EPSILON, min(1 - LOG_EPSILON,
                probability if actual else 1 - probability))) for actual, probability in zip(labels, probabilities)),
            "confusion": {"true_up": tp, "true_not_up": tn, "false_up": not_up - tn, "false_not_up": up - tp},
            "up_samples": up, "not_up_samples": not_up,
            "expected_calibration_error": sum(item["count"] * abs(item["mean_probability"] -
                item["observed_up_rate"]) for item in bins if item["count"]) / len(labels),
            "reliability_bins": bins}


def paired_uncertainty(points, settings):
    import numpy as np

    names, deltas = [], []
    for model in MODELS:
        names.extend((f"{model}_accuracy_gain_vs_always_up", f"{model}_brier_reduction_vs_historical_up"))
        deltas.extend(([(int(point[model] >= 0.5) == point["actual_up"]) - point["actual_up"] for point in points],
                       [(point["historical_up"] - point["actual_up"]) ** 2 -
                        (point[model] - point["actual_up"]) ** 2 for point in points]))
    differences = np.asarray(deltas, dtype=np.float64).T
    count = len(points)
    results = []
    for block in settings["block_sizes"]:
        result = {"block_size": block, "effective_blocks": count // block, "comparisons": {}}
        samples = None
        if count // block >= 10:
            generator = np.random.default_rng(settings["seed"])
            full, remainder = divmod(count, block)
            block_sums = sum(np.roll(differences, -offset, axis=0) for offset in range(block))
            partial_sums = sum(np.roll(differences, -offset, axis=0) for offset in range(remainder))
            batches = []
            for start in range(0, settings["samples"], 64):
                size = min(64, settings["samples"] - start)
                indices = generator.integers(count, size=(size, full + bool(remainder)))
                totals = block_sums[indices[:, :full]].sum(axis=1)
                if remainder:
                    totals += partial_sums[indices[:, -1]]
                batches.append(totals / count)
            samples = np.concatenate(batches)
        for index, name in enumerate(names):
            lower, upper = (map(float, np.quantile(samples[:, index], [0.025, 0.975]))
                            if samples is not None else (None, None))
            result["comparisons"][name] = {"mean": float(differences[:, index].mean()),
                "lower_95": lower, "upper_95": upper,
                "conclusion": "better" if lower is not None and lower > 0 else
                              "worse" if upper is not None and upper < 0 else "inconclusive"}
        results.append(result)
    return results


def evaluate(manifest, context_manifest):
    began = time.perf_counter()
    prepared = prepare_features(manifest, context_manifest)
    config = prepared["config"]
    if config["lookback_days"] < 80:
        raise ValueError("direction models require lookback_days of at least 80")
    report = {key: value for key, value in prepared.items() if key not in ("results", "runtime_seconds", "phase")}
    report.update(phase="validation", task="binary_direction", results=[], aggregate_metrics={},
                  models={"logistic": {"parameters": LOGISTIC_PARAMETERS, "regularization": "L2",
                                        "scaling": "training-only StandardScaler"},
                          "lightgbm": {"parameters": BINARY_PARAMETERS, "rounds": ROUNDS}},
                  procedure={"training": "separate per asset/horizon; yearly chronological refits",
                             "fold_start": "max(validation_start, January 1 of origin year)",
                             "training_target_cutoff": "strictly before fold_start",
                             "lookback_sessions": config["lookback_days"], "minimum_labels": MINIMUM_LABELS,
                             "origin_step_sessions": "equal to horizon; same grid for all models/baselines",
                             "label": "target selected price > origin selected price; ties are not up",
                             "threshold": 0.5, "log_loss_clip": LOG_EPSILON,
                             "single_class": "explicit constant probability for both models; no native fit",
                             "selection": "none; parameters fixed, no tuning or probability calibration"})
    report["source_hashes"]["direction_forecasts.py"] = sha256(Path(__file__).read_bytes())
    report["source_hashes"]["requirements-direction.txt"] = sha256(
        Path(__file__).with_name("requirements-direction.txt").read_bytes())
    report["limitations"] = [item for item in prepared["limitations"] if not item.startswith("Features only:")] + [
        "Inspected validation only; no final test or demonstrated trading edge.",
        "Probabilities are uncalibrated; classification does not predict a price level or Greek.",
        "Earlier validation labels may enter later yearly folds only after their targets mature.",
        "No class weighting or resampling; latest-price_not_up is a binary tie baseline, not a price forecast.",
        "Balanced accuracy is undefined when the scored sample has only one class.",
        "Within-group paired circular block-bootstrap intervals are sensitivity checks, not multiple-test corrections.",
        "Aggregate metrics pool correlated asset/date rows descriptively; no aggregate independence claim.",
        "Annual frozen fits differ from the existing regression model's per-origin refits."]
    if report["status"] == "blocked":
        report["runtime_seconds"] = time.perf_counter() - began
        return report
    report["dependencies"] = {name: version(name) for name in ("scikit-learn", "lightgbm", "numpy", "scipy", "threadpoolctl")}
    all_points = []
    for group in prepared["results"]:
        feature_points = group["points"]
        for horizon in HORIZONS:
            origins = [index for index in range(config["lookback_days"] - WARMUP,
                                                len(feature_points) - horizon)
                       if feature_points[index]["origin_date"] >= config["validation_start"]]
            origins = origins[::horizon]
            if not origins:
                raise ValueError(f"{group['symbol']} {horizon}d: no eligible forecasts")
            folds = {}
            for index in origins:
                day = feature_points[index]["origin_date"]
                fold = max(config["validation_start"], day[:4] + "-01-01")
                folds.setdefault(fold, []).append(index)
            points, fitted_folds = [], []
            for fold, indices in folds.items():
                x, y, training = training_data(feature_points, fold, horizon, config["lookback_days"])
                probabilities, diagnostics = fit_predict(x, y, [feature_points[index]["features"] for index in indices])
                fitted_folds.append({"fold_start": fold, **training, "models": diagnostics})
                for position, index in enumerate(indices):
                    origin, target = feature_points[index], feature_points[index + horizon]
                    points.append({"origin_date": origin["origin_date"], "target_date": target["origin_date"],
                                   "origin_index": origin["origin_index"], "current_price": origin["current_price"],
                                   "actual_price": target["current_price"],
                                   "actual_up": int(target["current_price"] > origin["current_price"]),
                                   "fold_start": fold, "last_training_target": training["last_training_target"],
                                   "last_training_target_index": training["last_training_target_index"],
                                   "always_up": 1.0, "latest_price_not_up": 0.0,
                                   "historical_up": training["historical_up_probability"],
                                   **{name: probabilities[name][position] for name in MODELS}})
            labels = [point["actual_up"] for point in points]
            metrics = {name: classification_metrics(labels, [point[name] for point in points])
                       for name in (*MODELS, *BASELINES)}
            settings = config.get("bootstrap", {"samples": 2000, "seed": 42, "block_sizes": [1, 2, 4]})
            report["results"].append({"symbol": group["symbol"], "horizon_days": horizon,
                                      "forecasts": len(points), "metrics": metrics, "folds": fitted_folds,
                                      "bootstrap_sensitivity": paired_uncertainty(points, settings), "points": points})
            all_points.extend(points)
    report["aggregate_metrics"] = {name: classification_metrics([point["actual_up"] for point in all_points],
        [point[name] for point in all_points]) for name in (*MODELS, *BASELINES)}
    report["runtime_seconds"] = time.perf_counter() - began
    return report


def main():
    parser = argparse.ArgumentParser(description="Validation-only logistic and LightGBM stock direction models")
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
        parser.exit(1, f"Direction experiment failed: {error}. Optional dependencies: python/requirements-direction.txt\n")
    if report["status"] != "ok":
        parser.exit(1, f"Direction experiment blocked by data checks; see {args.output}\n")
    for group in report["results"]:
        scores = group["metrics"]
        print(f"{group['symbol']} {group['horizon_days']}d: n={group['forecasts']}, "
              f"logistic={scores['logistic']['accuracy']:.3f}, lightgbm={scores['lightgbm']['accuracy']:.3f}, "
              f"always-up={scores['always_up']['accuracy']:.3f}")
    print(f"Saved validation direction results to {args.output}; no final period scored")


if __name__ == "__main__":
    main()
