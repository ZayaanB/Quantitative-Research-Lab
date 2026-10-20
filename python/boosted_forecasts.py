import argparse
import csv
import json
import math
import platform
import statistics
import time
from importlib.metadata import version
from pathlib import Path

from benchmark_forecasts import HORIZONS, bootstrap_sensitivity, read_asset, read_manifest, sha256
from ensemble_forecasts import point_metrics


FEATURES = ("return_1", "return_2", "return_5", "return_10", "return_20",
            "volatility_5", "volatility_20", "distance_from_mean_20")
PARAMETERS = {"objective": "regression_l1", "learning_rate": 0.03, "num_leaves": 7,
              "max_depth": 3, "min_data_in_leaf": 20, "lambda_l2": 1.0,
              "max_bin": 63, "num_threads": 1, "seed": 42, "deterministic": True,
              "force_col_wise": True, "device_type": "cpu", "verbosity": -1}
ROUNDS = 80


def price_features(logs, returns, index):
    return ([logs[index] - logs[index - lag] for lag in (1, 2, 5, 10, 20)] +
            [statistics.pstdev(returns[index - span:index]) for span in (5, 20)] +
            [logs[index] - statistics.mean(logs[index - 19:index + 1])])


def training_set(prices, horizon, lookback):
    if type(horizon) is not int or horizon <= 0 or type(lookback) is not int or lookback < 60:
        raise ValueError("require positive horizon and lookback of at least 60")
    if any(not math.isfinite(price) or price <= 0 for price in prices):
        raise ValueError("prices must be finite and positive")
    logs = [math.log(price) for price in prices]
    origin = len(logs) - 1
    returns = [logs[i] - logs[i - 1] for i in range(1, len(logs))]

    indices = range(max(20, origin - lookback), origin - horizon + 1)
    if len(indices) < 40:
        raise ValueError("insufficient history for 40 matured training labels")
    return ([price_features(logs, returns, index) for index in indices],
            [logs[index + horizon] - logs[index] for index in indices], price_features(logs, returns, origin))


def forecast(prices, horizon, lookback):
    features, targets, current = training_set(prices, horizon, lookback)
    import lightgbm as lgb
    import numpy as np

    dataset = lgb.Dataset(np.asarray(features, dtype=np.float64), label=np.asarray(targets),
                          feature_name=list(FEATURES))
    try:
        fitted = lgb.train(dict(PARAMETERS), dataset, num_boost_round=ROUNDS)
        predicted_return = float(fitted.predict(np.asarray([current]), num_threads=1)[0])
    except lgb.basic.LightGBMError as error:
        raise ValueError(f"LightGBM fit or prediction failed: {error}") from error
    try:
        price = prices[-1] * math.exp(predicted_return)
    except OverflowError as error:
        raise ValueError("boosted forecast exceeds finite range") from error
    if not math.isfinite(price) or price <= 0:
        raise ValueError("boosted forecast exceeds positive finite range")
    gains = dict(zip(FEATURES, map(float, fitted.feature_importance(importance_type="gain"))))
    if any(not math.isfinite(gain) for gain in gains.values()):
        raise ValueError("feature gains exceed finite range")
    return {"forecast_price": price, "predicted_log_return": predicted_return,
            "training_rows": len(targets), "last_training_target_index": len(prices) - 1,
            "trees": fitted.current_iteration(), "feature_gain": gains}


def evaluate(manifest):
    dependencies = {name: version(name) for name in ("lightgbm", "numpy", "scipy")}
    config, digest = read_manifest(manifest)
    if config["lookback_days"] < 60:
        raise ValueError("boosted models require lookback_days of at least 60")
    began = time.perf_counter()
    groups, provenance, ratios = [], [], []
    ineligible = False
    for asset in config["assets"]:
        header, rows, metadata, _ = read_asset(asset, manifest.parent)
        column = next(csv.reader([header])).index(metadata["price_column"])
        retained = [(day, float(next(csv.reader([line]))[column])) for day, line in rows
                    if day < config["holdout_start"] and day <= config["test_end"]]
        dates, prices = zip(*retained) if retained else ((), ())
        provenance.append(metadata)
        for horizon in HORIZONS:
            first = max(config["lookback_days"], 20 + horizon + 39)
            first = next((i for i in range(first, len(prices) - horizon)
                          if dates[i] >= config["validation_start"]), len(prices))
            points = []
            for origin in range(first, len(prices) - horizon, horizon):
                prediction = forecast(prices[:origin + 1], horizon, config["lookback_days"])
                points.append({"origin_date": dates[origin], "target_date": dates[origin + horizon],
                               "current_price": prices[origin], "actual_price": prices[origin + horizon],
                               "latest_price_forecast": prices[origin], **prediction})
            if not points:
                raise ValueError(f"{asset['symbol']} {horizon}d: no eligible forecasts after warmup")
            metrics = {name: point_metrics(points, key) for name, key in
                       (("lightgbm", "forecast_price"), ("latest_price", "current_price"))}
            baseline, error = metrics["latest_price"]["mae"], metrics["lightgbm"]["mae"]
            if baseline:
                ratio = error / baseline
                if not math.isfinite(ratio):
                    raise ValueError("relative MAE exceeds finite range")
                ratios.append(ratio)
            elif error:
                ineligible = True
            settings = config.get("bootstrap", {"samples": 2000, "seed": 42, "block_sizes": [1, 2, 4]})
            groups.append({"symbol": asset["symbol"], "horizon_days": horizon, "forecasts": len(points),
                           "metrics": metrics, "points": points,
                           "always_up_accuracy": statistics.mean(p["actual_price"] > p["current_price"] for p in points),
                           "bootstrap_sensitivity": bootstrap_sensitivity(points, settings)})
    score = statistics.mean(ratios) if ratios and not ineligible else None
    return {"schema_version": 1, "phase": "validation", "manifest_sha256": digest, "config": config,
            "dependencies": dependencies, "provenance": provenance, "python_version": platform.python_version(),
            "platform": platform.platform(), "runtime_seconds": time.perf_counter() - began,
            "source_hashes": {name: sha256(Path(__file__).with_name(name).read_bytes()) for name in
                              ("boosted_forecasts.py", "benchmark_forecasts.py", "ensemble_forecasts.py")},
            "model": {"name": "lightgbm", "parameters": PARAMETERS, "rounds": ROUNDS,
                      "features": FEATURES, "target": "direct-horizon log return", "minimum_labels": 40},
            "mean_relative_mae": score, "selected": "lightgbm" if score is not None and score < 1 else "latest_price",
            "selection_rule": "Equal-weight asset/horizon mean MAE ratio; baseline wins ties; "
                              "both-zero groups excluded, nonzero/perfect-baseline is unavailable.",
            "limitations": ["Validation-only challenger; does not replace the frozen ensemble or C++ CLI.",
                            "Settings fixed before evaluation; previously inspected Apple history is exploratory.",
                            "Point forecasts only; no calibrated probabilities or prediction intervals.",
                            "Exponentiated log-return forecast is not expected arithmetic price.",
                            "Feature gain is descriptive, not causal explanation; price-only inputs may contain little signal.",
                            "Bootstrap labels are not corrected for the history of multiple model experiments.",
                            "Determinism is limited to matching dependency versions and implementation."], "results": groups}


def main():
    parser = argparse.ArgumentParser(description="Optional validation-only boosted return model")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("output already exists; choose a new path")
        report = evaluate(args.manifest.resolve())
        encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    except (ValueError, TypeError, KeyError, OSError, csv.Error, ImportError, OverflowError) as error:
        parser.exit(1, f"Boosted experiment failed: {error}. Install python/requirements-ml.txt if dependencies are missing.\n")
    print(f"Validation choice: {report['selected']}; mean relative MAE={report['mean_relative_mae']}")


if __name__ == "__main__":
    main()
