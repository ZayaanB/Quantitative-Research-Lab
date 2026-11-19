import argparse
import csv
import json
import math
import platform
import statistics
import time
import warnings
from importlib.metadata import version
from pathlib import Path

from benchmark_forecasts import HORIZONS, read_asset, read_manifest, sha256
from ensemble_forecasts import point_metrics


MODELS = ("arima_110", "ets_level", "ets_damped")


def predict(prices, horizon, model):
    if model not in MODELS or type(horizon) is not int or horizon <= 0 or len(prices) < 31:
        raise ValueError("require a supported model, positive horizon and at least 31 prices")
    if any(not math.isfinite(price) or price <= 0 for price in prices):
        raise ValueError("prices must be finite and positive")
    if all(price == prices[0] for price in prices):
        return {"price": prices[-1], "warnings": []}
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    offset = math.log(prices[-1])
    logs = [math.log(price) - offset for price in prices]
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        if model == "arima_110":
            fit = ARIMA(logs, order=(1, 1, 0), trend="n").fit(method_kwargs={"maxiter": 200})
        else:
            fit = ETSModel(logs, error="add", trend="add" if model == "ets_damped" else None,
                           damped_trend=model == "ets_damped", seasonal=None).fit(disp=False, maxiter=200)
        if not fit.mle_retvals.get("converged", False):
            raise ValueError("optimizer did not converge")
        prediction = float(fit.forecast(horizon)[-1])
    try:
        price = math.exp(offset + prediction)
    except OverflowError as error:
        raise ValueError("classical forecast exceeds finite range") from error
    if not math.isfinite(price) or price <= 0:
        raise ValueError("classical forecast exceeds positive finite range")
    return {"price": price, "warnings": [str(item.message)[:300] for item in captured]}


def evaluate(manifest):
    dependencies = {name: version(name) for name in ("statsmodels", "numpy", "scipy", "pandas")}
    from numpy.linalg import LinAlgError
    config, digest = read_manifest(manifest)
    if config["lookback_days"] < 30:
        raise ValueError("classical models require lookback_days of at least 30")
    results, provenance, failed = [], [], set()
    began = time.perf_counter()
    for asset in config["assets"]:
        header, rows, metadata, _ = read_asset(asset, manifest.parent)
        column = next(csv.reader([header])).index(metadata["price_column"])
        retained = [(day, float(next(csv.reader([line]))[column])) for day, line in rows
                    if day <= config["test_end"] and day < config["holdout_start"]]
        dates, prices = zip(*retained) if retained else ((), ())
        provenance.append(metadata)
        for horizon in HORIZONS:
            points, failures, messages = [], [], []
            first = next((i for i in range(config["lookback_days"], len(prices) - horizon)
                          if dates[i] >= config["validation_start"]), len(prices))
            for origin in range(first, len(prices) - horizon, horizon):
                point = {"origin_date": dates[origin], "target_date": dates[origin + horizon],
                         "current_price": prices[origin], "actual_price": prices[origin + horizon]}
                training = prices[origin - config["lookback_days"]:origin + 1]
                for model in MODELS:
                    try:
                        prediction = predict(training, horizon, model)
                        point[model] = prediction["price"]
                        if prediction["warnings"]:
                            messages.append({"model": model, "origin_date": dates[origin],
                                             "messages": prediction["warnings"]})
                    except (ValueError, RuntimeError, ArithmeticError, LinAlgError) as error:
                        point[model] = None
                        failed.add(model)
                        failures.append({"model": model, "origin_date": dates[origin], "error": str(error)[:300]})
                points.append(point)
            if not points:
                raise ValueError(f"{asset['symbol']} {horizon}d: no eligible forecasts")
            metrics = {"latest_price": point_metrics(points, "current_price")}
            for model in MODELS:
                metrics[model] = point_metrics(points, model) if all(p[model] is not None for p in points) else None
            results.append({"symbol": asset["symbol"], "horizon_days": horizon, "forecasts": len(points),
                            "metrics": metrics, "points": points, "fit_failures": failures, "fit_warnings": messages})
    ranking = [{"model": "latest_price", "mean_relative_mae": 1.0}]
    for model in MODELS:
        ratios, ineligible = [], model in failed
        for group in results:
            if group["metrics"][model] is None:
                continue
            baseline, error = group["metrics"]["latest_price"]["mae"], group["metrics"][model]["mae"]
            if baseline:
                ratio = error / baseline
                if not math.isfinite(ratio):
                    raise ValueError("relative MAE exceeds finite range")
                ratios.append(ratio)
            elif error:
                ineligible = True
        ranking.append({"model": model, "mean_relative_mae": statistics.mean(ratios)
                        if ratios and not ineligible else None})
    return {"schema_version": 1, "phase": "validation", "manifest_sha256": digest, "config": config,
            "provenance": provenance, "dependencies": dependencies, "python_version": platform.python_version(),
            "platform": platform.platform(), "runtime_seconds": time.perf_counter() - began,
            "source_hashes": {name: sha256(Path(__file__).with_name(name).read_bytes())
                              for name in ("classical_forecasts.py", "benchmark_forecasts.py", "ensemble_forecasts.py")},
            "models": {"arima_110": "ARIMA(1,1,0), no drift", "ets_level": "ETS(A,N,N)",
                       "ets_damped": "ETS(A,Ad,N)", "transform": "log prices centered on last training close",
                       "constant_window": "exact latest price; bypass degenerate optimization", "maxiter": 200},
            "ranking": ranking, "selected": min((r for r in ranking if r["mean_relative_mae"] is not None),
                                                  key=lambda r: r["mean_relative_mae"])["model"],
            "limitations": ["Validation-only exploratory comparison; no predictive-edge claim or holdout scored.",
                            "Any failed fit disqualifies a candidate globally; no silent baseline fallback.",
                            "Fixed specifications, same paired dates, no seasonal assumption or order search.",
                            "Point forecasts are exponentiated fitted log values, not expected arithmetic prices.",
                            "No prediction intervals; latest price wins ties; zero/zero groups excluded.",
                            "Does not replace the frozen ensemble procedure."], "results": results}


def main():
    parser = argparse.ArgumentParser(description="Optional validation-only ARIMA/ETS experiment")
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
        parser.exit(1, f"Classical experiment failed: {error}. Install python/requirements-forecast.txt if dependencies are missing.\n")
    print(f"Validation choice: {report['selected']}; saved {args.output}")


if __name__ == "__main__":
    main()
