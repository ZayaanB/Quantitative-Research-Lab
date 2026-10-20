import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time

from benchmark_forecasts import read_manifest, sha256
from boosted_forecasts import FEATURES as PRICE_FEATURES, price_features
from forecast_quality import inspect_asset


CONTEXT_SYMBOLS = ("SPY", "QQQ", "XLK")
WARMUP = 20
OHLCV = ("Open", "High", "Low", "Close", "Volume")
CONTEXT_FEATURES = ("return_1", "return_5", "return_20", "volatility_5", "volatility_20")
FEATURES = (*PRICE_FEATURES, "intraday_log_range", "intraday_log_return", "close_location",
            "log_volume_change_1", "log_volume_relative_20",
            *(f"{symbol.lower()}_{name}" for symbol in CONTEXT_SYMBOLS for name in CONTEXT_FEATURES))


@dataclass(frozen=True)
class DailyHistory:
    dates: tuple
    prices: tuple
    logs: tuple
    returns: tuple
    bars: tuple


def load_history(asset, directory, holdout_start, require_ohlcv):
    loaded, quality = inspect_asset(asset, directory, holdout_start)
    if loaded is None or quality["status"] != "ok":
        return None, quality
    header, rows, metadata = loaded
    columns = next(csv.reader([header]))
    if require_ohlcv and any(columns.count(name) != 1 for name in OHLCV):
        quality["status"] = "error"
        quality["issues"].append({"severity": "error", "code": "missing_required_ohlcv",
                                  "detail": "stock features require unique Open/High/Low/Close/Volume columns"})
        return None, quality
    price_column = columns.index(metadata["price_column"])
    retained = [(day, next(csv.reader([line]))) for day, line in rows if day < holdout_start]
    dates = tuple(day for day, _ in retained)
    prices = tuple(float(fields[price_column]) for _, fields in retained)
    logs = tuple(math.log(price) for price in prices)
    returns = tuple(logs[index] - logs[index - 1] for index in range(1, len(logs)))
    bars = tuple(tuple(float(fields[columns.index(name)]) for name in OHLCV)
                 for _, fields in retained) if require_ohlcv else ()
    return DailyHistory(dates, prices, logs, returns, bars), quality


def feature_rows(stock, contexts):
    if len(stock.dates) <= WARMUP:
        raise ValueError("stock features require at least 21 development sessions")
    if set(contexts) != set(CONTEXT_SYMBOLS):
        raise ValueError("provide exactly SPY, QQQ and XLK contexts")
    positions = {}
    for symbol in CONTEXT_SYMBOLS:
        history = contexts[symbol]
        aligned = tuple(day for day in history.dates if stock.dates[0] <= day <= stock.dates[-1])
        if aligned != stock.dates:
            raise ValueError(f"{symbol}: context session grid differs from stock; no filling or dropping is allowed")
        positions[symbol] = {day: index for index, day in enumerate(history.dates)}
    log_volume = [math.log1p(bar[4]) for bar in stock.bars]
    rows = []
    for index in range(WARMUP, len(stock.dates)):
        opened, high, low, closed, _ = stock.bars[index]
        values = price_features(stock.logs, stock.returns, index)
        values.extend((math.log(high) - math.log(low), math.log(closed) - math.log(opened),
                       (closed - low) / (high - low) * 2 - 1 if high != low else 0.0,
                       log_volume[index] - log_volume[index - 1],
                       log_volume[index] - statistics.mean(log_volume[index - 19:index + 1])))
        for symbol in CONTEXT_SYMBOLS:
            history = contexts[symbol]
            position = positions[symbol][stock.dates[index]]
            values.extend(history.logs[position] - history.logs[position - lag] for lag in (1, 5, 20))
            values.extend(statistics.pstdev(history.returns[position - span:position]) for span in (5, 20))
        if len(values) != len(FEATURES) or any(not math.isfinite(value) for value in values):
            raise ValueError(f"{stock.dates[index]}: features exceed finite numeric range")
        rows.append({"origin_date": stock.dates[index], "origin_index": index,
                     "current_price": stock.prices[index], "features": values})
    return rows


def evaluate(manifest, context_manifest):
    began = time.perf_counter()
    config, digest = read_manifest(manifest)
    context_config, context_digest = read_manifest(context_manifest)
    if {key: value for key, value in config.items() if key != "assets"} != {
            key: value for key, value in context_config.items() if key != "assets"}:
        raise ValueError("context and stock manifests must use identical settings except assets")
    if {asset["symbol"] for asset in context_config["assets"]} != set(CONTEXT_SYMBOLS):
        raise ValueError("context manifest must contain exactly SPY, QQQ and XLK")
    histories, contexts, quality = {}, {}, []
    for role, assets, directory, destination in (
            ("stock", config["assets"], manifest.parent, histories),
            ("context", context_config["assets"], context_manifest.parent, contexts)):
        for asset in assets:
            history, inspection = load_history(asset, directory, config["holdout_start"], role == "stock")
            quality.append({"role": role, **inspection})
            if history is not None:
                destination[asset["symbol"]] = history
    report = {"schema_version": 1, "phase": "development_features", "manifest_sha256": digest,
              "context_manifest_sha256": context_digest, "config": config, "context_config": context_config,
              "status": "blocked" if any(item["status"] == "error" for item in quality) else "ok",
              "quality": quality, "feature_names": FEATURES, "warmup_sessions": WARMUP, "results": [],
              "python_version": platform.python_version(), "platform": platform.platform(),
              "source_hashes": {name: sha256(Path(__file__).with_name(name).read_bytes()) for name in
                  ("stock_features.py", "boosted_forecasts.py", "forecast_quality.py",
                   "benchmark_forecasts.py", "ensemble_forecasts.py")},
              "availability": "After all stock/context daily closes and provider publication on the origin date.",
              "limitations": ["Features only: no labels, fitting, selection, calibration or accuracy claim.",
                              "Only dates strictly before holdout_start are emitted; first 20 sessions are warmup.",
                              "Same-day close availability is assumed, not verified by publication timestamps.",
                              "Whole-file hashes and basic price/date validation include reserved rows.",
                              "Retrospective provider adjustments are not point-in-time or survivorship-free.",
                              "Volume is provider-reported; log1p supports zero, not verified split normalization.",
                              "SPY/QQQ/XLK are fixed research proxies, not per-stock sector assignments.",
                              "Context sessions must exactly match stock sessions; no fills or silent row removal."]}
    try:
        report["source_revision"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            text=True, capture_output=True, check=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        report["source_revision"] = None
    if report["status"] == "ok":
        for symbol, history in histories.items():
            rows = feature_rows(history, contexts)
            report["results"].append({"symbol": symbol, "rows": len(rows), "warmup_rows": WARMUP,
                                      "validation_rows": sum(row["origin_date"] >= config["validation_start"]
                                                             for row in rows),
                                      "first_origin": rows[0]["origin_date"], "last_origin": rows[-1]["origin_date"],
                                      "points": rows})
    report["runtime_seconds"] = time.perf_counter() - began
    return report


def main():
    parser = argparse.ArgumentParser(description="Export past-only OHLCV and market-context stock features")
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
    except (ValueError, TypeError, KeyError, OSError, UnicodeError, csv.Error, OverflowError) as error:
        parser.exit(1, f"Stock feature export failed: {error}\n")
    if report["status"] != "ok":
        parser.exit(1, f"Stock feature export blocked by data checks; see {args.output}\n")
    for result in report["results"]:
        print(f"{result['symbol']}: {result['rows']} rows x {len(FEATURES)} features; "
              f"{result['first_origin']} to {result['last_origin']}")
    print(f"Saved development features to {args.output}; no models fitted or scored")


if __name__ == "__main__":
    main()
