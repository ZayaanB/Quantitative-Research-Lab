import argparse
from bisect import bisect_left
import csv
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from benchmark_forecasts import HORIZONS, sha256
from direction_forecasts import BASELINES, classification_metrics, origin_folds, paired_uncertainty, pooled_training_data
from stock_features import FEATURES, evaluate as prepare_features


SESSIONS = 20
EPOCHS = 20
BATCH_SIZE = 128
SEED = 42
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001


def windows(histories, requests):
    result = []
    for _, symbol, index in requests:
        if index < SESSIONS - 1:
            raise ValueError(f"{symbol}: insufficient sequence history; requested workload was not changed")
        rows = [point["features"] for point in histories[symbol][index - SESSIONS + 1:index + 1]]
        if len(rows) != SESSIONS or any(len(row) != len(FEATURES) for row in rows):
            raise ValueError("sequence dimensions do not match the feature contract")
        result.append([list(channel) for channel in zip(*rows)])
    return result


def training_sequences(histories, fold_start, horizon, lookback):
    _, labels, details = pooled_training_data(histories, fold_start, horizon, lookback)
    requests = []
    for symbol, local in details["by_asset"].items():
        points = histories[symbol]
        first = bisect_left([point["origin_date"] for point in points], local["first_training_origin"])
        requests.extend((points[index]["origin_date"], symbol, index)
                        for index in range(first, first + local["training_rows"]))
    requests.sort()
    return windows(histories, requests), labels, details


def normalize(training, current):
    import numpy as np

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        train, observed = np.asarray(training, dtype=np.float64), np.asarray(current, dtype=np.float64)
        expected = (len(FEATURES), SESSIONS)
        if (train.ndim != 3 or observed.ndim != 3 or train.shape[1:] != expected or observed.shape[1:] != expected or
                not len(train) or not len(observed) or not np.isfinite(train).all() or not np.isfinite(observed).all()):
            raise ValueError("require nonempty finite [rows, 28 features, 20 sessions] matrices")
        mean, scale = train.mean(axis=(0, 2)), train.std(axis=(0, 2), ddof=0)
        scale = np.where(scale == 0, 1.0, scale)
        train = ((train - mean[None, :, None]) / scale[None, :, None]).astype(np.float32)
        observed = ((observed - mean[None, :, None]) / scale[None, :, None]).astype(np.float32)
    if not np.isfinite(train).all() or not np.isfinite(observed).all():
        raise ValueError("normalized sequences exceed finite float32 range")
    return train, observed, {"mean": mean.tolist(), "scale": scale.tolist(),
                             "scope": "training windows only; repeated sessions weighted by window occurrence"}


def make_network():
    import torch

    nn = torch.nn
    factory = {"device": "cpu", "dtype": torch.float32}
    return nn.Sequential(nn.Conv1d(len(FEATURES), 8, 3, padding=1, **factory), nn.ReLU(),
                         nn.Conv1d(8, 8, 3, padding=1, **factory), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
                         nn.Flatten(), nn.Linear(8, 1, **factory))


def fit_cnn(training, labels, current):
    import torch

    began = time.perf_counter()
    train, observed, scaler = normalize(training, current)
    if len(labels) != len(train) or len(labels) < 40 or any(type(label) is not int or label not in (0, 1) for label in labels):
        raise ValueError("CNN requires at least 40 binary training labels")
    diagnostics = {"normalizer": scaler, "training_rows": len(labels), "prediction_rows": len(observed),
                   "training_input_bytes": train.nbytes, "prediction_input_bytes": observed.nbytes,
                   "device": "cpu", "dtype": "float32"}
    if len(set(labels)) == 1:
        return [float(labels[0])] * len(observed), {**diagnostics, "fit_kind": "constant_single_class",
            "constant_probability": float(labels[0]), "parameter_count": 0, "parameter_bytes": 0,
            "runtime_seconds": time.perf_counter() - began}
    previous_threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(SEED)
            model = make_network()
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            loss_function = torch.nn.BCEWithLogitsLoss()
            x, y = torch.from_numpy(train), torch.tensor(labels, dtype=torch.float32, device="cpu")
            model.train()
            for _ in range(EPOCHS):
                weighted_loss = 0.0
                for start in range(0, len(labels), BATCH_SIZE):
                    end = min(start + BATCH_SIZE, len(labels))
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_function(model(x[start:end]).flatten(), y[start:end])
                    if not torch.isfinite(loss):
                        raise ValueError("CNN training loss is not finite")
                    loss.backward()
                    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                        raise ValueError("CNN gradient is not finite")
                    optimizer.step()
                    weighted_loss += float(loss.detach()) * (end - start)
            if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise ValueError("CNN parameters are not finite")
            model.eval()
            with torch.inference_mode():
                probabilities = torch.cat([torch.sigmoid(model(torch.from_numpy(observed[start:start + BATCH_SIZE])).flatten())
                                           for start in range(0, len(observed), BATCH_SIZE)]).tolist()
            diagnostics.update(fit_kind="native_cnn", parameter_count=sum(parameter.numel() for parameter in model.parameters()),
                               parameter_bytes=sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()),
                               epochs=EPOCHS, final_epoch_training_loss=weighted_loss / len(labels))
    except (RuntimeError, FloatingPointError) as error:
        raise ValueError(f"CNN fit/prediction failed; no fallback applied: {error}") from error
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(previous_threads)
    if len(probabilities) != len(observed) or any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError("CNN returned invalid probabilities")
    diagnostics["runtime_seconds"] = time.perf_counter() - began
    return probabilities, diagnostics


def environment():
    result = {"cpu_model": platform.processor() or None, "logical_cores": os.cpu_count(),
              "physical_cores": None, "ram_bytes": None}
    try:
        if sys.platform.startswith("linux"):
            records = [{key.strip(): value.strip() for line in block.splitlines() if ":" in line
                        for key, value in [line.split(":", 1)]}
                       for block in Path("/proc/cpuinfo").read_text().split("\n\n")]
            result["cpu_model"] = next((record["model name"] for record in records if "model name" in record), result["cpu_model"])
            cores = {(record["physical id"], record["core id"]) for record in records if "physical id" in record and "core id" in record}
            result["physical_cores"] = len(cores) or None
            result["ram_bytes"] = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        elif sys.platform == "darwin":
            for field, key in (("cpu_model", "machdep.cpu.brand_string"), ("physical_cores", "hw.physicalcpu"), ("ram_bytes", "hw.memsize")):
                value = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True, timeout=5).stdout.strip()
                result[field] = value if field == "cpu_model" else int(value)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return result


def peak_rss_bytes():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    except ImportError:
        return None


def evaluate(manifest, context_manifest):
    began = time.perf_counter()
    prepared = prepare_features(manifest, context_manifest)
    config = prepared["config"]
    if config["lookback_days"] < 80:
        raise ValueError("CNN requires lookback_days of at least 80")
    report = {key: value for key, value in prepared.items() if key not in ("results", "runtime_seconds", "phase")}
    report.update(phase="validation", task="binary_direction", training_mode="pooled_cnn", results=[], shared_fits=[],
                  aggregate_metrics={}, hardware=environment(),
                  procedure={"sequence_sessions": SESSIONS, "feature_channels": len(FEATURES),
                             "architecture": "Conv1d(28,8,3,pad=1)-ReLU-Conv1d(8,8,3,pad=1)-ReLU-global_mean-Linear(8,1)",
                             "epochs": EPOCHS, "batch_size": BATCH_SIZE, "optimizer": "AdamW",
                             "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "seed": SEED,
                             "device": "cpu", "threads": 1, "dtype": "float32", "deterministic_algorithms": True,
                             "row_order": "origin_date, symbol; no shuffle", "threshold": 0.5,
                             "normalization": "per-feature training-window mean/population std; zero variance scale one",
                             "training_targets": "strictly before common yearly fold start; same per-asset windows as Day 4",
                             "selection": "none; fixed architecture/epochs/seed, no early stopping or calibration"})
    report["source_hashes"].update({name: sha256(Path(__file__).with_name(name).read_bytes()) for name in
                                   ("neural_direction.py", "direction_forecasts.py", "requirements-neural.txt")})
    report["limitations"] = [item for item in prepared["limitations"] if not item.startswith("Features only:")] + [
        "Validation-only, one small randomly initialized CNN; no pretrained model, GPU benchmark or demonstrated trading edge.",
        "Inspected history and a survivor-selected universe are not a clean final test or point-in-time sample.",
        "Sequence windows and training labels overlap; assets and errors are dependent.",
        "No prediction grid is shifted or intersected; insufficient sequence history fails the run.",
        "Probabilities are uncalibrated; lower Brier alone is not proof of better reliability or price prediction.",
        "Peak RSS is process-lifetime high-water including interpreter/data/framework, not isolated model or GPU memory.",
        "Input/parameter bytes exclude optimizer state, gradients, activations, Python objects and framework overhead.",
        "Exact reproducibility is limited to the same configuration, libraries and implementation; no cross-platform guarantee.",
        "Paired within-stock bootstrap checks do not establish joint cross-stock significance or correct multiple experiments."]
    if report["status"] == "blocked":
        report["runtime_seconds"] = time.perf_counter() - began
        return report
    import torch
    report["dependencies"] = {name: version(name) for name in ("torch", "numpy")}
    report["torch_build_config"] = torch.__config__.show()
    histories = {group["symbol"]: group["points"] for group in prepared["results"]}
    schedules = {(symbol, horizon): origin_folds(points, config, horizon)
                 for symbol, points in histories.items() for horizon in HORIZONS}
    routed = {}
    for horizon in HORIZONS:
        for fold in sorted({fold for symbol in histories for fold in schedules[(symbol, horizon)]}):
            train, labels, training = training_sequences(histories, fold, horizon, config["lookback_days"])
            requests = sorted((histories[symbol][index]["origin_date"], symbol, index)
                              for symbol in histories for index in schedules[(symbol, horizon)].get(fold, []))
            probabilities, fitted = fit_cnn(train, labels, windows(histories, requests))
            fit_id = f"cnn:{horizon}d:{fold}"
            report["shared_fits"].append({"fit_id": fit_id, "fold_start": fold, "horizon_days": horizon,
                                         **training, "model": fitted})
            for (_, symbol, index), probability in zip(requests, probabilities):
                point, target = histories[symbol][index], histories[symbol][index + horizon]
                local = training["by_asset"][symbol]
                routed[(symbol, horizon, index)] = {"origin_date": point["origin_date"], "target_date": target["origin_date"],
                    "origin_index": point["origin_index"], "current_price": point["current_price"],
                    "actual_price": target["current_price"], "actual_up": int(target["current_price"] > point["current_price"]),
                    "fold_start": fold, "shared_fit_id": fit_id, "last_training_target": local["last_training_target"],
                    "last_training_target_index": local["last_training_target_index"],
                    "pooled_last_training_target": training["last_training_target"], "cnn": probability,
                    "always_up": 1.0, "latest_price_not_up": 0.0, "historical_up": local["historical_up_probability"]}
    settings = config.get("bootstrap", {"samples": 2000, "seed": 42, "block_sizes": [1, 2, 4]})
    all_points = []
    for group in prepared["results"]:
        symbol = group["symbol"]
        for horizon in HORIZONS:
            points = [routed[(symbol, horizon, index)] for indices in schedules[(symbol, horizon)].values() for index in indices]
            labels = [point["actual_up"] for point in points]
            report["results"].append({"symbol": symbol, "horizon_days": horizon, "forecasts": len(points),
                "metrics": {name: classification_metrics(labels, [point[name] for point in points]) for name in ("cnn", *BASELINES)},
                "bootstrap_sensitivity": paired_uncertainty(points, settings, models=("cnn",)), "points": points})
            all_points.extend(points)
    report["aggregate_metrics"] = {name: classification_metrics([point["actual_up"] for point in all_points],
        [point[name] for point in all_points]) for name in ("cnn", *BASELINES)}
    report["peak_process_rss_bytes"] = peak_rss_bytes()
    report["runtime_seconds"] = time.perf_counter() - began
    return report


def main():
    parser = argparse.ArgumentParser(description="Optional fixed CPU CNN stock-direction challenger")
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
    except (ValueError, TypeError, KeyError, OSError, UnicodeError, csv.Error, ImportError, FloatingPointError, OverflowError) as error:
        parser.exit(1, f"Neural direction experiment failed: {error}; see docs/neural-direction.md for CPU dependencies\n")
    if report["status"] != "ok":
        parser.exit(1, f"Neural direction blocked by data checks; see {args.output}\n")
    for group in report["results"]:
        print(f"{group['symbol']} {group['horizon_days']}d: n={group['forecasts']}, "
              f"CNN/always-up accuracy={group['metrics']['cnn']['accuracy']:.3f}/{group['metrics']['always_up']['accuracy']:.3f}")
    print(f"Runtime={report['runtime_seconds']:.2f}s, peak process RSS={report['peak_process_rss_bytes']} bytes; no default changed")


if __name__ == "__main__":
    main()
