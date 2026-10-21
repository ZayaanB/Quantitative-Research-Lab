import csv
from datetime import date, timedelta
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import direction_forecasts as direction
from benchmark_forecasts import sha256
from stock_features import CONTEXT_SYMBOLS, FEATURES, OHLCV, evaluate as prepare_features


HAS_MODELS = all(importlib.util.find_spec(name) is not None for name in ("sklearn", "lightgbm", "numpy"))


class DirectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.dates = []
        day = date(2020, 1, 1)
        while len(self.dates) < 340:
            if day.weekday() < 5:
                self.dates.append(day.isoformat())
            day += timedelta(days=1)
        settings = {"schema_version": 1, "lookback_days": 80, "validation_start": self.dates[120],
                    "holdout_start": self.dates[320], "test_end": self.dates[-1],
                    "bootstrap": {"samples": 100, "seed": 42, "block_sizes": [1, 2, 4]}}
        self.config = {**settings, "assets": []}
        self.context_config = {**settings, "assets": []}
        for position, symbol in enumerate(("WAVE", "FLAT", *CONTEXT_SYMBOLS)):
            path = self.directory / f"{symbol}.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("Date", "Adj Close", *OHLCV))
                for index, day in enumerate(self.dates):
                    price = 100 if symbol == "FLAT" else 100 * math.exp(
                        0.0002 * index + 0.04 * math.sin(index / (5 + position)) + 0.01 * math.cos(index * 0.7))
                    writer.writerow((day, price, price, price + 1, price - 1, price, 1000 + index))
            meta = self.directory / f"{symbol}.meta"
            meta.write_text(f"provider=synthetic\nsymbol={symbol}\nadjustment=none\n"
                            f"retrieved_on={self.dates[-1]}\nsource_url=local:fixture\n"
                            "price_column=Adj Close\nfrequency=daily\n")
            entry = {"symbol": symbol, "csv": path.name, "metadata": meta.name,
                     "csv_sha256": sha256(path.read_bytes()), "metadata_sha256": sha256(meta.read_bytes())}
            (self.context_config if symbol in CONTEXT_SYMBOLS else self.config)["assets"].append(entry)
        self.manifest = self.directory / "stocks.json"
        self.context_manifest = self.directory / "contexts.json"
        self.save_manifests()

    def save_manifests(self):
        self.manifest.write_text(json.dumps(self.config))
        self.context_manifest.write_text(json.dumps(self.context_config))

    def mutate(self, symbol, start, reserved=False):
        asset = next(asset for config in (self.config, self.context_config) for asset in config["assets"]
                     if asset["symbol"] == symbol)
        path = self.directory / asset["csv"]
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        for row in rows[start:]:
            for name in ("Adj Close", "Open", "High", "Low", "Close"):
                row[name] = str(float(row[name]) * 1.7)
            row["Volume"] = "nan" if reserved else "9000000"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        asset["csv_sha256"] = sha256(path.read_bytes())
        self.save_manifests()

    def evaluate(self):
        return direction.evaluate(self.manifest, self.context_manifest)

    def test_training_targets_and_features_stay_strictly_before_fold(self):
        points = prepare_features(self.manifest, self.context_manifest)["results"][0]["points"]
        for horizon in (1, 5, 20):
            x, y, details = direction.training_data(points, self.dates[120], horizon, 80)
            self.assertEqual(len(y), 80 - horizon)
            self.assertEqual(x[0], points[20]["features"])
            self.assertEqual(details["first_training_origin"], self.dates[40])
            self.assertEqual(details["last_training_target"], self.dates[119])
            self.assertEqual(details["last_training_target_index"], 119)
            self.assertEqual(y[-1], int(points[99]["current_price"] > points[99 - horizon]["current_price"]))
            modified = [dict(point) for point in points]
            for point in modified[100:]:
                point["current_price"] *= 100
                point["features"] = [999.0] * len(FEATURES)
            self.assertEqual((x, y, details), direction.training_data(modified, self.dates[120], horizon, 80))
        for horizon, lookback, fold in ((0, 80, self.dates[120]), (1, 79, self.dates[120]),
                                         (20, 80, self.dates[40])):
            with self.assertRaises(ValueError):
                direction.training_data(points, fold, horizon, lookback)

    def test_metrics_hand_calculated_and_edge_cases(self):
        result = direction.classification_metrics([0, 1, 1, 0], [0.1, 0.9, 0.4, 0.8])
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(result["brier_score"], (0.01 + 0.01 + 0.36 + 0.64) / 4)
        self.assertAlmostEqual(result["log_loss"], -sum(map(math.log, (0.9, 0.9, 0.4, 0.2))) / 4)
        self.assertEqual(result["confusion"], {"true_up": 1, "true_not_up": 1, "false_up": 1, "false_not_up": 1})
        self.assertEqual(sum(item["count"] for item in result["reliability_bins"]), 4)
        self.assertIsNone(direction.classification_metrics([0, 0], [0, 0])["balanced_accuracy"])
        self.assertTrue(math.isfinite(direction.classification_metrics([0, 1], [1, 0])["log_loss"]))
        self.assertEqual(direction.classification_metrics([1], [0.5])["accuracy"], 1)
        for labels, probabilities in (([], []), ([0], []), ([2], [0.5]), ([0], [float("nan")]),
                                       ([0], [1.1]), ([True], [0.5])):
            with self.assertRaises(ValueError):
                direction.classification_metrics(labels, probabilities)

    def test_bad_data_block_before_any_fit_or_dependency_lookup(self):
        self.mutate("WAVE", 30, reserved=True)
        with patch.object(direction, "fit_predict", side_effect=AssertionError("model ran")), \
                patch.object(direction, "version", side_effect=AssertionError("dependencies checked")):
            report = self.evaluate()
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["results"], [])

    @unittest.skipUnless(HAS_MODELS, "optional direction dependencies are not installed")
    def test_native_fit_scaler_uses_training_only_and_is_deterministic(self):
        x = [[math.sin(index / 5 + column) for column in range(len(FEATURES))] for index in range(80)]
        y = [index % 2 for index in range(80)]
        first, details = direction.fit_predict(x, y, [x[-1]])
        repeated, again = direction.fit_predict(x, y, [x[-1], [1e8] * len(FEATURES)])
        self.assertEqual(details, again)
        for name in direction.MODELS:
            self.assertEqual(first[name][0], repeated[name][0])
            self.assertTrue(0 <= first[name][0] <= 1)
        for column, value in enumerate(details["logistic"]["scaler_mean"]):
            self.assertAlmostEqual(value, sum(row[column] for row in x) / len(x))
        flat, details = direction.fit_predict(x, [0] * 80, [x[-1]])
        self.assertEqual(flat, {"logistic": [0.0], "lightgbm": [0.0]})
        self.assertTrue(all(info["fit_kind"] == "constant_single_class" for info in details.values()))
        with self.assertRaises(ValueError):
            direction.fit_predict(x[:-1], y, [x[-1]])

    @unittest.skipUnless(HAS_MODELS, "optional direction dependencies are not installed")
    def test_native_failures_and_nonfinite_probabilities_never_fall_back(self):
        import lightgbm
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
        x = [[math.sin(index + column) for column in range(len(FEATURES))] for index in range(80)]
        y = [index % 2 for index in range(80)]
        with patch.object(lightgbm, "train", side_effect=lightgbm.basic.LightGBMError("simulated")):
            with self.assertRaisesRegex(ValueError, "no fallback"):
                direction.fit_predict(x, y, [x[-1]])
        with patch.object(LogisticRegression, "fit", side_effect=ConvergenceWarning("not converged")):
            with self.assertRaisesRegex(ValueError, "not converged"):
                direction.fit_predict(x, y, [x[-1]])
        with patch.object(lightgbm.Booster, "predict", return_value=__import__("numpy").array([float("nan")])):
            with self.assertRaisesRegex(ValueError, "invalid probability"):
                direction.fit_predict(x, y, [x[-1]])

    @unittest.skipUnless(HAS_MODELS, "optional direction dependencies are not installed")
    def test_yearly_folds_matched_origins_and_baselines(self):
        report = self.evaluate()
        self.assertEqual(report["task"], "binary_direction")
        self.assertEqual(len(report["results"]), 6)
        for group in report["results"]:
            horizon = group["horizon_days"]
            self.assertEqual([point["origin_index"] for point in group["points"]],
                             list(range(120, 320 - horizon, horizon)))
            self.assertEqual(len(group["folds"]), 2)
            self.assertEqual(set(group["metrics"]), {*direction.MODELS, *direction.BASELINES})
            for point in group["points"]:
                self.assertLess(point["target_date"], self.dates[320])
                self.assertLess(point["last_training_target"], point["fold_start"])
                self.assertLessEqual(point["fold_start"], point["origin_date"])
                self.assertEqual(point["actual_up"], int(point["actual_price"] > point["current_price"]))
                self.assertEqual(point["always_up"], 1.0)
                self.assertEqual(point["latest_price_not_up"], 0.0)
            if group["symbol"] == "FLAT":
                self.assertEqual(group["metrics"]["logistic"]["accuracy"], 1)
                self.assertEqual(group["metrics"]["always_up"]["accuracy"], 0)
                self.assertIsNone(group["metrics"]["logistic"]["balanced_accuracy"])
        self.assertEqual(sum(group["forecasts"] for group in report["results"]),
                         report["aggregate_metrics"]["logistic"]["samples"])

    @unittest.skipUnless(HAS_MODELS, "optional direction dependencies are not installed")
    def test_reserved_and_future_mutations_preserve_earlier_predictions(self):
        original = self.evaluate()
        for symbol in ("WAVE", "FLAT", *CONTEXT_SYMBOLS):
            self.mutate(symbol, 320, reserved=True)
        self.assertEqual(original["results"], self.evaluate()["results"])
        for symbol in ("WAVE", *CONTEXT_SYMBOLS):
            self.mutate(symbol, 180)
        changed = self.evaluate()
        for before, after in zip(original["results"], changed["results"]):
            self.assertEqual(before["folds"][0], after["folds"][0])
            for first, second in zip(before["points"], after["points"]):
                if first["target_date"] < self.dates[180]:
                    self.assertEqual(first, second)

    @unittest.skipUnless(HAS_MODELS, "optional direction dependencies are not installed")
    def test_bootstrap_constant_differences_and_small_samples(self):
        points = [{"actual_up": index % 2, "logistic": float(index % 2), "lightgbm": 1.0,
                   "historical_up": 0.5} for index in range(83)]
        settings = {"samples": 100, "seed": 42, "block_sizes": [1, 2, 4, 100]}
        result = direction.paired_uncertainty(points, settings)
        self.assertEqual(result, direction.paired_uncertainty(points, settings))
        for item in result[:-1]:
            zero = item["comparisons"]["lightgbm_accuracy_gain_vs_always_up"]
            self.assertEqual((zero["mean"], zero["lower_95"], zero["upper_95"]), (0, 0, 0))
            better = item["comparisons"]["logistic_brier_reduction_vs_historical_up"]
            self.assertEqual((better["mean"], better["lower_95"], better["upper_95"]), (0.25, 0.25, 0.25))
        self.assertTrue(all(item["lower_95"] is None for item in result[-1]["comparisons"].values()))

    @unittest.skipUnless(HAS_MODELS, "optional direction dependencies are not installed")
    def test_cli_preserves_outputs_and_reports_blocked_input(self):
        output = self.directory / "direction.json"
        command = [sys.executable, str(ROOT / "python/direction_forecasts.py"), "--manifest", str(self.manifest),
                   "--context-manifest", str(self.context_manifest), "--output", str(output)]
        process = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        saved = output.read_bytes()
        self.assertEqual(json.loads(saved)["status"], "ok")
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.assertEqual(output.read_bytes(), saved)
        self.mutate("WAVE", 30, reserved=True)
        command[-1] = str(self.directory / "blocked.json")
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.assertEqual(json.loads(Path(command[-1]).read_bytes())["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
