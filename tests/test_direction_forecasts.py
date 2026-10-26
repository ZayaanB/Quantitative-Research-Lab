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

    def evaluate_pooled(self):
        return direction.evaluate(self.manifest, self.context_manifest, "pooled")

    def test_pooled_rows_are_date_sorted_with_common_cutoffs_and_per_asset_counts(self):
        prepared = prepare_features(self.manifest, self.context_manifest)
        histories = {group["symbol"]: group["points"] for group in prepared["results"]}
        fold = self.dates[120]
        for horizon in (1, 5, 20):
            x, y, details = direction.pooled_training_data(histories, fold, horizon, 80)
            expected = []
            for symbol, points in histories.items():
                local_x, local_y, local = direction.training_data(points, fold, horizon, 80)
                self.assertEqual(details["by_asset"][symbol], local)
                dates = [point["origin_date"] for point in points if
                         local["first_training_origin"] <= point["origin_date"] <= local["last_training_origin"]]
                expected.extend(zip(dates, [symbol] * len(dates), local_x, local_y))
            expected.sort(key=lambda row: (row[0], row[1]))
            self.assertEqual(x, [row[2] for row in expected])
            self.assertEqual(y, [row[3] for row in expected])
            self.assertEqual(details["training_rows"], 2 * (80 - horizon))
            self.assertLess(details["last_training_target"], fold)
            self.assertEqual((x, y, details), direction.pooled_training_data(
                dict(reversed(list(histories.items()))), fold, horizon, 80))

    def test_pooled_cutoff_is_a_date_not_an_index_shared_across_assets(self):
        prepared = prepare_features(self.manifest, self.context_manifest)
        histories = {group["symbol"]: group["points"] for group in prepared["results"]}
        histories["FLAT"] = [{**point, "origin_index": point["origin_index"] - 30}
                              for point in histories["FLAT"][30:]]
        original = direction.pooled_training_data(histories, self.dates[120], 20, 80)
        details = original[2]
        self.assertEqual(details["by_asset"]["WAVE"]["last_training_target_index"], 119)
        self.assertEqual(details["by_asset"]["FLAT"]["last_training_target_index"], 89)
        self.assertEqual({item["last_training_target"] for item in details["by_asset"].values()}, {self.dates[119]})
        for points in histories.values():
            for point in points:
                if point["origin_date"] >= self.dates[120]:
                    point["current_price"] *= 100
                    point["features"] = [999.0] * len(FEATURES)
        self.assertEqual(original, direction.pooled_training_data(histories, self.dates[120], 20, 80))

    def test_pooled_training_does_not_skip_assets_with_insufficient_labels(self):
        prepared = prepare_features(self.manifest, self.context_manifest)
        histories = {group["symbol"]: group["points"] for group in prepared["results"]}
        histories["FLAT"] = histories["FLAT"][90:]
        with self.assertRaisesRegex(ValueError, "FLAT.*no asset was dropped"):
            direction.pooled_training_data(histories, self.dates[120], 20, 80)
        with self.assertRaisesRegex(ValueError, "at least two assets"):
            direction.pooled_training_data({"WAVE": histories["WAVE"]}, self.dates[120], 1, 80)
        with self.assertRaisesRegex(ValueError, "training_mode"):
            direction.evaluate(self.manifest, self.context_manifest, "invalid")

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
    def test_single_model_search_controls_preserve_default_fits(self):
        import lightgbm
        from sklearn.linear_model import LogisticRegression
        x = [[math.sin(index / 5 + column) for column in range(len(FEATURES))] for index in range(80)]
        y = [index % 2 for index in range(80)]
        both, diagnostics = direction.fit_predict(x, y, [x[-1]])
        for name in direction.MODELS:
            single, fitted = direction.fit_predict(x, y, [x[-1]], model_names=(name,))
            self.assertEqual(single, {name: both[name]})
            self.assertEqual(fitted, {name: diagnostics[name]})
        with patch("sklearn.linear_model.LogisticRegression", wraps=LogisticRegression) as fitted:
            direction.fit_predict(x, y, [x[-1]], model_names=("logistic",), logistic_parameters={"C": 0.1})
            self.assertEqual(fitted.call_args.kwargs["C"], 0.1)
        with patch.object(lightgbm, "train", wraps=lightgbm.train) as fitted:
            direction.fit_predict(x, y, [x[-1]], model_names=("lightgbm",), binary_parameters={"lambda_l2": 10.0})
            self.assertEqual(fitted.call_args.args[0]["lambda_l2"], 10.0)
        self.assertEqual(direction.LOGISTIC_PARAMETERS["C"], 1.0)
        self.assertEqual(direction.BINARY_PARAMETERS["lambda_l2"], 1.0)
        with self.assertRaisesRegex(ValueError, "supported"):
            direction.fit_predict(x, y, [x[-1]], model_names=("unknown",))

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

    @unittest.skipUnless(HAS_MODELS, "optional direction dependencies are not installed")
    def test_pooled_fits_once_per_horizon_fold_with_training_only_scaling(self):
        with patch.object(direction, "fit_predict", wraps=direction.fit_predict) as fitted:
            report = self.evaluate_pooled()
        self.assertEqual(report["training_mode"], "pooled")
        self.assertEqual(fitted.call_count, 6)
        self.assertEqual(len(report["shared_fits"]), 6)
        self.assertEqual(len(report["results"]), 6)
        shared = {item["fit_id"]: item for item in report["shared_fits"]}
        for fit, call in zip(report["shared_fits"], fitted.call_args_list):
            x, y, current = call.args
            self.assertEqual(len(y), fit["training_rows"])
            self.assertEqual(len(current), fit["predictions"])
            self.assertEqual(fit["training_rows"], sum(item["training_rows"] for item in fit["by_asset"].values()))
            self.assertEqual(set(fit["by_asset"]), {"WAVE", "FLAT"})
            self.assertLess(fit["last_training_target"], fit["fold_start"])
            for column, value in enumerate(fit["models"]["logistic"]["scaler_mean"]):
                self.assertAlmostEqual(value, sum(row[column] for row in x) / len(x))
        separate = self.evaluate()
        for pooled, local in zip(report["results"], separate["results"]):
            for before, after in zip(pooled["points"], local["points"]):
                for key in ("origin_date", "target_date", "actual_up", "current_price", "actual_price",
                            "historical_up", "always_up", "latest_price_not_up"):
                    self.assertEqual(before[key], after[key])
                self.assertLess(before["pooled_last_training_target"], before["fold_start"])
                self.assertIn(before["shared_fit_id"], shared)
            for name in direction.BASELINES:
                self.assertEqual(pooled["metrics"][name], local["metrics"][name])

    @unittest.skipUnless(HAS_MODELS, "optional direction dependencies are not installed")
    def test_pooled_predictions_do_not_depend_on_manifest_order(self):
        original = self.evaluate_pooled()
        self.config["assets"].reverse()
        self.save_manifests()
        changed = self.evaluate_pooled()
        by_key = lambda report: {(group["symbol"], group["horizon_days"]): group for group in report["results"]}
        self.assertEqual(by_key(original), by_key(changed))
        by_id = lambda report: {fit["fit_id"]: fit for fit in report["shared_fits"]}
        self.assertEqual(by_id(original), by_id(changed))
        self.assertEqual(original["aggregate_metrics"], changed["aggregate_metrics"])

    @unittest.skipUnless(HAS_MODELS, "optional direction dependencies are not installed")
    def test_pooled_reserved_and_other_asset_future_data_do_not_leak(self):
        original = self.evaluate_pooled()
        for symbol in ("WAVE", "FLAT", *CONTEXT_SYMBOLS):
            self.mutate(symbol, 320, reserved=True)
        reserved = self.evaluate_pooled()
        self.assertEqual(original["results"], reserved["results"])
        self.assertEqual(original["shared_fits"], reserved["shared_fits"])
        self.mutate("FLAT", 180)
        changed = self.evaluate_pooled()
        for before, after in zip(original["results"], changed["results"]):
            for first, second in zip(before["points"], after["points"]):
                if first["target_date"] < self.dates[180]:
                    self.assertEqual(first, second)
        for before, after in zip(original["shared_fits"], changed["shared_fits"]):
            if before["fold_start"] == self.dates[120]:
                self.assertEqual(before, after)

    @unittest.skipUnless(HAS_MODELS, "optional direction dependencies are not installed")
    def test_pooled_cli_flag_no_overwrite_and_no_fit_fallback(self):
        output = self.directory / "pooled.json"
        command = [sys.executable, str(ROOT / "python/direction_forecasts.py"), "--manifest", str(self.manifest),
                   "--context-manifest", str(self.context_manifest), "--training-mode", "pooled", "--output", str(output)]
        process = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        saved = output.read_bytes()
        self.assertEqual(json.loads(saved)["training_mode"], "pooled")
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.assertEqual(saved, output.read_bytes())
        with patch.object(direction, "fit_predict", side_effect=ValueError("simulated failure")):
            with self.assertRaisesRegex(ValueError, "simulated failure"):
                self.evaluate_pooled()


if __name__ == "__main__":
    unittest.main()
