import csv
import copy
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


from stock_fixtures import weekday_dates, write_price_asset


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("benchmark_forecasts", ROOT / "python/benchmark_forecasts.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
sys.path.insert(0, str(ROOT / "python"))
import compare_forecasts
import ridge_forecasts
import ensemble_forecasts


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.dates = weekday_dates(180)
        self.prices = [100 * math.exp(0.0003 * index + 0.01 * math.sin(index)) for index in range(180)]
        self.config = {"schema_version": 1, "lookback_days": 10,
                       "bootstrap": {"samples": 100, "seed": 42, "block_sizes": [1, 2, 4]},
                       "validation_start": self.dates[20], "holdout_start": self.dates[100],
                       "test_end": self.dates[-1], "assets": []}
        for symbol, prices in (("SYNTH_A", self.prices), ("SYNTH_B", [100.0] * 180)):
            self.config["assets"].append(write_price_asset(self.directory, symbol, self.dates, prices))
        self.manifest = self.directory / "manifest.json"
        self.save()
        self.engine = Path(os.environ.get("MC_ENGINE", ROOT / "build/mcprice")).resolve()

    def save(self):
        self.manifest.write_text(json.dumps(self.config), encoding="utf-8")

    def test_ensemble_blend_and_metrics(self):
        for magnitude in (1e308, 1e-200, 5e-324):
            self.assertEqual(benchmark.root_mean_square([magnitude] * 4), magnitude)
        for values in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                benchmark.root_mean_square(values)
        extreme = [{"current_price": 1, "forecast_price": 1e308, "actual_price": 1}] * 4
        self.assertAlmostEqual(ensemble_forecasts.point_metrics(extreme, "forecast_price")["rmse"] / 1e308, 1)
        self.assertEqual(ensemble_forecasts.blend(100, 120), 110)
        self.assertEqual(ensemble_forecasts.blend(120, 100), 110)
        for price in (5e-324, 100, sys.float_info.max):
            self.assertEqual(ensemble_forecasts.blend(price, price), price)
        self.assertTrue(math.isfinite(ensemble_forecasts.blend(sys.float_info.max, 1)))
        for price in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                ensemble_forecasts.blend(100, price)
        points = [{"current_price": 100, "forecast_price": 110, "actual_price": 120},
                  {"current_price": 100, "forecast_price": 90, "actual_price": 100}]
        metrics = ensemble_forecasts.point_metrics(points, "forecast_price")
        self.assertEqual(metrics["mae"], 10)
        self.assertAlmostEqual(metrics["rmse"], 10)
        self.assertEqual(metrics["directional_accuracy"], 0.5)
        self.assertIsNone(ensemble_forecasts.point_metrics(points, "current_price")["directional_accuracy"])
        with self.assertRaises(ValueError):
            ensemble_forecasts.point_metrics([], "forecast_price")

    def test_ensemble_selection_including_perfect_baseline(self):
        for actual, winner in ((100, "latest_price"), (110, "ensemble"), (120, "ridge")):
            source = {"model": {"name": "ridge"}, "source_sha256": "original",
                      "limitations": [], "results": [{"points": [{"current_price": 100,
                      "forecast_price": 120, "actual_price": actual, "predicted_log_return": 0.1}]}]}
            with patch.object(ensemble_forecasts, "evaluate_ridge", return_value=copy.deepcopy(source)):
                report = ensemble_forecasts.evaluate(self.manifest)
            self.assertEqual(report["selected"], winner)
            self.assertEqual(report["ridge_source_sha256"], "original")
            if actual == 100:
                self.assertIsNone(report["ranking"][1]["mean_relative_mae"])
        source["results"][0]["points"][0].update(forecast_price=100, actual_price=100)
        with patch.object(ensemble_forecasts, "evaluate_ridge", return_value=source):
            self.assertEqual(ensemble_forecasts.evaluate(self.manifest)["selected"], "latest_price")

    def test_ensemble_cli_and_paired_validation(self):
        self.config["lookback_days"] = 60
        self.config["validation_start"] = self.dates[65]
        self.save()
        ridge = ridge_forecasts.evaluate(self.manifest)
        report = ensemble_forecasts.evaluate(self.manifest)
        self.assertEqual(report, ensemble_forecasts.evaluate(self.manifest))
        for original, group in zip(ridge["results"], report["results"]):
            self.assertEqual(original["metrics"]["ridge"]["mae"], group["metrics"]["ridge"]["mae"])
            for before, after in zip(original["points"], group["points"]):
                self.assertEqual(before["forecast_price"], after["ridge_price"])
                self.assertEqual(after["forecast_price"],
                                 ensemble_forecasts.blend(before["current_price"], before["forecast_price"]))
                self.assertLess(after["target_date"], self.config["holdout_start"])
                self.assertEqual(before["origin_date"], after["origin_date"])
                self.assertNotIn("predicted_log_return", after)
        output = self.directory / "ensemble.json"
        command = [sys.executable, str(ROOT / "python/ensemble_forecasts.py"),
                   "--manifest", str(self.manifest), "--output", str(output)]
        process = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(report, json.loads(output.read_text()))
        prior = output.read_bytes()
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.assertEqual(prior, output.read_bytes())

    def test_ridge_matches_closed_form_and_handles_constant_features(self):
        self.assertAlmostEqual(ridge_forecasts.ridge_predict([[0], [1], [2]], [1, 3, 5], [3]),
                               6.0)
        self.assertEqual(ridge_forecasts.ridge_predict([[4], [4]], [2, 6], [4]), 4)
        self.assertEqual(ridge_forecasts.ridge_predict([[0, 0], [1, 1], [2, 2]],
                         [1, 3, 5], [1, 1]), 3)
        features = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
        targets = [3 + 2 * x - 4 * y for x, y in features]
        self.assertAlmostEqual(ridge_forecasts.ridge_predict(features, targets, [2, 3]), -3.4)
        for penalty in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                ridge_forecasts.ridge_predict([[0], [1]], [1, 2], [3], penalty)
        with self.assertRaises(ValueError):
            ridge_forecasts.ridge_predict([[0], [float("nan")]], [1, 2], [3])

    def test_ridge_forecast_flat_history_and_training_boundary(self):
        result = ridge_forecasts.forecast([100.0] * 100, 20, 60)
        self.assertEqual(result["forecast_price"], 100)
        self.assertEqual(result["predicted_log_return"], 0)
        self.assertEqual(result["last_training_target_index"], 99)
        self.assertEqual(result["training_rows"], 41)
        for prices, horizon, window in (([100] * 20, 20, 60), ([100] * 100, 0, 60),
                                       ([100] * 100, 1, 10), ([100, 0], 1, 60)):
            with self.assertRaises(ValueError):
                ridge_forecasts.forecast(prices, horizon, window)

    def test_ridge_validation_is_deterministic_and_ignores_future_prices(self):
        self.config["lookback_days"] = 60
        self.config["validation_start"] = self.dates[65]
        self.save()
        original = ridge_forecasts.evaluate(self.manifest)
        original_ensemble = ensemble_forecasts.evaluate(self.manifest)
        self.assertEqual(original, ridge_forecasts.evaluate(self.manifest))
        output = self.directory / "ridge-validation.json"
        process = subprocess.run([sys.executable, str(ROOT / "python/ridge_forecasts.py"),
                                 "--manifest", str(self.manifest), "--output", str(output)],
                                 capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(original, json.loads(output.read_text()))
        for group in original["results"]:
            errors = [abs(p["forecast_price"] - p["actual_price"]) for p in group["points"]]
            self.assertAlmostEqual(group["metrics"]["ridge"]["mae"], statistics.mean(errors))
            for point in group["points"]:
                self.assertLess(point["target_date"], self.config["holdout_start"])
            if group["symbol"] == "SYNTH_B":
                self.assertEqual(group["metrics"]["ridge"]["mae"], 0)
        asset = self.config["assets"][0]
        path = self.directory / asset["csv"]
        with path.open() as stream:
            rows = list(csv.reader(stream))
        for index, row in enumerate(rows[1:]):
            if index >= 100:
                row[1] = str(float(row[1]) * 3)
        with path.open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)
        asset["csv_sha256"] = benchmark.sha256(path.read_bytes())
        self.save()
        self.assertEqual(original["results"], ridge_forecasts.evaluate(self.manifest)["results"])
        self.assertEqual(original_ensemble["results"], ensemble_forecasts.evaluate(self.manifest)["results"])
        for index, row in enumerate(rows[1:]):
            if index >= 80:
                row[1] = str(float(row[1]) * 2)
        with path.open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)
        asset["csv_sha256"] = benchmark.sha256(path.read_bytes())
        self.save()
        changed = ridge_forecasts.evaluate(self.manifest)
        changed_ensemble = ensemble_forecasts.evaluate(self.manifest)
        for before, after in zip(original_ensemble["results"], changed_ensemble["results"]):
            earlier = {p["origin_date"]: p for p in after["points"]}
            for point in before["points"]:
                if point["origin_date"] < self.dates[80]:
                    self.assertEqual(point["forecast_price"], earlier[point["origin_date"]]["forecast_price"])
        for before, after in zip(original["results"], changed["results"]):
            earlier = {p["origin_date"]: p for p in after["points"]}
            for point in before["points"]:
                if point["origin_date"] < self.dates[80]:
                    prediction = earlier[point["origin_date"]]
                    self.assertEqual(point["forecast_price"], prediction["forecast_price"])
                    self.assertEqual(point["predicted_log_return"], prediction["predicted_log_return"])

    def test_ridge_cli_preserves_existing_output(self):
        output = self.directory / "ridge.json"
        output.write_text("keep")
        process = subprocess.run([sys.executable, str(ROOT / "python/ridge_forecasts.py"),
                                 "--manifest", str(self.manifest), "--output", str(output)],
                                 capture_output=True, text=True)
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(output.read_text(), "keep")

    def test_comparison_uses_common_validation_origins_and_baseline(self):
        execute = subprocess.run

        def checked(command, **kwargs):
            history = Path(command[command.index("--csv") + 1])
            with history.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(all(row["Date"] < self.config["holdout_start"] for row in rows))
            return execute(command, **kwargs)

        with patch("subprocess.run", side_effect=checked):
            report = compare_forecasts.compare(self.manifest, self.engine, lookbacks=(10, 13))
        self.assertEqual(len(report["candidates"]), 12)
        self.assertEqual(len(report["ranking"]), 13)
        self.assertEqual(report["phase"], "validation")
        for group in report["groups"]:
            self.assertTrue(group["origins"])
            self.assertTrue(all(self.config["validation_start"] <= day < self.config["holdout_start"]
                                for day in group["origins"]))
            self.assertEqual(len(group["metrics"]), 12)
            baselines = [metric["latest_price"]["mae"] for metric in group["metrics"]]
            self.assertTrue(all(value == baselines[0] for value in baselines))
        scores = [entry["mean_relative_mae"] for entry in report["ranking"]
                  if entry["mean_relative_mae"] is not None]
        chosen = next(entry for entry in report["ranking"] if entry["candidate"] == report["selected"])
        self.assertEqual(chosen["mean_relative_mae"], min(scores))
        self.assertEqual(report, compare_forecasts.compare(self.manifest, self.engine, lookbacks=(10, 13)))

    def test_comparison_rejects_invalid_windows(self):
        for windows in ((), (10, 10), (True,), (1,)):
            with self.assertRaises(ValueError):
                compare_forecasts.compare(self.manifest, self.engine, lookbacks=windows)

    def test_manifest_rejects_ambiguous_or_unfrozen_settings(self):
        for key, value in (("lookback_days", True), ("lookback_days", 1),
                           ("validation_start", "20200101"), ("test_end", "2019-01-01"),
                           ("schema_version", 2), ("assets", self.config["assets"][:1])):
            original = self.config[key]
            self.config[key] = value
            self.save()
            with self.assertRaises(ValueError):
                benchmark.read_manifest(self.manifest)
            self.config[key] = original
        self.manifest.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
        with self.assertRaises(ValueError):
            benchmark.read_manifest(self.manifest)

    def test_checksums_reject_changed_history_and_metadata(self):
        for key in ("csv", "metadata"):
            path = self.directory / self.config["assets"][0][key]
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                benchmark.read_asset(self.config["assets"][0], self.directory)
            path.write_bytes(original)

    def test_summary_treats_unchanged_actual_as_neither_up_nor_down(self):
        points = [{"current_price": 100, "forecast_price": 99, "actual_price": 100,
                   "latest_price_forecast": 100, "probability_above_current": 0.4,
                   "lower_95": 95, "upper_95": 105}]
        result = benchmark.summarize(points)
        self.assertEqual(result["selected"]["directional_accuracy"], 0)
        self.assertIsNone(result["latest_price"]["directional_accuracy"])
        self.assertIsNone(result["relative_mae_improvement"])
        self.assertAlmostEqual(result["brier_score"], 0.16)
        self.assertEqual(result["always_up_accuracy"], 0)
        self.assertEqual(result["always_up_brier_score"], 1)
        self.assertEqual(result["half_brier_score"], 0.25)
        self.assertIsNone(result["selected"]["mase"])
        with self.assertRaises(ValueError):
            benchmark.summarize([])

    def test_scaled_errors_and_probability_baselines_have_known_values(self):
        points = [{"current_price": 100, "forecast_price": 102, "actual_price": 104,
                   "latest_price_forecast": 100, "probability_above_current": 0.75,
                   "lower_95": 95, "upper_95": 105, "naive_error_scale": 2,
                   "historical_up_probability": 0.5}]
        result = benchmark.summarize(points)
        self.assertEqual(result["selected"]["mase"], 1)
        self.assertEqual(result["latest_price"]["mase"], 2)
        self.assertEqual(result["historical_up_brier_score"], 0.25)
        self.assertEqual(result["always_up_accuracy"], 1)
        self.assertEqual(result["always_up_brier_score"], 0)

    def test_bootstrap_is_deterministic_and_does_not_infer_from_tiny_samples(self):
        points = [{"latest_price_forecast": 100, "forecast_price": 101, "actual_price": 102}] * 40
        settings = {"samples": 100, "seed": 42, "block_sizes": [1, 2, 8]}
        result = benchmark.bootstrap_sensitivity(points, settings)
        self.assertEqual(result, benchmark.bootstrap_sensitivity(points, settings))
        self.assertEqual(result[0]["lower_95"], 1)
        self.assertEqual(result[0]["conclusion"], "better")
        self.assertIsNone(result[-1]["lower_95"])
        self.assertEqual(result[-1]["conclusion"], "inconclusive")

    def test_manifest_rejects_invalid_bootstrap_settings(self):
        for settings in ({"samples": 99, "seed": 42, "block_sizes": [1]},
                         {"samples": 100, "seed": -1, "block_sizes": [1]},
                         {"samples": 100, "seed": 42, "block_sizes": [0]},
                         {"samples": 100, "seed": 42, "block_sizes": [1, 1]}):
            self.config["bootstrap"] = settings
            self.save()
            with self.assertRaises(ValueError):
                benchmark.read_manifest(self.manifest)

    def test_validation_excludes_holdout_from_engine_inputs_and_matches_reference(self):
        real_run = subprocess.run

        def inspect(command, **kwargs):
            history = Path(command[command.index("--csv") + 1])
            with history.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(all(row["Date"] < self.config["holdout_start"] for row in rows))
            return real_run(command, **kwargs)

        with patch.object(benchmark.subprocess, "run", side_effect=inspect):
            report = benchmark.run_benchmark(self.manifest, self.engine, "validation")
        self.assertEqual(len(report["results"]), 6)
        self.assertEqual(report, benchmark.run_benchmark(self.manifest, self.engine, "validation"))
        for result in report["results"]:
            horizon = result["horizon_days"]
            self.assertLess(result["last_target"], self.config["holdout_start"])
            if result["symbol"] == "SYNTH_B":
                self.assertEqual(result["metrics"]["selected"]["mae"], 0)
                self.assertIsNone(result["metrics"]["selected"]["directional_accuracy"])
                continue
            errors = []
            for point in result["points"]:
                origin = self.dates.index(point["origin_date"])
                training = self.prices[origin - 10:origin + 1]
                self.assertAlmostEqual(point["naive_error_scale"], statistics.mean(
                    abs(b - a) for a, b in zip(training, training[1:])), places=12)
                expected_probability = (statistics.mean(
                    training[index] > training[index - horizon]
                    for index in range(horizon, len(training))) if horizon <= 10 else None)
                if expected_probability is None:
                    self.assertIsNone(point["historical_up_probability"])
                else:
                    self.assertAlmostEqual(point["historical_up_probability"], expected_probability, places=14)
                returns = [math.log(b) - math.log(a) for a, b in zip(training, training[1:])]
                forecast = training[-1] * math.exp(horizon * (
                    statistics.mean(returns) + statistics.variance(returns) / 2))
                self.assertAlmostEqual(point["forecast_price"], forecast, places=11)
                errors.append(abs(forecast - point["actual_price"]))
            self.assertAlmostEqual(result["metrics"]["selected"]["mae"], statistics.mean(errors), places=11)

    def test_holdout_is_explicit_and_stays_inside_test_dates(self):
        report = benchmark.run_benchmark(self.manifest, self.engine, "holdout")
        for result in report["results"]:
            self.assertGreaterEqual(result["first_origin"], self.config["holdout_start"])
            self.assertLessEqual(result["last_target"], self.config["test_end"])
            points = result["points"]
            for previous, following in zip(points, points[1:]):
                self.assertGreaterEqual(following["origin_date"], previous["target_date"])

    def test_cli_preserves_existing_output(self):
        output = self.directory / "result.json"
        output.write_text("keep me", encoding="utf-8")
        result = subprocess.run([sys.executable, str(ROOT / "python/benchmark_forecasts.py"),
                                 "--manifest", str(self.manifest), "--engine", str(self.engine),
                                 "--output", str(output)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(output.read_text(), "keep me")

    def test_cli_success_saves_default_validation_and_provenance(self):
        output = self.directory / "validation.json"
        process = subprocess.run([sys.executable, str(ROOT / "python/benchmark_forecasts.py"),
                                 "--manifest", str(self.manifest), "--engine", str(self.engine),
                                 "--output", str(output)], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        report = json.loads(output.read_text())
        self.assertEqual(report["phase"], "validation")
        self.assertEqual(report["engine_sha256"], benchmark.sha256(self.engine.read_bytes()))
        self.assertEqual(report["manifest_sha256"], benchmark.sha256(self.manifest.read_bytes()))
        self.assertEqual(len(report["results"]), 6)

    def test_changes_in_holdout_prices_cannot_change_validation_predictions(self):
        original = benchmark.run_benchmark(self.manifest, self.engine, "validation")
        history = self.directory / self.config["assets"][0]["csv"]
        with history.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            if row["Date"] >= self.config["holdout_start"]:
                row["Adj Close"] = str(float(row["Adj Close"]) * 2)
        with history.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["Date", "Adj Close"])
            writer.writeheader()
            writer.writerows(rows)
        self.config["assets"][0]["csv_sha256"] = benchmark.sha256(history.read_bytes())
        self.save()
        changed = benchmark.run_benchmark(self.manifest, self.engine, "validation")
        self.assertEqual(original["results"], changed["results"])

    def test_symbol_mismatch_and_empty_splits_fail_loudly(self):
        self.config["assets"][0]["symbol"] = "WRONG"
        self.save()
        with self.assertRaisesRegex(ValueError, "symbol"):
            benchmark.run_benchmark(self.manifest, self.engine, "validation")
        self.config["assets"][0]["symbol"] = "SYNTH_A"
        self.config["validation_start"] = self.dates[99]
        self.save()
        with self.assertRaisesRegex(ValueError, "no eligible forecasts"):
            benchmark.run_benchmark(self.manifest, self.engine, "validation")


if __name__ == "__main__":
    unittest.main()
