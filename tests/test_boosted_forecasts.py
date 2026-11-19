import csv
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


from stock_fixtures import weekday_dates, write_price_asset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import boosted_forecasts as boosted

HAS_MODELS = importlib.util.find_spec("lightgbm") is not None


class BoostedTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.dates = weekday_dates(180)
        self.config = {"schema_version": 1, "lookback_days": 60, "validation_start": self.dates[100],
                       "holdout_start": self.dates[160], "test_end": self.dates[-1],
                       "bootstrap": {"samples": 100, "seed": 42, "block_sizes": [1, 2, 4]}, "assets": []}
        for symbol in ("FLAT", "WAVE"):
            prices = (100 if symbol == "FLAT" else
                      100 * math.exp(0.0002 * i + 0.003 * math.sin(i)) for i in range(len(self.dates)))
            self.config["assets"].append(write_price_asset(self.directory, symbol, self.dates, prices))
        self.manifest = self.directory / "manifest.json"
        self.manifest.write_text(json.dumps(self.config))

    def test_features_and_matured_label_boundary(self):
        prices = [100 * math.exp(0.001 * i + 0.002 * math.sin(i)) for i in range(100)]
        features, targets, current = boosted.training_set(prices, 5, 60)
        self.assertEqual(len(features), 56)
        self.assertEqual(len(current), len(boosted.FEATURES))
        self.assertAlmostEqual(features[0][0], math.log(prices[39]) - math.log(prices[38]))
        self.assertAlmostEqual(targets[-1], math.log(prices[99]) - math.log(prices[94]))
        self.assertAlmostEqual(current[4], math.log(prices[99]) - math.log(prices[79]))
        for values, horizon, window in ((prices, 0, 60), (prices, 1, 59),
                                        ([100] * 40, 20, 60), ([100] * 99 + [0], 1, 60)):
            with self.assertRaises(ValueError):
                boosted.training_set(values, horizon, window)

    @unittest.skipUnless(HAS_MODELS, "optional LightGBM is not installed")
    def test_native_fit_is_deterministic_and_handles_flat_history(self):
        flat = boosted.forecast([100] * 120, 20, 60)
        self.assertEqual(flat["forecast_price"], 100)
        self.assertEqual(flat["predicted_log_return"], 0)
        self.assertEqual(flat["last_training_target_index"], 119)
        self.assertEqual(flat["training_rows"], 41)
        prices = [100 * math.exp(0.001 * i) for i in range(120)]
        first = boosted.forecast(prices, 5, 60)
        self.assertEqual(first, boosted.forecast(prices, 5, 60))
        self.assertAlmostEqual(first["forecast_price"], prices[-1] * math.exp(0.005), places=5)
        self.assertEqual(set(first["feature_gain"]), set(boosted.FEATURES))

    @unittest.skipUnless(HAS_MODELS, "optional LightGBM is not installed")
    def test_native_fit_failure_is_reported(self):
        import lightgbm
        with patch.object(lightgbm, "train", side_effect=lightgbm.basic.LightGBMError("simulated")):
            with self.assertRaisesRegex(ValueError, "simulated"):
                boosted.forecast([100] * 120, 1, 60)

    @unittest.skipUnless(HAS_MODELS, "optional LightGBM is not installed")
    def test_validation_ignores_future_data_and_pairs_baseline(self):
        original = boosted.evaluate(self.manifest)
        self.assertEqual(len(original["results"]), 6)
        self.assertEqual(original["results"], boosted.evaluate(self.manifest)["results"])
        for group in original["results"]:
            for point in group["points"]:
                self.assertLess(point["target_date"], self.config["holdout_start"])
                self.assertEqual(point["current_price"], point["latest_price_forecast"])
            if group["symbol"] == "FLAT":
                self.assertEqual(group["metrics"]["lightgbm"]["mae"], 0)
        asset = self.config["assets"][1]
        history = self.directory / asset["csv"]
        with history.open() as stream:
            rows = list(csv.reader(stream))

        def mutate(start):
            for index, row in enumerate(rows[1:]):
                if index >= start:
                    row[1] = str(float(row[1]) * 2)
            with history.open("w", newline="") as stream:
                csv.writer(stream).writerows(rows)
            asset["csv_sha256"] = boosted.sha256(history.read_bytes())
            self.manifest.write_text(json.dumps(self.config))

        mutate(160)
        self.assertEqual(original["results"], boosted.evaluate(self.manifest)["results"])
        mutate(130)
        changed = boosted.evaluate(self.manifest)
        for before, after in zip(original["results"], changed["results"]):
            for first, second in zip(before["points"], after["points"]):
                if first["origin_date"] < self.dates[130]:
                    self.assertEqual(first["forecast_price"], second["forecast_price"])
                    self.assertEqual(first["feature_gain"], second["feature_gain"])

    @unittest.skipUnless(HAS_MODELS, "optional LightGBM is not installed")
    def test_cli_output_protection(self):
        output = self.directory / "boosted.json"
        command = [sys.executable, str(ROOT / "python/boosted_forecasts.py"),
                   "--manifest", str(self.manifest), "--output", str(output)]
        process = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        report = json.loads(output.read_text())
        self.assertEqual(report["phase"], "validation")
        self.assertIn(report["selected"], {"latest_price", "lightgbm"})
        before = output.read_bytes()
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.assertEqual(before, output.read_bytes())


if __name__ == "__main__":
    unittest.main()
