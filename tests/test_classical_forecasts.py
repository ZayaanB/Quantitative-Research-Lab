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
import classical_forecasts as classical

HAS_MODELS = importlib.util.find_spec("statsmodels") is not None


class ClassicalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.dates = weekday_dates(100)
        self.config = {"schema_version": 1, "lookback_days": 30, "validation_start": self.dates[50],
                       "holdout_start": self.dates[75], "test_end": self.dates[-1], "assets": []}
        for symbol in ("FLAT", "WAVE"):
            prices = (100 if symbol == "FLAT" else
                      100 * math.exp(0.0002 * i + 0.003 * math.sin(i)) for i in range(len(self.dates)))
            self.config["assets"].append(write_price_asset(self.directory, symbol, self.dates, prices))
        self.manifest = self.directory / "manifest.json"
        self.manifest.write_text(json.dumps(self.config))

    def test_input_validation_and_exact_constant_window(self):
        for model in classical.MODELS:
            self.assertEqual(classical.predict([100] * 31, 5, model)["price"], 100)
        for prices, horizon, model in (([100] * 30, 1, "arima_110"), ([100] * 31, 0, "ets_level"),
                                      ([100] * 31, 1, "unknown"), ([100] * 30 + [0], 1, "ets_level")):
            with self.assertRaises(ValueError):
                classical.predict(prices, horizon, model)

    @unittest.skipUnless(HAS_MODELS, "optional statsmodels is not installed")
    def test_actual_models_and_independent_arima_reference(self):
        from statsmodels.tsa.arima.model import ARIMA
        prices = [100 * math.exp(0.001 * i + 0.003 * math.sin(i)) for i in range(61)]
        for model in classical.MODELS:
            first = classical.predict(prices, 5, model)
            self.assertTrue(math.isfinite(first["price"]) and first["price"] > 0)
            self.assertEqual(first, classical.predict(prices, 5, model))
        offset = math.log(prices[-1])
        fitted = ARIMA([math.log(price) - offset for price in prices], order=(1, 1, 0),
                       trend="n").fit(method_kwargs={"maxiter": 200})
        reference = math.exp(offset + float(fitted.forecast(5)[-1]))
        self.assertAlmostEqual(classical.predict(prices, 5, "arima_110")["price"], reference)

    @unittest.skipUnless(HAS_MODELS, "optional statsmodels is not installed")
    def test_nonconverged_fit_is_rejected(self):
        fake = type("Fit", (), {"mle_retvals": {"converged": False}})()
        with patch("statsmodels.tsa.arima.model.ARIMA.fit", return_value=fake):
            with self.assertRaisesRegex(ValueError, "converge"):
                classical.predict([100 + i for i in range(31)], 1, "arima_110")

    @unittest.skipUnless(HAS_MODELS, "optional statsmodels is not installed")
    def test_paired_dates_failures_and_no_future_training(self):
        seen = []

        def predictor(prices, horizon, model):
            seen.append(list(prices))
            if model == "ets_damped":
                raise ValueError("simulated nonconvergence")
            return {"price": prices[-1], "warnings": []}

        with patch.object(classical, "version", return_value="test"), patch.object(classical, "predict", side_effect=predictor):
            report = classical.evaluate(self.manifest)
        self.assertEqual(report["selected"], "latest_price")
        self.assertIsNone(report["ranking"][-1]["mean_relative_mae"])
        self.assertEqual(len(report["results"]), 6)
        expected = []
        for group in report["results"]:
            self.assertIsNone(group["metrics"]["ets_damped"])
            self.assertEqual(len(group["fit_failures"]), len(group["points"]))
            for point in group["points"]:
                self.assertLess(point["target_date"], self.config["holdout_start"])
                origin = self.dates.index(point["origin_date"])
                for model in classical.MODELS:
                    expected.append((origin, group["symbol"]))
        self.assertEqual(len(expected), len(seen))
        for (origin, symbol), window in zip(expected, seen):
            self.assertEqual(len(window), 31)
            wanted = 100 if symbol == "FLAT" else 100 * math.exp(0.0002 * origin + 0.003 * math.sin(origin))
            self.assertEqual(window[-1], wanted)
        asset = self.config["assets"][1]
        history = self.directory / asset["csv"]
        with history.open() as stream:
            rows = list(csv.reader(stream))
        for index, row in enumerate(rows[1:]):
            if index >= 75:
                row[1] = str(float(row[1]) * 2)
        with history.open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)
        asset["csv_sha256"] = classical.sha256(history.read_bytes())
        self.manifest.write_text(json.dumps(self.config))
        with patch.object(classical, "version", return_value="test"), patch.object(classical, "predict", side_effect=predictor):
            self.assertEqual(report["results"], classical.evaluate(self.manifest)["results"])

    @unittest.skipUnless(HAS_MODELS, "optional statsmodels is not installed")
    def test_cli_full_run_and_output_protection(self):
        output = self.directory / "classical.json"
        command = [sys.executable, str(ROOT / "python/classical_forecasts.py"),
                   "--manifest", str(self.manifest), "--output", str(output)]
        process = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        report = json.loads(output.read_text())
        self.assertEqual(report["phase"], "validation")
        self.assertEqual(len(report["results"]), 6)
        before = output.read_bytes()
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.assertEqual(before, output.read_bytes())


if __name__ == "__main__":
    unittest.main()
