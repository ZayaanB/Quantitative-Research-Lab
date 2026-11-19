import csv
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
import residual_forecasts as residual
import final_forecast


class ResidualTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.dates = weekday_dates(700)
        self.config = {"schema_version": 1, "lookback_days": 60,
                       "validation_start": self.dates[500], "holdout_start": self.dates[660],
                       "test_end": self.dates[-1], "assets": []}
        for symbol, prices in (("FLAT", [100.0] * 700),
                               ("WAVE", [100 * math.exp(0.0001 * i + 0.003 * math.sin(i)) for i in range(700)])):
            self.config["assets"].append(write_price_asset(self.directory, symbol, self.dates, prices))
        self.manifest = self.directory / "manifest.json"
        self.save()

    def save(self):
        self.manifest.write_text(json.dumps(self.config), encoding="utf-8")

    def test_frozen_final_workflow_and_holdout_boundaries(self):
        validation = self.directory / "validation.json"
        final_forecast.save_new(validation, residual.evaluate(self.manifest))
        procedure = final_forecast.freeze(self.manifest, validation, "ensemble")
        procedure_path = self.directory / "procedure.json"
        final_forecast.save_new(procedure_path, procedure)
        output = self.directory / "holdout.json"
        marker = self.directory / "procedure.json.used"
        with self.assertRaisesRegex(ValueError, "confirm"):
            final_forecast.run_final(procedure_path, self.manifest, output)
        self.assertFalse(marker.exists())
        with self.assertRaisesRegex(ValueError, "distinct"):
            final_forecast.run_final(procedure_path, self.manifest, marker, True)
        self.assertFalse(marker.exists())
        report = final_forecast.run_final(procedure_path, self.manifest, output, True)
        self.assertEqual(report, json.loads(output.read_text()))
        self.assertTrue(marker.exists())
        self.assertEqual(report["phase"], "holdout")
        self.assertEqual(report["primary_model"], "ensemble")
        ratios = []
        for group in report["results"]:
            for point in group["points"]:
                self.assertGreaterEqual(point["origin_date"], self.config["holdout_start"])
                self.assertLessEqual(point["target_date"], self.config["test_end"])
                self.assertLessEqual(point["last_calibration_target"], point["origin_date"])
            baseline = group["metrics"]["latest_price"]["point"]["mae"]
            if baseline:
                ratios.append(group["metrics"]["ensemble"]["point"]["mae"] / baseline)
        self.assertAlmostEqual(report["primary_mean_relative_mae"], sum(ratios) / len(ratios))
        with self.assertRaisesRegex(ValueError, "already recorded"):
            final_forecast.run_final(procedure_path, self.manifest, self.directory / "second.json", True)
        with self.assertRaises(FileExistsError):
            final_forecast.save_new(procedure_path, procedure)

    def test_final_rejects_stale_reports_sources_and_inputs(self):
        validation = self.directory / "validation.json"
        report = residual.evaluate(self.manifest)
        final_forecast.save_new(validation, report)
        procedure = final_forecast.freeze(self.manifest, validation, "latest_price")
        incomplete = {**report, "results": report["results"][:-1]}
        validation.write_text(json.dumps(incomplete))
        with self.assertRaisesRegex(ValueError, "missing"):
            final_forecast.freeze(self.manifest, validation, "ensemble")
        duplicate = {**report, "results": report["results"] + [report["results"][0]]}
        validation.write_text(json.dumps(duplicate))
        with self.assertRaisesRegex(ValueError, "distinct"):
            final_forecast.freeze(self.manifest, validation, "ensemble")
        duplicate_points = json.loads(json.dumps(report))
        group = duplicate_points["results"][0]
        group["points"].insert(1, group["points"][0])
        group["forecasts"] += 1
        validation.write_text(json.dumps(duplicate_points))
        with self.assertRaisesRegex(ValueError, "distinct"):
            final_forecast.freeze(self.manifest, validation, "ensemble")
        validation.write_text(json.dumps(report))
        changed = {**procedure, "source_hashes": {}}
        with self.assertRaisesRegex(ValueError, "source"):
            final_forecast.verify(changed, self.manifest)
        with self.assertRaisesRegex(ValueError, "Python"):
            final_forecast.verify({**procedure, "python_version": "0.0"}, self.manifest)
        with self.assertRaisesRegex(ValueError, "platform"):
            final_forecast.verify({**procedure, "platform": "other"}, self.manifest)
        report["phase"] = "holdout"
        validation.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "validation report"):
            final_forecast.freeze(self.manifest, validation, "ensemble")
        self.config["lookback_days"] = 61
        self.save()
        with self.assertRaisesRegex(ValueError, "manifest"):
            final_forecast.verify(procedure, self.manifest)
        self.config["lookback_days"] = 60
        self.save()
        history = self.directory / self.config["assets"][0]["csv"]
        history.write_bytes(history.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            final_forecast.verify(procedure, self.manifest)

    def test_final_cli_freezes_without_scoring_and_requires_confirmation(self):
        validation = self.directory / "validation.json"
        final_forecast.save_new(validation, residual.evaluate(self.manifest))
        procedure_path = self.directory / "procedure.json"
        freeze_command = [sys.executable, str(ROOT / "python/final_forecast.py"), "freeze",
                          "--manifest", str(self.manifest), "--validation", str(validation),
                          "--model", "latest_price", "--output", str(procedure_path)]
        process = subprocess.run(freeze_command, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertFalse(procedure_path.with_name("procedure.json.used").exists())
        run_command = [sys.executable, str(ROOT / "python/final_forecast.py"), "run",
                       "--manifest", str(self.manifest), "--procedure", str(procedure_path),
                       "--output", str(self.directory / "final.json")]
        self.assertNotEqual(subprocess.run(run_command, capture_output=True).returncode, 0)
        process = subprocess.run(run_command + ["--confirm-untouched-holdout"], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        report = json.loads((self.directory / "final.json").read_text())
        self.assertEqual(report["primary_model"], "latest_price")
        self.assertEqual(report["primary_mean_relative_mae"], 1)

    def test_failed_final_attempt_preserves_marker(self):
        validation = self.directory / "validation.json"
        final_forecast.save_new(validation, residual.evaluate(self.manifest))
        procedure_path = self.directory / "procedure.json"
        final_forecast.save_new(procedure_path, final_forecast.freeze(self.manifest, validation, "ensemble"))
        output = self.directory / "holdout.json"
        with patch.object(final_forecast, "evaluate", side_effect=ValueError("simulated failure")):
            with self.assertRaisesRegex(ValueError, "simulated"):
                final_forecast.run_final(procedure_path, self.manifest, output, True)
        self.assertTrue(self.directory.joinpath("procedure.json.used").exists())
        self.assertFalse(output.exists())
        with self.assertRaisesRegex(ValueError, "already recorded"):
            final_forecast.run_final(procedure_path, self.manifest, output, True)

    def test_empirical_quantiles_probabilities_and_limits(self):
        distribution = residual.empirical_distribution(100, 100, [-0.1, 0, 0.1, 0.2])
        self.assertEqual(distribution["probability_above_current"], 0.5)
        self.assertAlmostEqual(distribution["lower_95"], 100 * math.exp(-0.0925))
        self.assertAlmostEqual(distribution["upper_95"], 100 * math.exp(0.1925))
        self.assertLessEqual(distribution["lower_95"], distribution["lower_90"])
        self.assertLessEqual(distribution["upper_90"], distribution["upper_95"])
        flat = residual.empirical_distribution(100, 100, [0, 0])
        self.assertEqual(flat["probability_above_current"], 0)
        self.assertEqual(flat["lower_95"], 100)
        for price, values in ((0, [0, 0]), (100, [0]), (100, [0, float("nan")]),
                              (100, [1000, 1000]), (100, [-1000, -1000])):
            with self.assertRaises(ValueError):
                residual.empirical_distribution(price, 100, values)

    def test_interval_score_penalizes_misses(self):
        points = [{"actual_price": 120, "current_price": 100, "distributions": {
            "ensemble": {"probability_above_current": 0.75,
                         **{f"{side}_{level}": value for level in (80, 90, 95)
                            for side, value in (("lower", 90), ("upper", 110))}}}}]
        metrics = residual.distribution_metrics(points, "ensemble")
        self.assertEqual(metrics["brier_score"], 0.0625)
        self.assertEqual(metrics["coverage_95"], 0)
        self.assertEqual(metrics["mean_width_95"], 20)
        self.assertAlmostEqual(metrics["interval_score_95"], 420)
        points[0]["actual_price"] = 80
        self.assertAlmostEqual(residual.distribution_metrics(points, "ensemble")["interval_score_95"], 420)
        points[0]["actual_price"] = 100
        self.assertEqual(residual.distribution_metrics(points, "ensemble")["interval_score_95"], 20)

    def test_paired_evaluation_and_calibration_boundaries(self):
        report = residual.evaluate(self.manifest)
        self.assertEqual(report, residual.evaluate(self.manifest))
        self.assertEqual(len(report["results"]), 6)
        self.assertEqual(report["phase"], "validation")
        for group in report["results"]:
            self.assertTrue(group["points"])
            for point in group["points"]:
                self.assertLessEqual(point["last_calibration_target"], point["origin_date"])
                self.assertLessEqual(point["first_calibration_target"], point["last_calibration_target"])
                self.assertTrue(20 <= point["calibration_count"] <= 60)
                origin = self.dates.index(point["origin_date"])
                self.assertEqual(point["last_calibration_target"], point["origin_date"])
                self.assertEqual(point["first_calibration_target"],
                                 self.dates[origin - (point["calibration_count"] - 1) * group["horizon_days"]])
                self.assertLess(point["target_date"], self.config["holdout_start"])
                self.assertGreaterEqual(point["origin_date"], self.config["validation_start"])
                self.assertEqual(set(point["distributions"]), {"ensemble", "latest_price"})
            if group["horizon_days"] == 1:
                self.assertTrue(all(p["calibration_count"] == 60 for p in group["points"]))
            if group["symbol"] == "FLAT":
                for name in ("ensemble", "latest_price"):
                    metrics = group["metrics"][name]
                    self.assertEqual(metrics["point"]["mae"], 0)
                    self.assertEqual(metrics["distribution"]["coverage_95"], 1)
                    self.assertEqual(metrics["distribution"]["mean_width_95"], 0)
                    self.assertEqual(metrics["distribution"]["brier_score"], 0)

    def test_future_mutations_do_not_change_earlier_ranges(self):
        original = residual.evaluate(self.manifest)
        asset = self.config["assets"][1]
        history = self.directory / asset["csv"]
        with history.open() as stream:
            rows = list(csv.reader(stream))
        for index, row in enumerate(rows[1:]):
            if index >= 660:
                row[1] = str(float(row[1]) * 2)
        with history.open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)
        asset["csv_sha256"] = residual.sha256(history.read_bytes())
        self.save()
        self.assertEqual(original["results"], residual.evaluate(self.manifest)["results"])
        for index, row in enumerate(rows[1:]):
            if index >= 550:
                row[1] = str(float(row[1]) * 2)
        with history.open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)
        asset["csv_sha256"] = residual.sha256(history.read_bytes())
        self.save()
        changed = residual.evaluate(self.manifest)
        for before, after in zip(original["results"], changed["results"]):
            for first, second in zip(before["points"], after["points"]):
                if first["origin_date"] < self.dates[550]:
                    self.assertEqual(first["forecast_price"], second["forecast_price"])
                    self.assertEqual(first["distributions"], second["distributions"])

    def test_cli_output_protection_and_insufficient_history(self):
        output = self.directory / "ranges.json"
        command = [sys.executable, str(ROOT / "python/residual_forecasts.py"),
                   "--manifest", str(self.manifest), "--output", str(output)]
        process = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        report = json.loads(output.read_text())
        self.assertEqual(report["phase"], "validation")
        prior = output.read_bytes()
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.assertEqual(prior, output.read_bytes())
        self.config["holdout_start"] = self.dates[300]
        self.config["validation_start"] = self.dates[100]
        self.save()
        with self.assertRaisesRegex(ValueError, "warmup"):
            residual.evaluate(self.manifest)
        with self.assertRaises(ValueError):
            residual.rolling_points([], [], 0, 60, self.dates[0])


if __name__ == "__main__":
    unittest.main()
