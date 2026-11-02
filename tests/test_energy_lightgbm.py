from datetime import date, timedelta
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "energy"))
import baselines
import boosted_demand as energy
import ieso
from test_energy_baselines import report_bytes


NATIVE = all(importlib.util.find_spec(name) is not None for name in ("numpy", "lightgbm"))


class EnergyModelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.path = self.directory / "synthetic.csv"
        self.path.write_bytes(report_bytes(date(2020, 1, 1), 110))
        self.observations, _ = ieso.load_reports([self.path])
        self.start, self.cutoff = date(2020, 1, 15), date(2020, 3, 1)

    def test_feature_values_sources_and_future_mutation(self):
        origin = energy.origin_at(date(2020, 3, 1))
        rows, evidence = energy.feature_rows(self.observations, origin)
        self.assertEqual((len(rows), len(rows[0])), (24, len(energy.FEATURES)))
        lag = baselines.seasonal_value(self.observations, origin, origin, 24)
        self.assertEqual(rows[0][energy.FEATURES.index("daily_seasonal_mw")], lag["prediction_mw"])
        changed = {start: ieso.Observation(start, 999999) if ieso.assumed_available_at(start) > origin else row
                   for start, row in self.observations.items()}
        self.assertEqual((rows, evidence), energy.feature_rows(changed, origin))
        self.assertTrue(all(point["max_feature_available_at_utc"] <= ieso.iso(origin) for point in evidence))
        for point in evidence:
            self.assertTrue(all(ieso.assumed_available_at(ieso.parse_origin(start)) <= origin
                                for start in point["lag_sources_utc"].values()))
        with self.assertRaises(ValueError):
            energy.feature_rows(self.observations, origin + ieso.HOUR)

    def test_dst_calendar_and_missing_available_features(self):
        rows, _ = energy.feature_rows(self.observations, energy.origin_at(date(2020, 3, 7)))
        offsets = [row[energy.FEATURES.index("local_utc_offset_hours")] for row in rows]
        self.assertEqual(set(offsets), {-5, -4})
        origin = energy.origin_at(date(2020, 3, 1))
        changed = dict(self.observations)
        del changed[origin - 24 * ieso.HOUR]
        with self.assertRaisesRegex(ValueError, "missing|unavailable"):
            energy.feature_rows(changed, origin)

    def test_training_purges_whole_boundary_window_and_ignores_validation_labels(self):
        x, y, details = energy.training_data(self.observations, self.start, self.cutoff)
        self.assertEqual(details["purged_origin_dates"], ["2020-02-29"])
        self.assertEqual(details["rows"], details["origins"] * 24)
        self.assertLessEqual(details["last_target_end_utc"], details["boundary_utc"])
        self.assertLessEqual(details["last_label_assumed_available_at_utc"], details["fit_origin_utc"])
        self.assertEqual(len(x), len(y))
        boundary = ieso.market_midnight(self.cutoff)
        changed = {start: ieso.Observation(start, 1e6) if start >= boundary else row
                   for start, row in self.observations.items()}
        self.assertEqual((x, y, details), energy.training_data(changed, self.start, self.cutoff))

    def test_reject_missing_training_labels_short_ranges_and_final_period(self):
        changed = dict(self.observations)
        del changed[energy.origin_at(date(2020, 2, 1))]
        with self.assertRaisesRegex(ValueError, "missing training target"):
            energy.training_data(changed, self.start, self.cutoff)
        with self.assertRaisesRegex(ValueError, "at least"):
            energy.training_data(self.observations, self.start, self.start + timedelta(days=5))
        for start, end in ((date(2020, 2, 28), date(2020, 3, 2)), (date(2024, 1, 1), date(2024, 2, 1))):
            with self.assertRaisesRegex(ValueError, "require training"):
                energy.evaluate([self.path], self.start, self.cutoff, start, end)

    @unittest.skipUnless(NATIVE, "optional energy LightGBM requirements not installed")
    def test_native_matched_scores_provenance_and_training_only_fit(self):
        report = energy.evaluate([self.path], self.start, self.cutoff, self.cutoff, date(2020, 3, 4))
        baseline = baselines.backtest(self.observations, self.cutoff, date(2020, 3, 4))
        self.assertEqual(report["counts"], {"origins": 3, "hourly_targets": 72})
        for model in baselines.PERIODS:
            self.assertEqual(report["metrics"][model], baseline["metrics"][model])
        self.assertEqual(report["model_sha256"], ieso.digest(report["model_text"].encode()))
        self.assertEqual(report["dependencies"]["lightgbm"], energy.version("lightgbm"))
        points = [point for window in report["windows"] for point in window["points"]]
        self.assertEqual(len({point["start_utc"] for point in points}), 72)
        for window in report["windows"]:
            for point in window["points"]:
                prediction = point["models"]["lightgbm"]
                self.assertTrue(math.isfinite(prediction["prediction_mw"]))
                self.assertLessEqual(prediction["max_feature_available_at_utc"], window["origin_utc"])
        changed = dict(self.observations)
        boundary = ieso.market_midnight(self.cutoff)
        changed.update({start: ieso.Observation(start, row.demand_mw * 2) for start, row in changed.items() if start >= boundary})
        x, y, _ = energy.training_data(changed, self.start, self.cutoff)
        model = energy.fit_model(x, y)
        self.assertEqual(ieso.digest(model.model_to_string().encode()), report["model_sha256"])

    @unittest.skipUnless(NATIVE, "optional energy LightGBM requirements not installed")
    def test_native_invalid_outputs_fail_without_fallback(self):
        rows = [[1] * len(energy.FEATURES)]
        for values in ([float("nan")], [-1], [1, 2]):
            with self.assertRaisesRegex(ValueError, "invalid|negative"):
                energy.predict(Mock(predict=Mock(return_value=values)), rows)
        with self.assertRaisesRegex(ValueError, "invalid native"):
            energy.fit_model([[1]], [1])

    @unittest.skipUnless(NATIVE, "optional energy LightGBM requirements not installed")
    def test_native_cli_and_no_overwrite(self):
        output = self.directory / "report.json"
        command = [sys.executable, str(ROOT / "apps/energy/boosted_demand.py"), "--csv", str(self.path),
                   "--train-start", self.start.isoformat(), "--train-end", self.cutoff.isoformat(),
                   "--start", self.cutoff.isoformat(), "--end", "2020-03-04", "--output", str(output)]
        failed = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("--retrospective", failed.stderr)
        self.assertFalse(output.exists())
        completed = subprocess.run(command + ["--retrospective"], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        original = output.read_bytes()
        self.assertEqual(json.loads(original)["phase"], "validation")
        failed = subprocess.run(command + ["--retrospective"], capture_output=True, text=True)
        self.assertNotEqual(failed.returncode, 0)
        self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
