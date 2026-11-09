import copy
import csv
from datetime import date
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "energy"))
import baselines
import boosted_demand as energy
import ieso
import uncertainty as ranges
from test_energy_baselines import report_bytes


NATIVE = all(importlib.util.find_spec(name) is not None for name in ("numpy", "lightgbm"))


class UncertaintyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "load.csv"
        self.path.write_bytes(report_bytes(date(2020, 1, 1), 110))

    def point(self, actual=10, low=8, high=12):
        return {**ieso.calendar_fields(ieso.market_midnight(date(2020, 2, 1))), "actual_mw": actual,
                "ranges": {"raw": [low, high], "calibrated": [max(0, low - 2), high + 2]}}

    def test_finite_sample_rank_and_available_calibration_labels(self):
        points = [self.point(actual=index, low=0, high=0) for index in range(10)]
        cutoff = energy.origin_at(date(2020, 3, 1))
        fitted = ranges.correction(points, cutoff)
        self.assertEqual(fitted["rank_one_based"], 10)
        self.assertEqual(fitted["widening_mw"], 9)
        self.assertLessEqual(fitted["latest_label_assumed_available_at_utc"], fitted["frozen_at_utc"])
        with self.assertRaisesRegex(ValueError, "too few"):
            ranges.correction(points[:1], cutoff)
        with self.assertRaisesRegex(ValueError, "not available"):
            ranges.correction(points, ieso.market_midnight(date(2020, 2, 2)))
        self.assertEqual(ranges.correction([self.point()] * 10, cutoff)["widening_mw"], 0)

    def test_metrics_and_invalid_bounds(self):
        point = self.point()
        metrics = ranges.range_metrics([point], "raw")
        self.assertEqual(metrics["coverage"], 1)
        self.assertEqual(metrics["mean_width_mw"], 4)
        self.assertEqual(metrics["mean_interval_score_mw"], 4)
        self.assertAlmostEqual(metrics["mean_pinball_mw"], 0.1)
        for low, high in ((12, 8), (-1, 12), (0, float("inf"))):
            with self.assertRaisesRegex(ValueError, "invalid"):
                ranges.range_metrics([self.point(low=low, high=high)], "raw")
        with self.assertRaises(ValueError):
            ranges.range_metrics([], "raw")

    def test_whole_calibration_windows_are_purged_at_boundary(self):
        observations, _ = ieso.load_reports([self.path])
        result = baselines.backtest(observations, date(2020, 3, 1), date(2020, 4, 8))
        calibration, evaluation, purged = ranges.partition(result["windows"], date(2020, 4, 5))
        self.assertEqual(purged, ["2020-04-04"])
        self.assertEqual(len(calibration), 34)
        self.assertEqual(len(evaluation), 3)
        boundary = ieso.market_midnight(date(2020, 4, 5))
        self.assertTrue(all(ieso.parse_origin(p["end_utc"]) <= boundary for w in calibration for p in w["points"]))

    def test_hour_and_season_groups_preserve_repeated_local_hour(self):
        points = [self.point() for _ in range(5)]
        for point, local in zip(points, ("2020-11-01T01:00:00-04:00", "2020-11-01T01:00:00-05:00",
                                        "2020-03-01T03:00:00-05:00", "2020-06-01T03:00:00-04:00",
                                        "2020-12-01T03:00:00-05:00")):
            point["local_start"] = local
        groups = ranges.breakdown(points)
        self.assertEqual(set(groups["season"]), {"winter", "spring", "summer", "autumn"})
        self.assertEqual(groups["local_hour"]["01"]["raw"]["rows"], 2)
        self.assertEqual(sum(group["raw"]["rows"] for group in groups["season"].values()), 5)

    def test_invalid_periods_and_quantiles_fail_before_fit(self):
        with self.assertRaisesRegex(ValueError, "chronological"):
            ranges.evaluate([self.path], date(2018, 1, 15), date(2022, 1, 1), date(2022, 1, 1),
                            date(2023, 1, 1), date(2024, 1, 1))
        for quantile in (0, 1, float("nan"), True):
            with self.assertRaisesRegex(ValueError, "quantile"):
                energy.fit_model([], [], quantile=quantile)
        command = [sys.executable, str(ROOT / "apps/energy/uncertainty.py"), "--csv", str(self.path),
                   "--output", str(self.path.with_suffix(".json"))]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--retrospective", result.stderr)
        self.assertFalse(self.path.with_suffix(".json").exists())

    @unittest.skipUnless(NATIVE, "optional energy requirements not installed")
    def test_native_frozen_calibration_point_preservation_and_future_mutation(self):
        args = ([self.path], date(2020, 1, 15), date(2020, 3, 1), date(2020, 3, 1),
                date(2020, 4, 5), date(2020, 4, 8))
        parameters = copy.deepcopy(energy.PARAMETERS)
        report = ranges.evaluate(*args)
        original = energy.evaluate([self.path], args[1], args[2], args[4], args[5])
        self.assertEqual(report["model_sha256"], original["model_sha256"])
        self.assertEqual(report["windows"][0]["points"][0]["models"], original["windows"][0]["points"][0]["models"])
        self.assertEqual(report["point_metrics"], original["metrics"])
        self.assertEqual(report["evaluation"]["rows"], 72)
        self.assertEqual(report["calibration"]["purged_origin_dates"], ["2020-04-04"])
        self.assertEqual(energy.PARAMETERS, parameters)
        self.assertGreaterEqual(report["range_metrics"]["calibrated"]["coverage"], report["range_metrics"]["raw"]["coverage"])
        for window in report["windows"]:
            for point in window["points"]:
                low, high = point["ranges"]["calibrated"]
                self.assertTrue(0 <= low <= high and math.isfinite(high))
        for point in report["calibration_windows"][0]["points"]:
            self.assertNotIn("calibrated", point["ranges"])
        ieso.write_json(self.path.with_suffix(".json"), report)
        self.assertEqual(json.loads(self.path.with_suffix(".json").read_text())["status"], "ok")
        rows = list(csv.reader(io.StringIO(self.path.read_text())))
        for row in rows[2:]:
            if row[0] >= "2020-04-05":
                row[2:] = [str(float(value) * 2) for value in row[2:]]
        output = io.StringIO()
        csv.writer(output).writerows(rows)
        self.path.write_text(output.getvalue())
        changed = ranges.evaluate(*args)
        self.assertEqual(report["calibration"], changed["calibration"])
        self.assertEqual(report["quantile_models"], changed["quantile_models"])


if __name__ == "__main__":
    unittest.main()
