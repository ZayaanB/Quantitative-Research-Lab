from datetime import date
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "energy"))

from boosted_demand import origin_at
from ieso import HOUR, iso
from recent_benchmark import blocks, operator_values, paired_interval


class RecentBenchmarkTests(unittest.TestCase):
    def test_operator_rejects_late_report_and_only_scores_complete_windows(self):
        origin = origin_at(date(2026, 10, 8))
        next_origin = origin + 24 * HOUR
        archive = {"2026-10-08": [
            {"issued": iso(origin - HOUR), "last_modified": iso(origin),
             "hours": {iso(origin + step * HOUR): 10_000 + step for step in range(24)}}],
            "2026-10-09": [{"issued": iso(next_origin), "last_modified": iso(next_origin),
                            "hours": {iso(next_origin): 9_500}}]}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "archive.json"
            path.write_text(json.dumps(archive))
            values, mask, _ = operator_values(path, [origin, next_origin])
            self.assertEqual(mask.tolist(), [True, False])
            self.assertEqual(values[0].tolist(), list(range(10_000, 10_024)))
            archive["2026-10-09"][0]["last_modified"] = iso(next_origin + HOUR)
            path.write_text(json.dumps(archive))
            with self.assertRaisesRegex(ValueError, "ineligible"):
                operator_values(path, [origin, next_origin])

    def test_weather_gaps_require_declaration(self):
        start, end = date(2026, 10, 8), date(2026, 10, 9)
        with self.assertRaisesRegex(ValueError, "document the gap"):
            blocks({}, start, end, {}, purge=False, known_gaps=set())
        with self.assertRaisesRegex(ValueError, "no complete origins"):
            blocks({}, start, end, {}, purge=False, known_gaps={start})

    def test_block_interval_sign_and_minimum(self):
        actual = np.full((80, 24), 1_000.0)
        control, better = actual + 100, actual - 25
        self.assertIsNone(paired_interval(actual[:44], control[:44], better[:44]))
        interval = paired_interval(actual, control, better)
        self.assertEqual(interval["hourly_reduction_mw"], [75, 75])
        self.assertEqual(interval["peak_reduction_mw"], [75, 75])


if __name__ == "__main__":
    unittest.main()
