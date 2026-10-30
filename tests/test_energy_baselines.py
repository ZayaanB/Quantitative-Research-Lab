import csv
from datetime import date, timedelta
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.request import Request


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "energy"))
import baselines
import ieso


def report_bytes(start=date(2023, 1, 1), days=30, constant=None):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["\\\\Hourly Demand Report", "", "", ""])
    writer.writerow(ieso.HEADER)
    for index in range(days):
        day = start + timedelta(days=index)
        for hour in range(1, 25):
            load = constant if constant is not None else 10000 + index * 10 + hour * 50 + day.weekday() * 100
            writer.writerow([day.isoformat(), hour, load + 1000, load])
    return stream.getvalue().encode()


class EnergyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.path = self.directory / "load.csv"
        self.payload = report_bytes()
        self.path.write_bytes(self.payload)
        self.origin = ieso.parse_origin("2023-01-10T08:00:00-05:00")
        self.observations, _ = ieso.load_reports([self.path])

    def test_hour_ending_units_and_target_column(self):
        observations, preamble = ieso.parse_report(self.payload)
        self.assertIn("Hourly Demand Report", preamble[0])
        self.assertEqual(ieso.iso(observations[0].start), "2023-01-01T05:00:00+00:00")
        self.assertEqual(ieso.iso(observations[23].start), "2023-01-02T04:00:00+00:00")
        self.assertEqual(ieso.iso(observations[23].start + ieso.HOUR), "2023-01-02T05:00:00+00:00")
        self.assertEqual(observations[0].demand_mw, 10650)
        self.assertEqual(len(observations), 30 * 24)

    def test_reject_unknown_schema_or_broken_csv(self):
        for content in (b"", b"<html>no data</html>", self.payload.replace(b"Ontario Demand", b"Ontario Demand (kWh)"),
                        self.payload.replace(b"Date,Hour,Market Demand,Ontario Demand", b"Date,Hour,Hour,Ontario Demand"),
                        self.payload + b'"unterminated'):
            with self.subTest(content=content[:50]):
                with self.assertRaises((ValueError, csv.Error)):
                    ieso.parse_report(content)

    def test_invalid_values_dates_and_hours(self):
        for row in ("2023-01-01,0,100,90", "2023-01-01,25,100,90", "2023-01-01,1.0,100,90",
                    "2023-1-1,1,100,90", "2023-02-30,1,100,90", "2023-01-01,1,nan,90",
                    "2023-01-01,1,100,inf", "2023-01-01,1,-1,90", "2023-01-01,1,100,-1",
                    "2023-01-01,1,100,", "2023-01-01,1,100,90,extra", ""):
            with self.subTest(row=row), self.assertRaises(ValueError):
                ieso.parse_report((",".join(ieso.HEADER) + "\n" + row + "\n").encode())

    def test_gaps_duplicates_and_order_are_errors(self):
        lines = self.payload.splitlines(keepends=True)
        for broken in (lines[:5] + lines[6:], lines[:5] + lines[4:],
                       lines[:4] + [lines[5], lines[4]] + lines[6:]):
            with self.assertRaisesRegex(ValueError, "missing|duplicate|out-of-order"):
                ieso.parse_report(b"".join(broken))

    def test_spring_dst_does_not_delete_a_market_hour(self):
        observations, _ = ieso.parse_report(report_bytes(date(2023, 3, 11), 3))
        self.assertEqual(len(observations), 72)
        local = [ieso.calendar_fields(row.start) for row in observations]
        starts = [point["local_start"] for point in local if point["local_start"].startswith("2023-03-12")]
        self.assertEqual(len(starts), 23)
        self.assertFalse(any("T02:" in value for value in starts))
        self.assertTrue(any(value.endswith("-04:00") for value in starts))
        self.assertEqual(sum(point["market_date"] == "2023-03-12" for point in local), 24)

    def test_autumn_dst_keeps_both_local_ones_with_offsets_and_fold(self):
        observations, _ = ieso.parse_report(report_bytes(date(2023, 11, 4), 3))
        local = [ieso.calendar_fields(row.start) for row in observations]
        ones = [point for point in local if point["local_start"].startswith("2023-11-05T01:")]
        self.assertEqual(len(observations), 72)
        self.assertEqual(len(ones), 2)
        self.assertEqual({point["local_fold"] for point in ones}, {0, 1})
        self.assertEqual({point["local_start"][-6:] for point in ones}, {"-04:00", "-05:00"})
        self.assertNotEqual(ones[0]["start_utc"], ones[1]["start_utc"])

    def test_complete_year_and_leap_year_validation(self):
        for year, days in ((2023, 365), (2020, 366)):
            payload = report_bytes(date(year, 1, 1), days)
            rows, _ = ieso.parse_report(payload, expected_year=year)
            self.assertEqual(len(rows), days * 24)
        with self.assertRaisesRegex(ValueError, "complete"):
            ieso.parse_report(self.payload, expected_year=2023)
        with self.assertRaisesRegex(ValueError, "requested year"):
            ieso.parse_report(self.payload, expected_year=2022)

    def test_multiple_files_must_join_without_gaps_or_overlap(self):
        other = self.directory / "other.csv"
        other.write_bytes(report_bytes(date(2023, 1, 31), 2))
        rows, sources = ieso.load_reports([self.path, other])
        self.assertEqual(len(rows), 32 * 24)
        self.assertEqual(len(sources), 2)
        for paths in ([self.path, self.path], [other, self.path]):
            with self.assertRaises(ValueError):
                ieso.load_reports(paths)
        other.write_bytes(report_bytes(date(2023, 2, 1), 2))
        with self.assertRaisesRegex(ValueError, "missing"):
            ieso.load_reports([self.path, other])

    def test_snapshot_hash_mismatch_is_not_ignored(self):
        metadata = self.path.with_name("snapshot.json")
        for content in ({"raw_file": self.path.name, "raw_sha256": "wrong"}, []):
            metadata.write_text(json.dumps(content))
            with self.assertRaisesRegex(ValueError, "snapshot metadata"):
                ieso.load_reports([self.path])

    def test_origin_requires_offset_and_exact_hour(self):
        self.assertEqual(ieso.parse_origin("2023-01-10T13:00:00Z"), self.origin)
        for value in ("2023-01-10T08:00", "2023-01-10T08:30-05:00", "2023-01-10T08:00:01Z"):
            with self.assertRaises(ValueError):
                ieso.parse_origin(value)

    def test_assumed_availability_uses_report_day_not_midnight_hour_end(self):
        first = ieso.market_midnight(date(2023, 1, 9))
        expected = ieso.parse_origin("2023-01-10T08:00-05:00")
        self.assertEqual(ieso.assumed_available_at(first), expected)
        self.assertEqual(ieso.assumed_available_at(first + 23 * ieso.HOUR), expected)
        self.assertEqual(ieso.assumed_available_at(first + 24 * ieso.HOUR), expected + 24 * ieso.HOUR)

    def test_all_24_predictions_use_available_sources_and_disclose_recursion(self):
        points = baselines.forecast(self.observations, self.origin)["points"]
        self.assertEqual(len(points), 24)
        self.assertEqual(points[0]["start_utc"], ieso.iso(self.origin))
        self.assertEqual(points[-1]["end_utc"], ieso.iso(self.origin + 24 * ieso.HOUR))
        self.assertEqual(sum(point["models"]["previous_day"]["lag_periods"] > 1 for point in points), 8)
        for point in points:
            target = ieso.parse_origin(point["start_utc"])
            for name, period in baselines.PERIODS.items():
                prediction = point["models"][name]
                source = ieso.parse_origin(prediction["source_start_utc"])
                self.assertEqual(source, target - period * prediction["lag_periods"] * ieso.HOUR)
                self.assertLessEqual(ieso.assumed_available_at(source), self.origin)
                self.assertLessEqual(source + ieso.HOUR, self.origin)
                self.assertEqual(prediction["prediction_mw"], self.observations[source].demand_mw)
                if name == "previous_week":
                    self.assertEqual(prediction["lag_periods"], 1)

    def test_future_and_not_yet_published_values_cannot_change_predictions(self):
        changed = {start: (ieso.Observation(start, 999999) if ieso.assumed_available_at(start) > self.origin else row)
                   for start, row in self.observations.items()}
        self.assertEqual(baselines.forecast(self.observations, self.origin), baselines.forecast(changed, self.origin))
        past_only = {start: row for start, row in self.observations.items()
                     if ieso.assumed_available_at(start) <= self.origin}
        self.assertEqual(baselines.forecast(self.observations, self.origin), baselines.forecast(past_only, self.origin))

    def test_missing_available_lag_is_not_replaced_by_an_older_one(self):
        changed = dict(self.observations)
        del changed[self.origin - 24 * ieso.HOUR]
        with self.assertRaisesRegex(ValueError, "missing available seasonal source"):
            baselines.forecast(changed, self.origin)
        for period in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                baselines.seasonal_value(changed, self.origin, self.origin, period)

    def test_backtest_counts_peaks_and_metrics_on_identical_rows(self):
        result = baselines.backtest(self.observations, date(2023, 1, 10), date(2023, 1, 13))
        self.assertEqual(result["counts"], {"origins": 3, "hourly_targets": 72})
        points = [point for window in result["windows"] for point in window["points"]]
        for model in baselines.PERIODS:
            errors = [point["models"][model]["prediction_mw"] - point["actual_mw"] for point in points]
            metrics = result["metrics"][model]
            self.assertAlmostEqual(metrics["hourly_mae_mw"], sum(map(abs, errors)) / 72)
            self.assertAlmostEqual(metrics["hourly_rmse_mw"], math.sqrt(sum(error ** 2 for error in errors) / 72))
            peaks = [abs(max(point["models"][model]["prediction_mw"] for point in window["points"]) -
                         max(point["actual_mw"] for point in window["points"])) for window in result["windows"]]
            self.assertAlmostEqual(metrics["window_peak_mae_mw"], sum(peaks) / 3)
        self.assertEqual(result["metrics"]["previous_day"]["recursive_hours"], 24)

    def test_constant_series_and_zero_demand_have_defined_errors(self):
        for value in (0, 100):
            rows, _ = ieso.parse_report(report_bytes(constant=value))
            result = baselines.backtest({row.start: row for row in rows}, date(2023, 1, 10), date(2023, 1, 11))
            for metrics in result["metrics"].values():
                self.assertEqual(metrics["hourly_mae_mw"], 0)
                self.assertEqual(metrics["hourly_rmse_mw"], 0)
                self.assertEqual(metrics["window_peak_mae_mw"], 0)
                self.assertEqual(metrics["wape_percent"], None if value == 0 else 0)

    def test_backtest_missing_target_fails_without_trimming(self):
        changed = dict(self.observations)
        del changed[self.origin + 12 * ieso.HOUR]
        with self.assertRaisesRegex(ValueError, "missing target"):
            baselines.backtest(changed, date(2023, 1, 10), date(2023, 1, 11))
        with self.assertRaises(ValueError):
            baselines.backtest(changed, date(2023, 1, 11), date(2023, 1, 10))

    def test_provenance_determinism_and_explicit_configuration(self):
        first = baselines.evaluate([self.path], origin=self.origin)
        second = baselines.evaluate([self.path], origin=self.origin)
        first.pop("runtime_seconds")
        second.pop("runtime_seconds")
        self.assertEqual(first, second)
        self.assertFalse(first["availability"]["point_in_time"])
        self.assertEqual(first["inputs"][0]["sha256"], ieso.digest(self.payload))
        self.assertEqual(first["unit"], "MW")
        for name, expected in first["source_hashes"].items():
            self.assertEqual(expected, ieso.digest((ROOT / name).read_bytes()))
        for kwargs in ({}, {"start": date(2023, 1, 1)}, {"origin": self.origin, "start": date(2023, 1, 1)}):
            with self.assertRaises(ValueError):
                baselines.evaluate([self.path], **kwargs)

    def test_cli_sample_refuses_overwrite_and_requires_acknowledgment(self):
        output = self.directory / "forecast.json"
        command = [sys.executable, str(ROOT / "apps/energy/baselines.py"), "forecast", "--csv", str(self.path),
                   "--origin", self.origin.isoformat(), "--output", str(output)]
        rejected = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("--retrospective", rejected.stderr)
        self.assertFalse(output.exists())
        completed = subprocess.run(command + ["--retrospective"], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        original = output.read_bytes()
        self.assertEqual(len(json.loads(original)["points"]), 24)
        rejected = subprocess.run(command + ["--retrospective"], capture_output=True, text=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertEqual(output.read_bytes(), original)

    def test_cli_backtest_and_failure_leave_no_partial_report(self):
        output = self.directory / "backtest.json"
        command = [sys.executable, str(ROOT / "apps/energy/baselines.py"), "backtest", "--csv", str(self.path),
                   "--start", "2023-01-10", "--end", "2023-01-13", "--retrospective", "--output", str(output)]
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(output.read_bytes())["counts"]["origins"], 3)
        failed_output = self.directory / "failed.json"
        command[-1] = str(failed_output)
        command[command.index("--end") + 1] = "2023-02-02"
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("missing target", completed.stderr)
        self.assertFalse(failed_output.exists())

    def test_download_saves_untouched_bytes_and_provenance_with_no_overwrite(self):
        payload = report_bytes(days=365)
        output = self.directory / "snapshot"
        fetch = Mock(return_value=(payload, {"Last-Modified": "provider-date"}))
        metadata = ieso.download(2023, output, fetch=fetch)
        self.assertEqual((output / metadata["raw_file"]).read_bytes(), payload)
        self.assertEqual(metadata["raw_sha256"], ieso.digest(payload))
        self.assertEqual(metadata["rows"], 8760)
        self.assertIn("revisions", metadata["adjustment"])
        self.assertEqual(metadata["http_headers"]["Last-Modified"], "provider-date")
        fetch.assert_called_once_with(ieso.BASE_URL + "PUB_Demand_2023.csv")
        rows, sources = ieso.load_reports([output / metadata["raw_file"]])
        self.assertEqual(len(rows), 8760)
        self.assertEqual(sources[0]["snapshot"]["metadata"], metadata)
        with self.assertRaisesRegex(ValueError, "already exists"):
            ieso.download(2023, output, fetch=fetch)
        self.assertEqual(fetch.call_count, 1)

    def test_download_validates_before_writing_and_does_not_retry(self):
        output = self.directory / "bad-snapshot"
        for year in (2002, 9999, True):
            with self.assertRaises(ValueError):
                ieso.download(year, output)
        for payload in (b"<html>challenge</html>", self.payload):
            fetch = Mock(return_value=(payload, {}))
            with self.assertRaises(ValueError):
                ieso.download(2023, output, fetch=fetch)
            self.assertEqual(fetch.call_count, 1)
            self.assertFalse(output.exists())
        fetch = Mock(side_effect=OSError("provider unavailable"))
        with self.assertRaises(OSError):
            ieso.download(2023, output, fetch=fetch)
        self.assertEqual(fetch.call_count, 1)
        self.assertFalse(output.exists())

    def test_network_whitelist_redirect_and_size_limit(self):
        for url in ("http://reports-public.ieso.ca/public/Demand/PUB_Demand_2023.csv",
                    "https://example.com/PUB_Demand_2023.csv", ieso.BASE_URL + "PUB_Demand_2023.csv?x=1"):
            with self.assertRaises(ValueError):
                ieso.fetch_report(url)
        with self.assertRaisesRegex(ValueError, "redirect"):
            ieso.SameSourceRedirect().redirect_request(Request(ieso.BASE_URL + "PUB_Demand_2023.csv"),
                                                      None, 302, "moved", {}, "http://example.com")
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = b"x" * (ieso.MAX_BYTES + 1)
        response.headers = {}
        with patch("ieso.build_opener") as opener:
            opener.return_value.open.return_value = response
            with self.assertRaisesRegex(ValueError, "size limit"):
                ieso.fetch_report(ieso.BASE_URL + "PUB_Demand_2023.csv")
            response.read.assert_called_once_with(ieso.MAX_BYTES + 1)
        with self.assertRaisesRegex(ValueError, "size limit"):
            ieso.parse_report(b"x" * (ieso.MAX_BYTES + 1))


if __name__ == "__main__":
    unittest.main()
