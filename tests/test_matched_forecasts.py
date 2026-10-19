import copy
import csv
from datetime import date, timedelta
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import matched_forecasts as matched
from benchmark_forecasts import read_manifest, sha256
from ridge_forecasts import evaluate as evaluate_ridge


class MatchedForecastTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.engine = Path(os.environ.get("MC_ENGINE", ROOT / "build/mcprice")).resolve()
        self.dates = []
        day = date(2020, 1, 1)
        while len(self.dates) < 180:
            if day.weekday() < 5:
                self.dates.append(day.isoformat())
            day += timedelta(days=1)
        self.config = {"schema_version": 1, "lookback_days": 80, "validation_start": self.dates[100],
                       "holdout_start": self.dates[160], "test_end": self.dates[-1],
                       "bootstrap": {"samples": 100, "seed": 42, "block_sizes": [1, 2, 4]}, "assets": []}
        for symbol in ("WAVE", "FLAT"):
            history = self.directory / f"{symbol}.csv"
            with history.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["Date", "Adj Close", "Open", "High", "Low", "Close", "Volume"])
                for i, day in enumerate(self.dates):
                    price = 100 if symbol == "FLAT" else 100 * math.exp(0.0003 * i + 0.003 * math.sin(i))
                    writer.writerow([day, price, price, price + 1, price - 1, price, 1000])
            metadata = self.directory / f"{symbol}.meta"
            metadata.write_text(f"provider=synthetic\nsymbol={symbol}\nadjustment=none\n"
                                f"retrieved_on={self.dates[-1]}\nsource_url=local:fixture\n"
                                "price_column=Adj Close\nfrequency=daily\n")
            self.config["assets"].append({"symbol": symbol, "csv": history.name, "metadata": metadata.name,
                                          "csv_sha256": sha256(history.read_bytes()),
                                          "metadata_sha256": sha256(metadata.read_bytes())})
        self.manifest = self.directory / "manifest.json"
        self.save_manifest()
        self.reference = self.directory / "ridge.json"
        self.source = evaluate_ridge(self.manifest)
        self.save_reference()

    def save_manifest(self):
        self.manifest.write_text(json.dumps(self.config))

    def save_reference(self):
        self.reference.write_text(json.dumps(self.source))

    def mutate_history(self, transform):
        asset = self.config["assets"][0]
        history = self.directory / asset["csv"]
        with history.open() as stream:
            rows = list(csv.DictReader(stream))
        transform(rows)
        with history.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        asset["csv_sha256"] = sha256(history.read_bytes())
        self.save_manifest()

    def evaluate(self, **kwargs):
        return matched.evaluate(self.manifest, [self.reference], self.engine, **kwargs)

    def test_actual_engine_pairs_every_date_price_and_baseline(self):
        result = self.evaluate()
        repeated = self.evaluate()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["results"], repeated["results"])
        self.assertEqual(len(result["results"]), 6)
        self.assertEqual(result["references"][0]["sha256"], sha256(self.reference.read_bytes()))
        self.assertTrue(all(item["ohlcv_checked"] for item in result["quality"]))
        for group, source in zip(result["results"], self.source["results"]):
            self.assertEqual(group["forecasts"], len(source["points"]))
            for point, expected in zip(group["points"], source["points"]):
                self.assertEqual(point["origin_date"], expected["origin_date"])
                self.assertEqual(point["target_date"], expected["target_date"])
                self.assertEqual(point["reference_0"], expected["forecast_price"])
                self.assertEqual(point["latest_price_forecast"], point["current_price"])
                self.assertLess(point["target_date"], self.config["holdout_start"])
            if group["symbol"] == "FLAT":
                self.assertEqual(group["metrics"]["latest_price"]["mae"], 0)
                self.assertIsNone(group["metrics"]["latest_price"]["directional_accuracy"])
        self.assertTrue(any(group["dense_gbm_rows_not_scored"] > 0 for group in result["results"]))

    def test_reference_validation_rejects_tampering_before_engine_execution(self):
        original = copy.deepcopy(self.source)
        mutations = (
            lambda r: r.update(schema_version=True),
            lambda r: r.update(phase="holdout"),
            lambda r: r.update(manifest_sha256="0" * 64),
            lambda r: r["config"].update(lookback_days=81),
            lambda r: r["results"].pop(),
            lambda r: r["results"].append(r["results"][0]),
            lambda r: r["results"][0]["points"][0].update(actual_price=999),
            lambda r: r["results"][0]["points"][0].update(current_price=True),
            lambda r: r["results"][0]["points"][0].update(forecast_price=None),
            lambda r: r["results"][0]["points"][0].update(target_date=self.dates[160]),
            lambda r: r["results"][0]["points"][0].update(last_training_target_index=101),
            lambda r: r["results"][0]["points"].reverse(),
            lambda r: r["results"][0].update(forecasts=True),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                self.source = copy.deepcopy(original)
                mutate(self.source)
                self.save_reference()
                with patch.object(matched, "run_benchmark", side_effect=AssertionError("engine ran")):
                    with self.assertRaises(ValueError):
                        self.evaluate()
        self.source = copy.deepcopy(original)
        self.source["results"][0]["points"].pop(2)
        self.source["results"][0]["forecasts"] -= 1
        self.save_reference()
        with self.assertRaisesRegex(ValueError, "grid"):
            self.evaluate()

    def test_multiple_references_require_exact_grids_without_intersection(self):
        second = self.directory / "second.json"
        other = copy.deepcopy(self.source)
        for group in other["results"]:
            for point in group["points"]:
                point["forecast_price"] = point["current_price"]
        second.write_text(json.dumps(other))
        result = matched.evaluate(self.manifest, [self.reference, second], self.engine)
        self.assertEqual(len(result["references"]), 2)
        self.assertEqual(result["mean_relative_mae"]["reference_1"], 1.0)
        other["results"][0]["points"].pop(0)
        other["results"][0]["forecasts"] -= 1
        second.write_text(json.dumps(other))
        with self.assertRaisesRegex(ValueError, "grids differ"):
            matched.evaluate(self.manifest, [self.reference, second], self.engine)
        with self.assertRaisesRegex(ValueError, "duplicate reference"):
            matched.evaluate(self.manifest, [self.reference, self.reference], self.engine)

    def test_bad_ohlcv_blocks_scoring_and_quality_only_saves_issues(self):
        self.mutate_history(lambda rows: rows[20].update(High="1", Volume="nan"))
        with patch.object(matched, "run_benchmark", side_effect=AssertionError("engine ran")):
            result = self.evaluate()
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["results"], [])
        codes = {issue["code"] for issue in result["quality"][0]["issues"]}
        self.assertIn("inconsistent_ohlc", codes)
        self.assertIn("invalid_numeric_column", codes)
        output = self.directory / "quality.json"
        process = subprocess.run([sys.executable, str(ROOT / "python/matched_forecasts.py"),
                                  "--manifest", str(self.manifest), "--quality-only", "--output", str(output)],
                                 text=True, capture_output=True)
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(json.loads(output.read_text())["status"], "blocked")

    def test_session_warnings_are_not_repaired_or_treated_as_calendar_truth(self):
        self.mutate_history(lambda rows: rows[2].update(Date="2020-01-04", Volume="0"))
        result = matched.evaluate(self.manifest, quality_only=True)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["quality"][0]["source_rows"], 180)
        codes = {issue["code"] for issue in result["quality"][0]["issues"]}
        self.assertTrue({"weekend_row", "zero_volume"} <= codes)
        self.mutate_history(lambda rows: [rows.pop(20) for _ in range(5)])
        result = matched.evaluate(self.manifest, quality_only=True)
        self.assertTrue(any(issue["code"] == "long_calendar_gap" for issue in result["quality"][0]["issues"]))

    def test_hash_failure_and_minimal_price_only_inputs(self):
        history = self.directory / self.config["assets"][0]["csv"]
        history.write_text(history.read_text() + "\n")
        result = matched.evaluate(self.manifest, quality_only=True)
        self.assertEqual(result["status"], "blocked")
        self.assertIn("checksum", result["quality"][0]["issues"][0]["detail"])
        self.mutate_history(lambda rows: [row.pop(key) for row in rows for key in tuple(row)
                                          if key not in ("Date", "Adj Close")])
        result = matched.evaluate(self.manifest, quality_only=True)
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["quality"][0]["ohlcv_checked"])

    def test_reserved_prices_do_not_affect_matched_validation(self):
        original = self.evaluate()

        def mutate(rows):
            for row in rows[160:]:
                row["Adj Close"] = str(float(row["Adj Close"]) * 2)
                row["High"] = "1"

        self.mutate_history(mutate)
        self.source["config"] = copy.deepcopy(self.config)
        self.source["manifest_sha256"] = read_manifest(self.manifest)[1]
        self.save_reference()
        changed = self.evaluate()
        self.assertEqual(original["results"], changed["results"])

    def test_cli_never_overwrites_and_rejects_nonfinite_json(self):
        output = self.directory / "comparison.json"
        command = [sys.executable, str(ROOT / "python/matched_forecasts.py"), "--manifest", str(self.manifest),
                   "--reference", str(self.reference), "--engine", str(self.engine), "--output", str(output)]
        first = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        before = output.read_bytes()
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.assertEqual(before, output.read_bytes())
        self.reference.write_text(self.reference.read_text().replace('"penalty": 1.0', '"penalty": NaN'))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.evaluate()

    def test_named_prediction_column_and_perfect_baseline_handling(self):
        for group in self.source["results"]:
            for point in group["points"]:
                point["custom_price"] = point.pop("forecast_price")
                if group["symbol"] == "FLAT":
                    point["custom_price"] = 101
        self.save_reference()
        result = self.evaluate(column="custom_price")
        self.assertEqual(result["references"][0]["prediction_column"], "custom_price")
        self.assertIsNone(result["mean_relative_mae"]["reference_0"])

    def test_duplicate_optional_column_and_duplicate_json_keys_are_rejected(self):
        asset = self.config["assets"][0]
        history = self.directory / asset["csv"]
        with history.open() as stream:
            rows = list(csv.reader(stream))
        rows[0].append("Open")
        for row in rows[1:]:
            row.append(row[2])
        with history.open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)
        asset["csv_sha256"] = sha256(history.read_bytes())
        self.save_manifest()
        quality = matched.evaluate(self.manifest, quality_only=True)
        self.assertEqual(quality["status"], "blocked")
        self.assertIn("duplicate_column", [issue["code"] for issue in quality["quality"][0]["issues"]])
        raw = '{"schema_version": 1, ' + self.reference.read_text()[1:]
        self.reference.write_text(raw)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            matched.read_reference(self.reference, self.config, read_manifest(self.manifest)[1], {}, "forecast_price")


if __name__ == "__main__":
    unittest.main()
