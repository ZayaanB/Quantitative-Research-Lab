import copy
import csv
from datetime import date, timedelta
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import stock_features as features
from benchmark_forecasts import sha256
from boosted_forecasts import training_set


class StockFeatureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.dates = []
        day = date(2020, 1, 1)
        while len(self.dates) < 100:
            if day.weekday() < 5:
                self.dates.append(day.isoformat())
            day += timedelta(days=1)
        self.config = {"schema_version": 1, "lookback_days": 60, "validation_start": self.dates[50],
                       "holdout_start": self.dates[80], "test_end": self.dates[-1], "assets": []}
        self.context_config = copy.deepcopy(self.config)
        for position, symbol in enumerate(("WAVE", "FLAT", *features.CONTEXT_SYMBOLS)):
            path = self.directory / f"{symbol}.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("Date", "Adj Close", *features.OHLCV))
                for index, day in enumerate(self.dates):
                    price = 100 if symbol == "FLAT" else 100 * math.exp(
                        0.001 * index + 0.005 * math.sin(index / (position + 1)))
                    writer.writerow((day, price * 0.8, price * 0.99, price * 1.02,
                                     price * 0.98, price, 0 if index % 7 == 0 else 1000 + index))
            meta = self.directory / f"{symbol}.meta"
            meta.write_text(f"provider=synthetic\nsymbol={symbol}\nadjustment=fixture ratio\n"
                            f"retrieved_on={self.dates[-1]}\nsource_url=local:fixture\n"
                            "price_column=Adj Close\nfrequency=daily\n")
            entry = {"symbol": symbol, "csv": path.name, "metadata": meta.name,
                     "csv_sha256": sha256(path.read_bytes()), "metadata_sha256": sha256(meta.read_bytes())}
            (self.context_config if symbol in features.CONTEXT_SYMBOLS else self.config)["assets"].append(entry)
        self.manifest = self.directory / "stocks.json"
        self.context_manifest = self.directory / "contexts.json"
        self.save_manifests()

    def save_manifests(self):
        self.manifest.write_text(json.dumps(self.config))
        self.context_manifest.write_text(json.dumps(self.context_config))

    def mutate(self, symbol, transform):
        asset = next(asset for config in (self.config, self.context_config) for asset in config["assets"]
                     if asset["symbol"] == symbol)
        path = self.directory / asset["csv"]
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        transform(rows)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        asset["csv_sha256"] = sha256(path.read_bytes())
        self.save_manifests()

    def evaluate(self):
        return features.evaluate(self.manifest, self.context_manifest)

    def test_schema_provenance_counts_and_deterministic_output(self):
        report = self.evaluate()
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["phase"], "development_features")
        self.assertEqual(report["results"], self.evaluate()["results"])
        self.assertEqual(len(report["feature_names"]), 28)
        self.assertEqual(len(set(report["feature_names"])), 28)
        self.assertEqual(report["manifest_sha256"], sha256(self.manifest.read_bytes()))
        self.assertEqual(report["context_manifest_sha256"], sha256(self.context_manifest.read_bytes()))
        self.assertEqual(len(report["quality"]), 5)
        for name, digest in report["source_hashes"].items():
            self.assertEqual(digest, sha256((ROOT / "python" / name).read_bytes()))
        for group in report["results"]:
            self.assertEqual(group["rows"], 60)
            self.assertEqual(group["warmup_rows"], 20)
            self.assertEqual(group["validation_rows"], 30)
            self.assertEqual(group["first_origin"], self.dates[20])
            self.assertEqual(group["last_origin"], self.dates[79])
            self.assertTrue(all(len(point["features"]) == 28 and all(math.isfinite(value) for value in
                                point["features"]) for point in group["points"]))

    def test_feature_values_use_selected_prices_reported_bars_and_log_volume(self):
        report = self.evaluate()
        point = report["results"][0]["points"][0]
        values = dict(zip(report["feature_names"], point["features"]))
        with (self.directory / "WAVE.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        logs = [math.log(float(row["Adj Close"])) for row in rows]
        daily_returns = [logs[index] - logs[index - 1] for index in range(1, len(logs))]
        self.assertEqual(values["return_20"], logs[20] - logs[0])
        self.assertEqual(values["volatility_5"], statistics.pstdev(daily_returns[15:20]))
        self.assertAlmostEqual(values["intraday_log_range"], math.log(1.02 / 0.98))
        self.assertAlmostEqual(values["intraday_log_return"], math.log(1 / 0.99))
        self.assertAlmostEqual(values["close_location"], 0.0)
        self.assertEqual(values["log_volume_change_1"], math.log1p(1020) - math.log1p(1019))
        self.assertEqual(values["log_volume_relative_20"], math.log1p(1020) - statistics.mean(
            math.log1p(float(row["Volume"])) for row in rows[1:21]))
        for symbol in features.CONTEXT_SYMBOLS:
            with (self.directory / f"{symbol}.csv").open() as stream:
                context = [math.log(float(row["Adj Close"])) for row in csv.DictReader(stream)]
            self.assertEqual(values[f"{symbol.lower()}_return_5"], context[20] - context[15])
            self.assertEqual(values[f"{symbol.lower()}_volatility_20"], statistics.pstdev(
                context[index] - context[index - 1] for index in range(1, 21)))

    def test_valid_future_mutations_cannot_change_earlier_features(self):
        original = self.evaluate()["results"]

        def alter_future(rows):
            for row in rows[56:]:
                for name in ("Adj Close", "Open", "High", "Low", "Close"):
                    row[name] = str(float(row[name]) * 3)
                row["Volume"] = "9000000"

        for symbol in ("WAVE", "FLAT", *features.CONTEXT_SYMBOLS):
            self.mutate(symbol, alter_future)
        changed = self.evaluate()["results"]
        for before, after in zip(original, changed):
            early = lambda group: [point for point in group["points"] if point["origin_date"] <= self.dates[55]]
            self.assertEqual(early(before), early(after))
            self.assertNotEqual(before["points"][-1], after["points"][-1])

    def test_reserved_data_changes_do_not_change_features_or_diagnostics(self):
        original = self.evaluate()

        def alter_reserved(rows):
            for row in rows[80:]:
                row["Adj Close"] = str(float(row["Adj Close"]) * 10)
                row["High"] = "nan"
                row["Volume"] = "invalid"

        for symbol in ("WAVE", "FLAT", *features.CONTEXT_SYMBOLS):
            self.mutate(symbol, alter_reserved)
        changed = self.evaluate()
        self.assertEqual(original["results"], changed["results"])
        self.assertEqual([item["issues"] for item in original["quality"]],
                         [item["issues"] for item in changed["quality"]])

    def test_missing_or_extra_context_sessions_are_never_filled_or_intersected(self):
        self.mutate("SPY", lambda rows: rows.pop(30))
        with self.assertRaisesRegex(ValueError, "session grid"):
            self.evaluate()
        self.mutate("WAVE", lambda rows: rows.pop(30))
        with self.assertRaisesRegex(ValueError, "session grid"):
            self.evaluate()

    def test_context_history_may_start_earlier_but_cannot_start_later(self):
        for symbol in ("WAVE", "FLAT"):
            self.mutate(symbol, lambda rows: rows.__delitem__(slice(0, 5)))
        report = self.evaluate()
        self.assertTrue(all(group["first_origin"] == self.dates[25] for group in report["results"]))
        self.mutate("QQQ", lambda rows: rows.__delitem__(slice(0, 6)))
        with self.assertRaisesRegex(ValueError, "session grid"):
            self.evaluate()

    def test_invalid_ohlcv_hashes_and_missing_stock_columns_block_export(self):
        self.mutate("WAVE", lambda rows: rows[30].update(Volume="nan", Low="10000"))
        report = self.evaluate()
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["results"], [])
        self.assertIn("invalid_numeric_column", [item["code"] for item in report["quality"][0]["issues"]])
        self.mutate("WAVE", lambda rows: [row.pop(name) for row in rows for name in features.OHLCV])
        report = self.evaluate()
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["quality"][0]["issues"][-1]["code"], "missing_required_ohlcv")
        history = self.directory / "SPY.csv"
        history.write_text(history.read_text() + "\n")
        report = self.evaluate()
        self.assertEqual(report["status"], "blocked")
        self.assertIn("checksum", report["quality"][2]["issues"][0]["detail"])

    def test_price_only_contexts_work_and_context_order_does_not_matter(self):
        original = self.evaluate()["results"]
        for symbol in features.CONTEXT_SYMBOLS:
            self.mutate(symbol, lambda rows: [row.pop(name) for row in rows for name in features.OHLCV])
        self.context_config["assets"].reverse()
        self.save_manifests()
        self.assertEqual(original, self.evaluate()["results"])

    def test_flat_prices_zero_volume_and_flat_bars_are_finite(self):
        self.mutate("FLAT", lambda rows: [row.update(Open="100", High="100", Low="100", Close="100",
                                                   Volume="0") for row in rows])
        report = self.evaluate()
        for point in report["results"][1]["points"]:
            self.assertEqual(point["features"][:13], [0.0] * 13)

    def test_within_day_ohlc_scaling_does_not_mix_adjusted_and_raw_levels(self):
        original = self.evaluate()["results"]
        self.mutate("WAVE", lambda rows: [row.update({name: str(float(row[name]) * 4)
                                                     for name in ("Open", "High", "Low", "Close")}) for row in rows])
        changed = self.evaluate()["results"]
        for before, after in zip(original[0]["points"], changed[0]["points"]):
            for a, b in zip(before["features"], after["features"]):
                self.assertAlmostEqual(a, b, places=12)

    def test_manifests_and_warmup_require_explicit_compatible_inputs(self):
        original = copy.deepcopy(self.context_config)
        for alter in (lambda c: c.update(holdout_start=self.dates[81]),
                      lambda c: c["assets"].pop(),
                      lambda c: c["assets"][0].update(symbol="OTHER"),
                      lambda c: c.update(schema_version=True)):
            self.context_config = copy.deepcopy(original)
            alter(self.context_config)
            self.save_manifests()
            with self.assertRaises(ValueError):
                self.evaluate()
        self.context_config = original
        for symbol in ("WAVE", "FLAT"):
            self.mutate(symbol, lambda rows: rows.__delitem__(slice(0, 60)))
        with self.assertRaisesRegex(ValueError, "21 development sessions"):
            self.evaluate()

    def test_legacy_training_features_and_targets_are_exactly_unchanged(self):
        prices = [100 * math.exp(0.0003 * index + 0.01 * math.sin(index)) for index in range(150)]
        logs = [math.log(price) for price in prices]
        returns = [logs[index] - logs[index - 1] for index in range(1, len(logs))]

        def previous_formula(index):
            return ([logs[index] - logs[index - lag] for lag in (1, 2, 5, 10, 20)] +
                    [statistics.pstdev(returns[index - span:index]) for span in (5, 20)] +
                    [logs[index] - statistics.mean(logs[index - 19:index + 1])])

        for horizon in (1, 5, 20):
            x, y, current = training_set(prices, horizon, 80)
            indices = range(69, 150 - horizon)
            self.assertEqual(x, [previous_formula(index) for index in indices])
            self.assertEqual(y, [logs[index + horizon] - logs[index] for index in indices])
            self.assertEqual(current, previous_formula(149))

    def test_cli_no_overwrite_and_failures_are_explicit(self):
        output = self.directory / "features.json"
        command = [sys.executable, str(ROOT / "python/stock_features.py"), "--manifest", str(self.manifest),
                   "--context-manifest", str(self.context_manifest), "--output", str(output)]
        process = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        saved = output.read_bytes()
        self.assertEqual(json.loads(saved)["results"], self.evaluate()["results"])
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.assertEqual(saved, output.read_bytes())
        self.mutate("WAVE", lambda rows: rows[30].update(Volume="-1"))
        blocked = self.directory / "blocked.json"
        command[-1] = str(blocked)
        process = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(json.loads(blocked.read_bytes())["status"], "blocked")
        self.assertIn("blocked by data checks", process.stderr)


if __name__ == "__main__":
    unittest.main()
