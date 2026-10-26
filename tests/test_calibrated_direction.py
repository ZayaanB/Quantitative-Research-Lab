import copy
import csv
from datetime import date, timedelta
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import calibrated_direction as calibrated
from benchmark_forecasts import sha256
from direction_forecasts import classification_metrics
from stock_features import CONTEXT_SYMBOLS, OHLCV, evaluate as prepare_features


HAS_MODELS = all(importlib.util.find_spec(name) is not None for name in ("sklearn", "lightgbm", "numpy"))


class CalibratedDirectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.dates = []
        day = date(2013, 1, 1)
        while len(self.dates) < 1700:
            if day.weekday() < 5:
                self.dates.append(day.isoformat())
            day += timedelta(days=1)
        settings = {"schema_version": 1, "lookback_days": 80, "validation_start": self.dates[1200],
                    "holdout_start": self.dates[1500], "test_end": self.dates[-1],
                    "bootstrap": {"samples": 100, "seed": 42, "block_sizes": [1, 2, 4]}}
        self.config, self.context_config = {**settings, "assets": []}, {**settings, "assets": []}
        for position, symbol in enumerate(("WAVE", "FLAT", *CONTEXT_SYMBOLS)):
            path = self.directory / f"{symbol}.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("Date", "Adj Close", *OHLCV))
                for index, day in enumerate(self.dates):
                    price = 100 if symbol == "FLAT" else 100 * math.exp(
                        0.0001 * index + 0.02 * math.sin(index / (5 + position)))
                    writer.writerow((day, price, price, price + 1, price - 1, price, 1000 + index))
            meta = self.directory / f"{symbol}.meta"
            meta.write_text(f"provider=synthetic\nsymbol={symbol}\nadjustment=none\n"
                            f"retrieved_on={self.dates[-1]}\nsource_url=local:fixture\n"
                            "price_column=Adj Close\nfrequency=daily\n")
            entry = {"symbol": symbol, "csv": path.name, "metadata": meta.name,
                     "csv_sha256": sha256(path.read_bytes()), "metadata_sha256": sha256(meta.read_bytes())}
            (self.context_config if symbol in CONTEXT_SYMBOLS else self.config)["assets"].append(entry)
        self.manifest, self.context_manifest = self.directory / "stocks.json", self.directory / "contexts.json"
        self.save_manifests()

    def save_manifests(self):
        self.manifest.write_text(json.dumps(self.config))
        self.context_manifest.write_text(json.dumps(self.context_config))

    def mutate(self, symbol, start, invalid_volume=False):
        asset = next(asset for config in (self.config, self.context_config) for asset in config["assets"]
                     if asset["symbol"] == symbol)
        path = self.directory / asset["csv"]
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        for row in rows[start:]:
            for name in ("Adj Close", "Open", "High", "Low", "Close"):
                row[name] = str(float(row[name]) * 1.7)
            row["Volume"] = "nan" if invalid_volume else "9000000"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        asset["csv_sha256"] = sha256(path.read_bytes())
        self.save_manifests()

    def evaluate(self):
        return calibrated.evaluate(self.manifest, self.context_manifest)

    def test_selection_is_predeclared_baseline_wins_ties_and_mismatches_fail(self):
        rows = [{"symbol": "SYN", "origin_date": "2015-01-02", "target_date": "2015-01-05", "actual_up": 1}]
        blocks = {candidate["id"]: [{"points": copy.deepcopy(rows), "block_id": str(year),
                   "metrics_by_asset": {"SYN": {"brier_score": 0.25}}} for year in (2015, 2016)]
                  for candidate in calibrated.CANDIDATES}
        candidate, scores = calibrated.select_candidate(blocks)
        self.assertEqual(candidate["id"], "historical_up")
        self.assertEqual(len(scores), 5)
        blocks["logistic_c0.1"][0]["metrics_by_asset"]["SYN"]["brier_score"] = 0.1
        self.assertEqual(calibrated.select_candidate(blocks)[0]["id"], "logistic_c0.1")
        changed = copy.deepcopy(blocks)
        changed["logistic_c1"][0]["points"][0]["target_date"] = "2015-01-06"
        with self.assertRaisesRegex(ValueError, "grids differ"):
            calibrated.select_candidate(changed)
        blocks.pop("lightgbm_l2_10")
        with self.assertRaisesRegex(ValueError, "every predeclared"):
            calibrated.select_candidate(blocks)

    def test_oof_baseline_is_past_only_and_purges_year_boundary_targets(self):
        prepared = prepare_features(self.manifest, self.context_manifest)
        histories = {group["symbol"]: group["points"] for group in prepared["results"]}
        block = calibrated.oof_block(histories, 2015, 20, 80, calibrated.CANDIDATES[0])
        self.assertEqual(set(block["fit"]["by_asset"]), {"WAVE", "FLAT"})
        for row in block["points"]:
            self.assertLess(row["origin_date"], row["target_date"])
            self.assertLess(row["target_date"], "2016-01-01")
            self.assertLess(block["fit"]["by_asset"][row["symbol"]]["last_training_target"], "2015-01-01")
        modified = copy.deepcopy(histories)
        for points in modified.values():
            for point in points:
                if point["origin_date"] >= "2016-01-01":
                    point["current_price"] *= 100
                    point["features"] = [999.0] * 28
        self.assertEqual(block, calibrated.oof_block(modified, 2015, 20, 80, calibrated.CANDIDATES[0]))
        with self.assertRaisesRegex(ValueError, "no asset was dropped"):
            calibrated.oof_block(histories, 2030, 20, 80, calibrated.CANDIDATES[0])

    def test_bad_data_blocks_before_fit_or_dependency_lookup(self):
        self.mutate("WAVE", 30, invalid_volume=True)
        with patch.object(calibrated, "version", side_effect=AssertionError("dependencies queried")), \
                patch.object(calibrated, "predict_candidate", side_effect=AssertionError("fit ran")):
            report = self.evaluate()
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["results"], [])

    @unittest.skipUnless(HAS_MODELS, "optional direction dependencies are not installed")
    def test_calibrator_uses_only_oof_rows_and_handles_extremes(self):
        rows = [{"origin_date": f"2015-01-{index % 20 + 1:02}", "actual_up": index % 2,
                 "probability": 0.2 if index % 2 == 0 else 0.8} for index in range(80)]
        first, fitted = calibrated.calibrate(rows, [0.2, 0.8])
        repeated, same = calibrated.calibrate(rows, [0.2, 0.8, 0.0, 1.0])
        self.assertEqual(fitted, same)
        self.assertEqual(first, repeated[:2])
        self.assertTrue(all(0 <= value <= 1 and math.isfinite(value) for value in repeated))
        self.assertLess(first[0], first[1])
        flat = [{**row, "actual_up": 0} for row in rows]
        values, fitted = calibrated.calibrate(flat, [0.2, 0.8])
        self.assertEqual(values, [0, 0])
        self.assertEqual(fitted["fit_kind"], "constant_single_class")
        for invalid_rows, current in ((rows[:10], [0.5]), (rows, [float("nan")]), (rows, [])):
            with self.assertRaises(ValueError):
                calibrated.calibrate(invalid_rows, current)

    @unittest.skipUnless(HAS_MODELS, "optional direction dependencies are not installed")
    def test_outer_results_preserve_grids_baselines_and_temporal_evidence(self):
        report = self.evaluate()
        self.assertEqual(report["status"], "ok")
        self.assertEqual(len(report["results"]), 6)
        self.assertEqual(len(report["outer_folds"]), 6)
        blocks = {block["block_id"]: block for block in report["oof_blocks"]}
        self.assertEqual(len(blocks), len(report["oof_blocks"]))
        for fold in report["outer_folds"]:
            self.assertEqual(len(fold["candidate_scores"]), 5)
            calibration = blocks[fold["calibration_block_id"]]
            self.assertLessEqual(calibration["end_exclusive"], fold["fold_start"])
            self.assertLess(fold["calibration_last_target"], fold["fold_start"])
            self.assertLess(fold["base_fit"]["last_training_target"], fold["fold_start"])
            for candidate in fold["candidate_scores"]:
                for block_id in candidate["block_ids"]:
                    block = blocks[block_id]
                    self.assertLessEqual(block["end_exclusive"], calibration["start"])
                    self.assertLess(block["fit"]["last_training_target"], block["start"])
                    self.assertTrue(all(point["target_date"] < block["end_exclusive"] for point in block["points"]))
        for group in report["results"]:
            horizon = group["horizon_days"]
            self.assertEqual([point["origin_index"] for point in group["points"]], list(range(1200, 1500 - horizon, horizon)))
            for point in group["points"]:
                self.assertLess(point["pooled_last_training_target"], point["fold_start"])
                self.assertLess(point["target_date"], self.config["holdout_start"])
                self.assertEqual(point["actual_up"], int(point["actual_price"] > point["current_price"]))
                self.assertEqual(point["always_up"], 1.0)
                if point["selected_candidate"] == "historical_up":
                    self.assertEqual(point["selected_raw"], point["historical_up"])
            self.assertEqual(group["metrics"]["calibrated"], classification_metrics(
                [point["actual_up"] for point in group["points"]], [point["calibrated"] for point in group["points"]]))

    @unittest.skipUnless(HAS_MODELS, "optional direction dependencies are not installed")
    def test_future_and_reserved_rows_cannot_change_past_choices_or_calibration(self):
        original = self.evaluate()
        for symbol in ("WAVE", "FLAT", *CONTEXT_SYMBOLS):
            self.mutate(symbol, 1500, invalid_volume=True)
        reserved = self.evaluate()
        for key in ("results", "outer_folds", "oof_blocks", "aggregate_metrics"):
            self.assertEqual(original[key], reserved[key])
        self.mutate("WAVE", 1350)
        changed = self.evaluate()
        self.assertEqual(original["outer_folds"], changed["outer_folds"])
        self.assertEqual(original["oof_blocks"], changed["oof_blocks"])
        for before, after in zip(original["results"], changed["results"]):
            for first, second in zip(before["points"], after["points"]):
                if first["target_date"] < self.dates[1350]:
                    self.assertEqual(first, second)

    @unittest.skipUnless(HAS_MODELS, "optional direction dependencies are not installed")
    def test_native_failures_have_no_fallback_and_no_in_sample_calibration(self):
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
        rows = [{"origin_date": f"2015-01-{index % 20 + 1:02}", "actual_up": index % 2,
                 "probability": 0.2 if index % 2 == 0 else 0.8} for index in range(80)]
        with patch.object(LogisticRegression, "fit", side_effect=ConvergenceWarning("simulated")):
            with self.assertRaisesRegex(ValueError, "no fallback"):
                calibrated.calibrate(rows, [0.2])
        with patch.object(calibrated, "predict_candidate", side_effect=ValueError("fit failed")):
            with self.assertRaisesRegex(ValueError, "fit failed"):
                self.evaluate()

    @unittest.skipUnless(HAS_MODELS, "optional direction dependencies are not installed")
    def test_cli_will_not_overwrite_and_rejects_insufficient_history(self):
        output = self.directory / "calibrated.json"
        command = [sys.executable, str(ROOT / "python/calibrated_direction.py"), "--manifest", str(self.manifest),
                   "--context-manifest", str(self.context_manifest), "--output", str(output)]
        process = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        saved = output.read_bytes()
        self.assertEqual(json.loads(saved)["task"], "binary_direction")
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.assertEqual(saved, output.read_bytes())
        for config in (self.config, self.context_config):
            config["validation_start"] = self.dates[120]
        self.save_manifests()
        command[-1] = str(self.directory / "failed.json")
        process = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(Path(command[-1]).exists())


if __name__ == "__main__":
    unittest.main()
