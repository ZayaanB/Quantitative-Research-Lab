import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import neural_direction as neural
from direction_forecasts import evaluate as evaluate_fixed, pooled_training_data
from stock_features import CONTEXT_SYMBOLS, FEATURES, evaluate as prepare_features
import test_direction_forecasts as direction_fixtures


HAS_TORCH = all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy"))
HAS_CONTROLS = HAS_TORCH and all(importlib.util.find_spec(name) is not None for name in ("sklearn", "lightgbm"))


class NeuralTests(unittest.TestCase):
    def setUp(self):
        self.fixture = direction_fixtures.DirectionTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.directory = self.fixture.directory
        self.dates = self.fixture.dates
        self.manifest = self.fixture.manifest
        self.context_manifest = self.fixture.context_manifest

    def histories(self):
        report = prepare_features(self.manifest, self.context_manifest)
        return {group["symbol"]: group["points"] for group in report["results"]}

    def evaluate(self):
        return neural.evaluate(self.manifest, self.context_manifest)

    def test_sequences_end_at_origin_and_keep_the_old_training_labels(self):
        histories = self.histories()
        sequences, labels, details = neural.training_sequences(histories, self.dates[120], 20, 80)
        _, expected_labels, expected = pooled_training_data(histories, self.dates[120], 20, 80)
        self.assertEqual(labels, expected_labels)
        self.assertEqual(details, expected)
        self.assertEqual(len(sequences), 120)
        self.assertEqual(len(sequences[0]), 28)
        self.assertEqual(len(sequences[0][0]), 20)
        self.assertEqual(sequences[0][0][-1], histories["FLAT"][20]["features"][0])
        self.assertEqual(sequences[1][0][0], histories["WAVE"][1]["features"][0])
        changed = copy.deepcopy(histories)
        for points in changed.values():
            for point in points:
                if point["origin_date"] >= self.dates[120]:
                    point["features"] = [999.0] * len(FEATURES)
                    point["current_price"] *= 2
        self.assertEqual((sequences, labels, details), neural.training_sequences(changed, self.dates[120], 20, 80))
        with self.assertRaisesRegex(ValueError, "workload was not changed"):
            neural.training_sequences(histories, self.dates[80], 20, 80)

    def test_future_changes_do_not_change_earlier_sequence_windows(self):
        histories = self.histories()
        request = [(self.dates[120], "WAVE", 100)]
        original = neural.windows(histories, request)
        for point in histories["WAVE"][101:]:
            point["features"] = [999.0] * len(FEATURES)
        self.assertEqual(original, neural.windows(histories, request))
        self.assertEqual(neural.environment()["logical_cores"], __import__("os").cpu_count())
        self.assertTrue(neural.peak_rss_bytes() is None or neural.peak_rss_bytes() > 0)

    def test_bad_data_blocks_before_optional_model_import_or_fit(self):
        self.fixture.mutate("WAVE", 30, reserved=True)
        with patch.object(neural, "fit_cnn", side_effect=AssertionError("fit ran")), \
                patch.object(neural, "version", side_effect=AssertionError("dependencies queried")):
            report = self.evaluate()
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["results"], [])

    @unittest.skipUnless(HAS_TORCH, "optional CPU PyTorch is not installed")
    def test_normalizer_uses_training_only_handles_constants_and_rejects_bad_values(self):
        import numpy as np
        train = [[[float(index + channel + step) for step in range(20)] for channel in range(28)] for index in range(40)]
        x, _, fitted = neural.normalize(train, train[:2])
        _, _, changed = neural.normalize(train, [[[1e6] * 20 for _ in range(28)]])
        self.assertEqual(fitted, changed)
        self.assertAlmostEqual(fitted["mean"][0], 29.0)
        self.assertEqual(x.dtype, np.float32)
        self.assertTrue(np.allclose(x.mean(axis=(0, 2)), 0, atol=1e-6))
        zero, _, fitted = neural.normalize([[[1.0] * 20 for _ in range(28)]] * 40, train[:2])
        self.assertTrue((zero == 0).all())
        self.assertEqual(fitted["scale"], [1] * 28)
        for current in ([], [[[float("nan")] * 20 for _ in range(28)]], [[[0.0] * 19 for _ in range(28)]]):
            with self.assertRaises(ValueError):
                neural.normalize(train, current)

    @unittest.skipUnless(HAS_TORCH, "optional CPU PyTorch is not installed")
    def test_native_cnn_is_small_deterministic_and_restores_torch_state(self):
        import torch
        training = [[[math.sin(index / 5 + channel + step / 10) for step in range(20)]
                     for channel in range(28)] for index in range(40)]
        labels = [index % 2 for index in range(40)]
        before = torch.random.get_rng_state().clone()
        threads = torch.get_num_threads()
        deterministic = torch.are_deterministic_algorithms_enabled()
        first, fitted = neural.fit_cnn(training, labels, training[:2])
        repeated, again = neural.fit_cnn(training, labels, training[:2])
        self.assertEqual(first, repeated)
        self.assertEqual({k: v for k, v in fitted.items() if k != "runtime_seconds"},
                         {k: v for k, v in again.items() if k != "runtime_seconds"})
        self.assertEqual(fitted["parameter_count"], 889)
        self.assertEqual(fitted["parameter_bytes"], 3556)
        self.assertEqual(fitted["epochs"], 20)
        self.assertEqual(fitted["device"], "cpu")
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(torch.get_num_threads(), threads)
        self.assertEqual(torch.are_deterministic_algorithms_enabled(), deterministic)
        flat, fitted = neural.fit_cnn(training, [0] * 40, training[:2])
        self.assertEqual(flat, [0.0, 0.0])
        self.assertEqual(fitted["fit_kind"], "constant_single_class")

    @unittest.skipUnless(HAS_TORCH, "optional CPU PyTorch is not installed")
    def test_native_fit_failures_have_no_fallback_and_restore_state(self):
        import torch
        training = [[[math.sin(index + channel + step) for step in range(20)]
                     for channel in range(28)] for index in range(40)]
        before = torch.random.get_rng_state().clone()
        threads = torch.get_num_threads()
        with patch.object(neural, "make_network", side_effect=RuntimeError("simulated")):
            with self.assertRaisesRegex(ValueError, "no fallback"):
                neural.fit_cnn(training, [index % 2 for index in range(40)], training[:2])
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(torch.get_num_threads(), threads)
        with self.assertRaisesRegex(ValueError, "binary"):
            neural.fit_cnn(training, [2] * 40, training[:2])

    @unittest.skipUnless(HAS_CONTROLS, "optional PyTorch/classical control dependencies are not installed")
    def test_native_pipeline_keeps_fixed_folds_grids_and_baselines(self):
        report = self.evaluate()
        self.assertEqual(report["training_mode"], "pooled_cnn")
        self.assertEqual(len(report["shared_fits"]), 6)
        self.assertEqual(len(report["results"]), 6)
        self.assertGreater(report["peak_process_rss_bytes"], 0)
        fixed = evaluate_fixed(self.manifest, self.context_manifest, "pooled")
        for group, reference in zip(report["results"], fixed["results"]):
            self.assertEqual(group["forecasts"], reference["forecasts"])
            for point, expected in zip(group["points"], reference["points"]):
                for field in ("origin_date", "target_date", "actual_up", "current_price", "actual_price", "historical_up"):
                    self.assertEqual(point[field], expected[field])
                self.assertLess(point["pooled_last_training_target"], point["fold_start"])
            for name in ("historical_up", "always_up", "latest_price_not_up"):
                self.assertEqual(group["metrics"][name], reference["metrics"][name])

    @unittest.skipUnless(HAS_TORCH, "optional CPU PyTorch is not installed")
    def test_reserved_and_other_asset_future_mutations_do_not_leak(self):
        original = self.evaluate()
        for symbol in ("WAVE", "FLAT", *CONTEXT_SYMBOLS):
            self.fixture.mutate(symbol, 320, reserved=True)
        reserved = self.evaluate()
        self.assertEqual(original["results"], reserved["results"])
        for before, after in zip(original["shared_fits"], reserved["shared_fits"]):
            self.assertEqual(before["model"]["normalizer"], after["model"]["normalizer"])
        self.fixture.mutate("FLAT", 180)
        changed = self.evaluate()
        for before, after in zip(original["results"], changed["results"]):
            for first, second in zip(before["points"], after["points"]):
                if first["target_date"] < self.dates[180]:
                    self.assertEqual(first, second)

    @unittest.skipUnless(HAS_TORCH, "optional CPU PyTorch is not installed")
    def test_cli_protects_outputs_and_sequence_failures_save_no_partial_scores(self):
        output = self.directory / "cnn.json"
        command = [sys.executable, str(ROOT / "python/neural_direction.py"), "--manifest", str(self.manifest),
                   "--context-manifest", str(self.context_manifest), "--output", str(output)]
        process = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        saved = output.read_bytes()
        self.assertEqual(json.loads(saved)["task"], "binary_direction")
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.assertEqual(saved, output.read_bytes())
        for config in (self.fixture.config, self.fixture.context_config):
            config["validation_start"] = self.dates[80]
        self.fixture.save_manifests()
        command[-1] = str(self.directory / "failed.json")
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.assertFalse(Path(command[-1]).exists())


if __name__ == "__main__":
    unittest.main()
