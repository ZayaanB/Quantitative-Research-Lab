import importlib.util
import math
import os
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dashboard", ROOT / "apps/demo/serve.py")
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)
ENGINE = Path(os.environ.get("MC_ENGINE", ROOT / "build/mcprice"))
OUTPUT = """Monte Carlo estimate: 10.5
Analytical price: 10.45
95% confidence interval: [10.4, 10.6]
Runtime: 0.1 s
Throughput: 10 M paths/s
Delta: 0.6
Gamma: 0.02
Vega (per 1 vol point): 0.4
Greek runtime (5 runs): 0.5 s
"""


def query(**values):
    return {key: [str(value)] for key, value in values.items()}


class DashboardTests(unittest.TestCase):
    def test_checkpoint_budgets_are_even_increasing_and_finish_at_requested_paths(self):
        for paths, checkpoints in ((1000, 120), (100000000, 2), (1000000, 24)):
            with self.subTest(paths=paths, checkpoints=checkpoints), patch.object(
                    dashboard, "native", return_value=OUTPUT) as native:
                events = list(dashboard.pricing(ENGINE, query(paths=paths, checkpoints=checkpoints)))
                counts = [event["paths"] for event in events if "paths" in event]
                self.assertEqual(len(counts), checkpoints)
                self.assertEqual(counts[-1], paths)
                self.assertTrue(all(count >= 4 and count % 2 == 0 for count in counts))
                self.assertTrue(all(a < b for a, b in zip(counts, counts[1:])))
                for count, call in zip(counts, native.call_args_list):
                    args = call.args
                    self.assertEqual(int(args[args.index("--paths") + 1]), count)

    def test_invalid_values_fail_before_starting_native_compute(self):
        cases = [query(paths=1001), query(paths=100000002), query(paths="1000.5"),
                 query(checkpoints=1), query(checkpoints=121), query(checkpoints="2.5"),
                 query(vol="nan"), query(maturity="inf"), query(maturity=-1),
                 query(type="asian-call", paths=1000000, steps=4096),
                 query(type="asian-call", steps=4097), query(type="other"),
                 query(type="asian-call", steps="2.5")]
        for values in cases:
            with self.subTest(values=values), patch.object(dashboard, "native") as native:
                with self.assertRaises(ValueError):
                    list(dashboard.pricing(ENGINE, values))
                native.assert_not_called()

    def test_long_asian_path_keeps_all_monitoring_steps(self):
        with patch.object(dashboard, "native", return_value=OUTPUT):
            events = list(dashboard.pricing(ENGINE, query(
                type="asian-call", paths=1000, steps=4096, checkpoints=2)))
        self.assertEqual(len(events[0]["samples"]), 8)
        for sample in events[0]["samples"]:
            self.assertEqual(len(sample["path"]), 4097)
            self.assertTrue(all(math.isfinite(value) for value in sample["path"]))
            self.assertAlmostEqual(sample["average"], sum(sample["path"][1:]) / 4096)
        self.assertEqual(events[-1]["paths"], 1000)

    def test_nonfinite_native_results_are_rejected(self):
        with self.assertRaises(ValueError):
            dashboard.extract({"price": "nan"}, "price")
        with self.assertRaises(ValueError):
            dashboard.interval({"range": "[0, inf]"}, "range")

    def test_long_scenario_horizon_keeps_every_forecast_session(self):
        output = """95% model interval: [90, 110]
Median price: 100
Current selected price: 100
Probability above today: 50%
Annualized volatility: 20%
Annualized drift: 5%
"""
        with patch.object(dashboard, "native", return_value=output) as native:
            result = dashboard.scenarios(ENGINE, query(horizon=252), ROOT / "examples/sample_prices.csv")
        self.assertEqual(len(result["fan"]), 253)
        self.assertEqual(result["fan"][-1]["step"], 252)
        self.assertEqual(result["rows"], [])
        self.assertEqual(native.call_count, 252)
        with patch.object(dashboard, "native") as native:
            with self.assertRaises(ValueError):
                dashboard.scenarios(ENGINE, query(horizon=253), ROOT / "examples/sample_prices.csv")
            native.assert_not_called()

    @unittest.skipUnless(ENGINE.is_file(), "Build mcprice to test native boundary cases")
    def test_zero_volatility_and_expiry_produce_finite_native_results(self):
        for kind in ("call", "put", "asian-call"):
            for values in (dict(vol=0), dict(maturity=0), dict(vol=0, maturity=0, rate=-.1)):
                with self.subTest(kind=kind, values=values):
                    events = list(dashboard.pricing(ENGINE, query(
                        type=kind, paths=1000, checkpoints=2, **values)))
                    checkpoints = [event for event in events if "estimate" in event]
                    self.assertEqual(len(checkpoints), 2)
                    self.assertTrue(all(math.isfinite(event["estimate"]) for event in checkpoints))


if __name__ == "__main__":
    unittest.main()
