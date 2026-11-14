import copy
from datetime import date
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "energy"))
import boosted_demand as energy
import final_evaluation as final
import ieso
from test_energy_baselines import report_bytes


NATIVE = all(importlib.util.find_spec(name) is not None for name in ("numpy", "lightgbm", "scipy"))


def validation_fixture(environment, warmup, models):
    cutoff = energy.origin_at(date(2023, 1, 1)).isoformat()
    return {"schema_version": 1, "task": "ontario_hourly_demand_uncertainty", "phase": "validation", "status": "ok",
            "mode": "retrospective_scenario", "unit": "MW", "series": "Ontario Demand", "horizon_hours": 24,
            **{key: value for key, value in environment.items() if key != "source_hashes"},
            "source_hashes": {key: value for key, value in environment["source_hashes"].items() if not key.endswith("/final_evaluation.py")},
            "procedure": {"parameters": energy.PARAMETERS, "rounds": energy.ROUNDS, "feature_names": list(energy.FEATURES),
                          "quantiles": [0.05, 0.95], "coverage_target": 0.9},
            "training": {"origin_start_est": "2018-01-15", "origin_end_exclusive_est": "2022-01-01"},
            "calibration": {"origin_start_est": "2022-01-01", "origin_end_exclusive_est": "2023-01-01", "rows": 8736,
                            "purged_origin_dates": ["2022-12-31"], "widening_mw": 1.0,
                            "latest_label_assumed_available_at_utc": cutoff, "frozen_at_utc": cutoff},
            "evaluation": {"origin_start_est": "2023-01-01", "origin_end_exclusive_est": "2023-12-31", "rows": 8736},
            "model_text": models["point"]["model_text"], "model_sha256": models["point"]["sha256"],
            "quantile_models": {key: value for key, value in models.items() if key != "point"},
            "inputs": [{"sha256": ieso.digest(warmup)}]}


class FinalEnergyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.paths = []
        for year, days in ((2023, 365), (2024, 366), (2025, 365)):
            path = self.directory / f"PUB_Demand_{year}.csv"
            path.write_bytes(report_bytes(date(year, 1, 1), days))
            self.paths.append(path)
        self.validation = self.directory / "validation.json"
        self.procedure = self.directory / "procedure.json"
        self.output = self.directory / "final.json"
        self.models = {name: {"model_text": "synthetic-contract-fixture", "sha256": ieso.digest(b"synthetic-contract-fixture")}
                       for name in ("point", "0.05", "0.95")}
        self.environment = {"source_hashes": {f"apps/energy/{name}": name for name in final.SOURCES},
                            "dependencies": {"lightgbm": "synthetic"}, "python": "fixture", "platform": "fixture",
                            "calendar_zone": "America/Toronto", "system_zonefile_sha256": "fixture"}
        self.report = validation_fixture(self.environment, self.paths[0].read_bytes(), self.models)
        self.validation.write_text(json.dumps(self.report))

    def test_freeze_binds_models_inputs_source_runtime_and_validation(self):
        with patch.object(final, "runtime", return_value=self.environment):
            procedure = final.freeze(self.validation, self.paths)
            self.assertEqual(procedure["artifacts"]["models"], self.models)
            self.assertFalse(procedure["protocol"]["refit_or_recalibration"])
            self.assertEqual(len(final.verify(procedure)), 26304)
            for key in ("runtime", "protocol", "artifacts"):
                changed = copy.deepcopy(procedure)
                changed[key] = {}
                with self.assertRaises(ValueError):
                    final.verify(changed)
            lines = self.paths[-1].read_text().splitlines()
            fields = lines[2].split(",")
            fields[-1] = str(float(fields[-1]) + 1)
            lines[2] = ",".join(fields)
            self.paths[-1].write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(ValueError, "snapshots changed"):
                final.verify(procedure)

    def test_gap_is_rejected_without_substitution_or_final_marker(self):
        lines = self.paths[-1].read_bytes().splitlines(keepends=True)
        self.paths[-1].write_bytes(b"".join(lines[:50] + lines[51:]))
        with patch.object(final, "runtime", return_value=self.environment):
            with self.assertRaisesRegex(ValueError, "missing hour"):
                final.freeze(self.validation, self.paths)
        self.assertFalse(self.procedure.exists())
        self.assertFalse(self.procedure.with_name(self.procedure.name + ".used").exists())

    def test_validation_and_confirmation_guards(self):
        with patch.object(final, "runtime", return_value=self.environment):
            for key, value in (("unit", "kWh"), ("phase", "holdout"), ("source_hashes", {})):
                changed = copy.deepcopy(self.report)
                changed[key] = value
                with self.assertRaises(ValueError):
                    final.validation_artifacts(changed)
        with self.assertRaisesRegex(ValueError, "confirm"):
            final.run_final(self.procedure, self.output)
        self.assertFalse(self.output.exists())
        duplicate = self.directory / "duplicate.json"
        duplicate.write_text('{"schema_version": 1, "schema_version": 2}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            final.read_json(duplicate)

    def test_failed_evaluation_keeps_used_marker_and_refuses_retry(self):
        ieso.write_json(self.procedure, {"artifacts": {}})
        with patch.object(final, "verify", return_value={}), patch.object(final, "load_models", return_value={}), \
                patch.object(final, "evaluate_frozen", side_effect=ValueError("native failure")):
            with self.assertRaisesRegex(ValueError, "native failure"):
                final.run_final(self.procedure, self.output, confirmed=True, retrospective=True)
        marker = self.procedure.with_name(self.procedure.name + ".used")
        self.assertEqual(json.loads(marker.read_text())["status"], "final_started")
        self.assertFalse(self.output.exists())
        with self.assertRaisesRegex(ValueError, "used marker"):
            final.run_final(self.procedure, self.output, confirmed=True, retrospective=True)

    @unittest.skipUnless(NATIVE, "optional energy requirements not installed")
    def test_paired_block_intervals_preserve_matching_and_small_sample_limits(self):
        point = {"actual_mw": 0, "models": {"previous_day": {"prediction_mw": 10},
                  "previous_week": {"prediction_mw": 12}, "lightgbm": {"prediction_mw": 8}}}
        result = final.paired_intervals([{"points": [point] * 24}] * 300)
        for name, expected in (("previous_day", 2), ("previous_week", 4)):
            self.assertEqual(result[name]["mean_mae_reduction_mw"], expected)
            self.assertTrue(all(item["interval_95_mw"] == [expected, expected]
                                for item in result[name]["block_sensitivity"]))
        small = final.paired_intervals([{"points": [point] * 24}] * 5)
        self.assertTrue(all(item["interval_95_mw"] is None for item in small["previous_day"]["block_sensitivity"]))

    @unittest.skipUnless(NATIVE, "optional energy requirements not installed")
    def test_native_frozen_inference_has_no_refit_and_is_single_use(self):
        models = {}
        for name, quantile in (("point", None), ("0.05", 0.05), ("0.95", 0.95)):
            model = energy.fit_model([[0] * len(energy.FEATURES)] * 720, [10000] * 720, quantile=quantile)
            text = model.model_to_string()
            models[name] = {"model_text": text, "sha256": ieso.digest(text.encode())}
        self.validation.write_text(json.dumps(validation_fixture(final.runtime(), self.paths[0].read_bytes(), models)))
        ieso.write_json(self.procedure, final.freeze(self.validation, self.paths))
        with patch.object(energy, "fit_model", side_effect=AssertionError("final must not refit")):
            report = final.run_final(self.procedure, self.output, confirmed=True, retrospective=True)
        self.assertEqual(report["phase"], "holdout")
        self.assertEqual(report["counts"], {"origins": 730, "hourly_targets": 17520})
        self.assertTrue(report["final_period_consumed"])
        self.assertEqual(sum(group["calibrated"]["rows"] for group in report["breakdown"]["season"].values()), 17520)
        with self.assertRaisesRegex(ValueError, "exists"):
            final.run_final(self.procedure, self.directory / "second.json", confirmed=True, retrospective=True)


if __name__ == "__main__":
    unittest.main()
