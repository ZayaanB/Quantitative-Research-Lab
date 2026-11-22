import csv
from datetime import date, timedelta
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "energy"))
from energy_fixtures import report_bytes

import demand_experiment as experiment
import adequacy_input
from boosted_demand import origin_at
from ieso import HOUR, iso


class DemandExperimentTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def write_external(self, filename, start, end, column, *, late=False):
        path = self.directory / filename
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow((*experiment.KEYS, column))
            for day in range((end - start).days):
                origin = origin_at(start + timedelta(days=day))
                available = origin + HOUR if late else origin - HOUR
                for hour in range(24):
                    value = 12 + hour / 2 if column == "toronto_c" else 15_000 + hour
                    writer.writerow((iso(origin), iso(origin + hour * HOUR), iso(origin - 2 * HOUR),
                                     iso(available), "synthetic forecast", value))
        return path

    def test_cross_year_holiday_and_late_inputs(self):
        self.assertEqual(experiment.holiday_features(date(2022, 12, 31)), [0, 1, 0])
        self.assertIn(date(2023, 4, 7), experiment.holidays(2023))
        day = date(2020, 4, 1)
        path = self.write_external("late.csv", day, day + timedelta(days=1), "toronto_c", late=True)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            experiment.read_forecasts(path)

    def test_weather_models_and_operator_use_identical_hours(self):
        train, cal, start, end = map(date.fromisoformat, ("2020-01-15", "2020-03-01", "2020-04-02", "2020-04-06"))
        demand = self.directory / "demand.csv"
        demand.write_bytes(report_bytes(date(2020, 1, 1), 110))
        weather = self.write_external("weather.csv", train, end, "toronto_c")
        operator = self.write_external("operator.csv", cal, end, "average_demand_mw")
        report = experiment.evaluate([demand], train, cal, start, end, 22_000, weather, operator)
        self.assertEqual(report["hourly_targets"], 96)
        self.assertEqual(report["calibration_windows"], 31)
        self.assertEqual(len(report["forecast_sources"]["operator_calibration_then_scoring"]), 35)
        for model in ("lightgbm", "lightgbm_holidays", "lightgbm_weather", "spline", "blend", "ieso_average_demand"):
            result = report["metrics"][model]["all"]
            self.assertEqual(result["windows"], 4)
            self.assertGreaterEqual(result["peak_90_mean_width_mw"], 0)
            self.assertLessEqual(result["peak_exceedance_brier"], 1)
        self.assertEqual(report["metrics"]["spline"]["forecast_hot"], {"windows": 0})
        incomplete = self.write_external("incomplete.csv", cal, start, "average_demand_mw")
        with self.assertRaisesRegex(ValueError, "no scoring rows were trimmed"):
            experiment.evaluate([demand], train, cal, start, end, 22_000, weather, incomplete)

    def test_adequacy_uses_average_and_preserves_two_market_dates(self):
        day = date(2026, 3, 8)
        origin = origin_at(day)
        manifest = self.directory / "manifest.csv"
        with manifest.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("origin_date_est", "available_at_utc", "source", "xml"))
            for offset in (0, 1):
                delivery = day + timedelta(days=offset)
                name = f"{delivery}.xml"
                hours = "".join(f"<Demand><DeliveryHour>{hour}</DeliveryHour><EnergyMW>{15_000 + hour}</EnergyMW>"
                                "</Demand>" for hour in range(1, 25))
                xml = (f'<Document docID="Adequacy3" xmlns="{adequacy_input.NS["i"]}"><DocHeader>'
                       f"<CreatedAt>{day}T07:{20 + offset}:00</CreatedAt></DocHeader>"
                       f"<DocBody><DeliveryDate>{delivery}</DeliveryDate><ForecastDemand><OntarioDemand>"
                       f"<ForecastOntDemand/><AverageDemand>{hours}</AverageDemand></OntarioDemand>"
                       "</ForecastDemand></DocBody></Document>")
                (self.directory / name).write_text(xml)
                writer.writerow((day, iso(origin), adequacy_input.BASE + name, name))
        normalized = adequacy_input.normalize(manifest)
        self.assertEqual(len(normalized), 24)
        self.assertEqual([normalized[i][-1] for i in (0, 15, 16, 23)], [15009, 15024, 15001, 15008])
        self.assertNotEqual(normalized[15][2], normalized[16][2])
        self.assertEqual(normalized[16][1], iso(origin + 16 * HOUR))
        self.assertIn("sha256=", normalized[16][4])


if __name__ == "__main__":
    unittest.main()
