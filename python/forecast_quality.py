import csv
from datetime import date
import math

from benchmark_forecasts import read_asset


def inspect_asset(asset, directory, holdout_start):
    try:
        header, rows, metadata, _ = read_asset(asset, directory)
    except (ValueError, KeyError, OSError, UnicodeError, csv.Error, StopIteration) as error:
        return None, {"symbol": asset["symbol"], "status": "error",
                      "issues": [{"severity": "error", "code": "invalid_input", "detail": str(error)}]}
    columns = next(csv.reader([header]))
    retained = [(day, next(csv.reader([line]))) for day, line in rows if day < holdout_start]
    issues = []

    def issue(severity, code, day, detail):
        issues.append({"severity": severity, "code": code, "date": day, "detail": detail})

    numeric = [name for name in ("Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits")
               if name in columns]
    for name in numeric:
        if columns.count(name) != 1:
            issue("error", "duplicate_column", None, name)
    if any(item["severity"] == "error" for item in issues):
        retained = []
    previous = None
    for day, fields in retained:
        current = date.fromisoformat(day)
        if current.weekday() >= 5:
            issue("warning", "weekend_row", day, "daily equity observation falls on a weekend")
        if previous and (current - previous).days > 4:
            issue("warning", "long_calendar_gap", day,
                  f"{(current - previous).days} calendar days since {previous.isoformat()}")
        previous = current
        values = {}
        for name in numeric:
            try:
                value = float(fields[columns.index(name)])
                if not math.isfinite(value) or value < 0 or (name in ("Open", "High", "Low", "Close") and value == 0):
                    raise ValueError("invalid domain")
                values[name] = value
            except ValueError:
                issue("error", "invalid_numeric_column", day, name)
        if all(name in values for name in ("Open", "High", "Low", "Close")):
            tolerance = 1e-9 * max(values[name] for name in ("Open", "High", "Low", "Close"))
            if (values["Low"] > min(values["Open"], values["Close"]) + tolerance or
                    values["High"] < max(values["Open"], values["Close"]) - tolerance or
                    values["High"] < values["Low"]):
                issue("error", "inconsistent_ohlc", day, "reported OHLC bounds disagree")
        if "Volume" in values:
            if values["Volume"] != int(values["Volume"]):
                issue("error", "fractional_volume", day, "daily share volume is not an integer")
            elif values["Volume"] == 0:
                issue("warning", "zero_volume", day, "check source and session availability")
    report = {"symbol": asset["symbol"], "status": "error" if any(
                  item["severity"] == "error" for item in issues) else "ok",
              "source_rows": len(rows), "source_first_date": rows[0][0], "source_last_date": rows[-1][0],
              "inspected_rows": len(retained), "scope": "diagnostics only before holdout_start",
              "columns": columns, "ohlcv_checked": all(name in columns for name in
                  ("Open", "High", "Low", "Close", "Volume")),
              "csv_sha256": asset["csv_sha256"], "metadata_sha256": asset["metadata_sha256"],
              "provenance": metadata,
              "issues": issues,
              "limitations": ["Whole-file price/date validation and hashes still include reserved rows.",
                              "Weekend/gap checks are heuristic, not an exchange-session calendar.",
                              "No rows are filled, dropped or repaired; adjustment semantics are declared."]}
    if not retained and report["status"] == "ok":
        report["status"] = "error"
        issue("error", "empty_validation_history", None, "no history before holdout_start")
    return (header, rows, metadata), report
