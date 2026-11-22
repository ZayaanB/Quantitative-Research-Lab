"""Convert saved IESO Adequacy3 XML snapshots to matched hourly averages."""

import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from boosted_demand import origin_at
from demand_experiment import KEYS, timestamp
from ieso import EST, HOUR, digest, iso, market_midnight, parse_date


NS = {"i": "http://www.ieso.ca/schema"}
BASE = "https://reports-public.ieso.ca/public/Adequacy3/"


def decode(payload):
    if len(payload) > 2_000_000 or b"<!DOCTYPE" in payload.upper():
        raise ValueError("invalid or oversized Adequacy XML")
    root = ET.fromstring(payload)
    if root.attrib.get("docID") != "Adequacy3":
        raise ValueError("expected IESO Adequacy3 XML")
    issued = datetime.fromisoformat(root.findtext("i:DocHeader/i:CreatedAt", "", NS))
    if issued.tzinfo is None:
        issued = issued.replace(tzinfo=EST)
    elif issued.utcoffset() != EST.utcoffset(None):
        raise ValueError("expected fixed EST in Adequacy CreatedAt")
    origin = market_midnight(parse_date(root.findtext("i:DocBody/i:DeliveryDate", "", NS)))
    hours = {}
    for hour in root.findall("i:DocBody/i:ForecastDemand/i:OntarioDemand/i:AverageDemand/i:Demand", NS):
        ending = int(hour.findtext("i:DeliveryHour", "", NS))
        value = float(hour.findtext("i:EnergyMW", "", NS))
        if not 1 <= ending <= 24 or ending in hours or not math.isfinite(value) or value < 0:
            raise ValueError("invalid Adequacy Average Demand row")
        hours[ending] = value
    if len(hours) != 24:
        raise ValueError("require 24 Ontario Average Demand hours in each report")
    return issued, {origin + (ending - 1) * HOUR: value for ending, value in hours.items()}


def normalize(manifest):
    candidates = {}
    with manifest.open(newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if reader.fieldnames != ["origin_date_est", "available_at_utc", "source", "xml"]:
            raise ValueError("invalid Adequacy snapshot manifest header")
        for row in reader:
            if None in row or any(not value for value in row.values()):
                raise ValueError("incomplete Adequacy snapshot manifest row")
            origin = origin_at(parse_date(row["origin_date_est"]))
            available = timestamp(row["available_at_utc"])
            path = manifest.parent / row["xml"]
            payload = path.read_bytes()
            issued, hours = decode(payload)
            if not row["source"].startswith(BASE) or not issued <= available <= origin:
                raise ValueError("source must be public IESO Adequacy3 and available by origin")
            key = (origin, min(hours))
            if not any(origin <= target < origin + 24 * HOUR for target in hours):
                raise ValueError("Adequacy delivery date does not intersect its forecast window")
            previous = candidates.get(key)
            if previous is not None and previous[0] == issued:
                raise ValueError("duplicate Adequacy issuance for a delivery date")
            if previous is None or issued > previous[0]:
                candidates[key] = issued, available, f'{row["source"]} sha256={digest(payload)}', hours
    if not candidates:
        raise ValueError("empty Adequacy snapshot manifest")
    results = []
    for origin in sorted({key[0] for key in candidates}):
        for step in range(24):
            target = origin + step * HOUR
            day = market_midnight(target.astimezone(EST).date())
            if (origin, day) not in candidates:
                raise ValueError(f"missing Adequacy delivery day for {iso(target)}")
            issued, available, source, hours = candidates[origin, day]
            results.append((iso(origin), iso(target), iso(issued), iso(available), source, hours[target]))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        rows = normalize(args.manifest)
        with args.output.open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow((*KEYS, "average_demand_mw"))
            writer.writerows(rows)
    except (ValueError, OSError, csv.Error, ET.ParseError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
