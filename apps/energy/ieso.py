import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import re
import subprocess
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zoneinfo import TZPATH, ZoneInfo, ZoneInfoNotFoundError


UTC = timezone.utc
EST = timezone(timedelta(hours=-5), "EST")
HOUR = timedelta(hours=1)
HEADER = ["Date", "Hour", "Market Demand", "Ontario Demand"]
BASE_URL = "https://reports-public.ieso.ca/public/Demand/"
TERMS_URL = "https://www.ieso.ca/Terms-of-Use"
NOTICE = "This information is subject to the Terms of Use set out in the IESO's website (www.ieso.ca)"
MAX_BYTES = 2_000_000
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Observation:
    start: datetime
    demand_mw: float


def iso(value):
    return value.astimezone(UTC).isoformat()


def market_midnight(day):
    return datetime.combine(day, time(), EST).astimezone(UTC)


def assumed_available_at(start):
    day_after = start.astimezone(EST).date() + timedelta(days=1)
    return market_midnight(day_after) + 8 * HOUR


def parse_date(value):
    parsed = date.fromisoformat(value)
    if value != parsed.isoformat():
        raise ValueError("dates must use YYYY-MM-DD")
    return parsed


def parse_origin(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("origin needs an explicit UTC offset")
    parsed = parsed.astimezone(UTC)
    if parsed.minute or parsed.second or parsed.microsecond:
        raise ValueError("origin must be on an exact UTC hour")
    return parsed


def validate_sequence(observations):
    if not observations:
        raise ValueError("no hourly demand rows")
    for previous, current in zip(observations, observations[1:]):
        if current.start <= previous.start:
            raise ValueError(f"duplicate or out-of-order hour: {iso(current.start)}")
        if current.start != previous.start + HOUR:
            raise ValueError(f"missing hour after {iso(previous.start)}; no filling or dropping applied")


def parse_report(payload, expected_year=None):
    if len(payload) > MAX_BYTES:
        raise ValueError("demand CSV exceeds the size limit")
    reader = csv.reader(io.StringIO(payload.decode("utf-8-sig"), newline=""), strict=True)
    preamble = []
    for row in reader:
        if row and row[0].startswith("\\\\"):
            preamble.append(",".join(row).rstrip(","))
        elif row == HEADER:
            break
        else:
            raise ValueError("expected IESO Date,Hour,Market Demand,Ontario Demand header")
    else:
        raise ValueError("missing IESO demand header")
    observations = []
    for row in reader:
        if len(row) != len(HEADER):
            raise ValueError(f"invalid demand row at CSV line {reader.line_num}")
        day = parse_date(row[0])
        if expected_year is not None and day.year != expected_year:
            raise ValueError("report contains a date outside the requested year")
        if not re.fullmatch(r"\d{1,2}", row[1]) or not 1 <= int(row[1]) <= 24:
            raise ValueError("Hour must be an hour-ending integer from 1 to 24")
        values = [float(cell) for cell in row[2:]]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"demand must be finite nonnegative MW on {day}")
        start = market_midnight(day) + (int(row[1]) - 1) * HOUR
        observations.append(Observation(start, values[1]))
    validate_sequence(observations)
    if expected_year is not None:
        first = market_midnight(date(expected_year, 1, 1))
        end = market_midnight(date(expected_year + 1, 1, 1))
        if observations[0].start != first or observations[-1].start + HOUR != end:
            raise ValueError("download must contain a complete calendar year; no partial snapshots")
    return observations, preamble


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def provenance():
    paths = [Path(__file__), Path(__file__).with_name("baselines.py")]
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                  capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    zone_digest = None
    for base in TZPATH:
        zone_file = Path(base) / "America/Toronto"
        if zone_file.is_file():
            zone_digest = digest(zone_file.read_bytes())
            break
    return {"source_revision": revision,
            "source_hashes": {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in paths},
            "python": platform.python_version(), "platform": platform.platform(),
            "dependencies": "Python standard library only",
            "calendar_zone": "America/Toronto", "system_zonefile_sha256": zone_digest}


class SameSourceRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        if new_url != request.full_url:
            raise ValueError("unexpected IESO redirect; stop rather than change data source")
        return super().redirect_request(request, fp, code, message, headers, new_url)


def fetch_report(url):
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != "reports-public.ieso.ca" or
            not re.fullmatch(r"/public/Demand/PUB_Demand_\d{4}\.csv", parsed.path) or
            parsed.query or parsed.fragment):
        raise ValueError("only the public IESO annual demand endpoint is supported")
    request = Request(url, headers={"User-Agent": "MC-Options-Engine/energy-research", "Accept": "text/csv"})
    with build_opener(SameSourceRedirect()).open(request, timeout=30) as response:
        payload = response.read(MAX_BYTES + 1)
        headers = {key: response.headers.get(key) for key in ("Last-Modified", "ETag", "Content-Type")}
    if len(payload) > MAX_BYTES:
        raise ValueError("demand CSV exceeds the size limit")
    return payload, headers


def write_json(path, content):
    serialized = json.dumps(content, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)


def download(year, output, fetch=fetch_report):
    if type(year) is not int or not 2003 <= year < datetime.now(UTC).year:
        raise ValueError("choose a completed calendar year from 2003 onward")
    output = Path(output)
    if output.exists():
        raise ValueError("snapshot directory already exists; choose a new name")
    name = f"PUB_Demand_{year}.csv"
    url = BASE_URL + name
    payload, headers = fetch(url)
    retrieved = datetime.now(UTC).isoformat()
    observations, preamble = parse_report(payload, expected_year=year)
    metadata = {"schema_version": 1, "provider": "IESO", "url": url, "retrieved_at_utc": retrieved,
                "raw_file": name, "raw_sha256": digest(payload), "raw_bytes": len(payload),
                "http_headers": headers, "preamble": preamble, "rows": len(observations),
                "series": "Ontario Demand", "unit": "MW", "timezone": "fixed EST (UTC-05:00), hour ending 1..24",
                "first_start_utc": iso(observations[0].start), "last_end_utc": iso(observations[-1].start + HOUR),
                "point_in_time": False, "adjustment": "annual report as retrieved; historical revisions not reversed",
                "terms_url": TERMS_URL, "notice": NOTICE, **provenance()}
    output.mkdir(parents=True, exist_ok=False)
    with (output / name).open("xb") as stream:
        stream.write(payload)
    write_json(output / "snapshot.json", metadata)
    return metadata


def load_reports(paths):
    observations, inputs = [], []
    if not paths or len({Path(path).resolve() for path in paths}) != len(paths):
        raise ValueError("provide distinct CSV paths in chronological order")
    for path in map(Path, paths):
        if path.stat().st_size > MAX_BYTES:
            raise ValueError("demand CSV exceeds the size limit")
        payload = path.read_bytes()
        rows, preamble = parse_report(payload)
        entry = {"path": str(path.resolve()), "sha256": digest(payload), "bytes": len(payload),
                 "rows": len(rows), "preamble": preamble, "snapshot": None}
        metadata_path = path.with_name("snapshot.json")
        if metadata_path.exists():
            metadata_payload = metadata_path.read_bytes()
            metadata = json.loads(metadata_payload)
            if (not isinstance(metadata, dict) or metadata.get("raw_file") != path.name or
                    metadata.get("raw_sha256") != entry["sha256"]):
                raise ValueError("snapshot metadata does not match raw demand file")
            entry["snapshot"] = {"path": str(metadata_path.resolve()), "sha256": digest(metadata_payload),
                                 "metadata": metadata}
        observations.extend(rows)
        inputs.append(entry)
    validate_sequence(observations)
    return {row.start: row for row in observations}, inputs


def calendar_fields(start):
    market = start.astimezone(EST)
    try:
        local = start.astimezone(ZoneInfo("America/Toronto"))
    except ZoneInfoNotFoundError as error:
        raise ValueError("America/Toronto time-zone data is missing; install the OS zone database or tzdata in a venv") from error
    return {"start_utc": iso(start), "end_utc": iso(start + HOUR),
            "market_date": market.date().isoformat(), "hour_ending_est": market.hour + 1,
            "local_start": local.isoformat(), "local_fold": local.fold}


def main():
    parser = argparse.ArgumentParser(description="Download one complete IESO Ontario hourly-demand year (MW).")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new private snapshot directory")
    args = parser.parse_args()
    try:
        metadata = download(args.year, args.output)
    except (ValueError, OSError, csv.Error) as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps({"rows": metadata["rows"], "sha256": metadata["raw_sha256"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
