"""Loopback-only dashboard for the existing quant research tools."""
import argparse
import csv
import io
import json
import math
import random
import subprocess
import sys
import time
from datetime import date, timedelta
from functools import cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STOCK_REPORT = "Notes-dont-commit/pooled-universe-20261021/calibrated-validation-phase2-day5-20261021.json"
ENERGY_REPORT = "Notes-dont-commit/energy/uncertainty-replay-20261117.json"


def settings(query, key, default, cast, low, high):
    value = cast(query.get(key, [default])[0])
    if not low <= value <= high:
        raise ValueError(f"{key} must be between {low} and {high}")
    return str(value)


def fields(output):
    return dict(line.split(":", 1) for line in output.splitlines() if ":" in line)


def extract(output, key):
    return float(output[key].strip().split()[0].rstrip("%"))


def interval(output, key):
    return [float(part) for part in output[key].strip().strip("[]").split(",")]


def native(binary, *args):
    if not binary.is_file():
        raise ValueError("mcprice is missing. Build it first or launch with --binary.")
    result = subprocess.run([str(binary), *args], capture_output=True, text=True,
                            timeout=90, cwd=ROOT)
    if result.returncode:
        raise ValueError(result.stderr.strip() or "mcprice exited with an error")
    return result.stdout


def pricing(binary, query):
    kind = query.get("type", ["call"])[0]
    if kind not in ("call", "put", "asian-call"):
        raise ValueError("Choose a supported option type.")
    asian = kind == "asian-call"
    spot = settings(query, "spot", 100, float, 1, 10000)
    strike = settings(query, "strike", 100, float, 1, 10000)
    vol = settings(query, "vol", .2, float, .01, 2)
    rate = settings(query, "rate", .05, float, -.25, .5)
    maturity = settings(query, "maturity", 1, float, .01, 10)
    paths = int(settings(query, "paths", 1_000_000, int, 1000, 10_000_000))
    threads = settings(query, "threads", 2, int, 1, 16)
    steps = settings(query, "steps", 64, int, 2, 252) if asian else "1"
    if paths % 2:
        raise ValueError("Antithetic sampling requires an even path count.")
    if asian and paths * int(steps) > 260_000_000:
        raise ValueError("Choose fewer paths or monitoring steps for this demo.")
    args = ("--type", kind, "--spot", spot, "--strike", strike, "--volatility", vol,
            "--rate", rate, "--maturity", maturity, "--threads", threads,
            "--steps", steps, "--seed", "42", "--antithetic")
    if asian:
        rng = random.Random(42)
        dt = float(maturity) / int(steps)
        samples = []
        for _ in range(8):
            path = [float(spot)]
            for _ in range(int(steps)):
                path.append(path[-1] * math.exp((float(rate) - float(vol)**2 / 2) * dt
                            + float(vol) * math.sqrt(dt) * rng.gauss(0, 1)))
            avg = sum(path[1:]) / int(steps)
            samples.append({"path": path, "average": avg, "payoff": max(0, avg - float(strike))})
        yield {"samples": samples}

    started = time.perf_counter()
    # Each checkpoint is a complete native run with a larger path budget.
    for fraction in (.01, .03, .07, .15, .30, .55, 1):
        count = max(4, int(paths * fraction) // 2 * 2)
        result = fields(native(binary, "price", *args, "--paths", str(count)))
        yield {"paths": count, "estimate": extract(result, "Monte Carlo estimate"),
               "reference": extract(result, "Analytical price") if not asian else None,
               "interval": interval(result, "95% confidence interval"),
               "engine": extract(result, "Runtime"), "wall": time.perf_counter() - started,
               "throughput": extract(result, "Throughput")}
    if not asian:
        result = fields(native(binary, "greeks", *args, "--paths", str(paths)))
        yield {"greeks": {key: extract(result, label) for key, label in
                          (("Delta", "Delta"), ("Gamma", "Gamma"), ("Vega", "Vega (per 1 vol point)"),
                           ("Greeks runtime", "Greek runtime (5 runs)"))},
               "wall": time.perf_counter() - started}


def scenarios(binary, query, history):
    horizon = int(settings(query, "horizon", 3, int, 1, 20))
    lookback = settings(query, "lookback", 10, int, 2, 252)
    args = ("--csv", str(history), "--lookback-days", lookback)
    started = time.perf_counter()
    fan = []
    for lead in range(1, horizon + 1):
        output = fields(native(binary, "forecast", *args, "--horizon-days", str(lead)))
        low, high = interval(output, "95% model interval")
        fan.append({"step": lead, "median": extract(output, "Median price"),
                    "low": low, "high": high})
    spot = extract(output, "Current selected price")
    fan.insert(0, {"step": 0, "median": spot, "low": spot, "high": spot})
    # Allow a long forecast on the short fixture even when no backtest origins fit.
    with history.open(newline="") as source:
        count = sum(1 for _ in csv.DictReader(source))
    rows = []
    if count > int(lookback) + horizon:
        report = native(binary, "backtest", *args, "--horizon-days", str(horizon), "--format", "csv")
        keys = ("actual_price", "forecast_price", "latest_price_forecast", "lower_95", "upper_95")
        rows = [{"date": row["target_date"], **{key: float(row[key]) for key in keys}}
                for row in csv.DictReader(io.StringIO(report))]
    return {"fan": fan, "rows": rows, "wall": time.perf_counter() - started,
            "probability": extract(output, "Probability above today"),
            "volatility": extract(output, "Annualized volatility"),
            "drift": extract(output, "Annualized drift"), "source": history.name,
            "synthetic": history.resolve() == (ROOT / "examples/sample_prices.csv").resolve()}


def synthetic_stock():
    sys.path.insert(0, str(ROOT / "python"))
    from ridge_forecasts import forecast

    rng = random.Random(11)
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(265)]
    dates = [day.isoformat() for day in dates if day.weekday() < 5]
    prices = [100.]
    for index in range(1, len(dates)):
        prices.append(prices[-1] * math.exp(.002 * math.sin(index / 10) + rng.gauss(0, .009)))
    start = time.perf_counter()
    points = []
    for index in range(65, len(prices) - 1):
        prediction = forecast(prices[:index + 1], 1, 60)["forecast_price"]
        points.append({"date": dates[index + 1], "actual": prices[index + 1],
                       "prediction": prediction, "baseline": prices[index]})
    return {"source": "Synthetic stock history", "runtime": time.perf_counter() - start,
            "mode": "price", "series": [{"label": "Ridge · 1 session", "points": points}]}


@cache
def stock_report(path):
    if path is None:
        return synthetic_stock()
    report = json.loads(path.read_text())
    series = []
    for result in report["results"]:
        points = [{"date": row["target_date"], "actual": row["actual_up"],
                   "prediction": row["calibrated"], "baseline": row["historical_up"],
                   "raw": row["selected_raw"]} for row in result["points"]]
        if points:
            series.append({"label": f'{result["symbol"]} · {result["horizon_days"]} sessions',
                           "points": points})
    return {"source": path.name, "runtime": report.get("runtime_seconds"),
            "mode": "probability", "series": series}


def synthetic_energy():
    sys.path.insert(0, str(ROOT / "apps/energy"))
    from baselines import backtest
    from ieso import HOUR, Observation, market_midnight

    day = date(2023, 1, 1)
    begin = market_midnight(day)
    rng = random.Random(12)
    observations = {}
    for index in range(24 * 54):
        stamp = begin + index * HOUR
        hour = index % 24
        load = (16000 + 1250 * math.cos((hour - 17) * math.tau / 24)
                + 500 * math.cos(index * math.tau / 168) + rng.gauss(0, 170))
        observations[stamp] = Observation(stamp, load)
    start = time.perf_counter()
    report = backtest(observations, day + timedelta(days=16), day + timedelta(days=53))
    return report["windows"], time.perf_counter() - start


@cache
def energy_report(path):
    if path is None:
        windows, runtime = synthetic_energy()
    else:
        report = json.loads(path.read_text())
        windows, runtime = report["windows"], report.get("runtime_seconds")

    def point(row):
        models = row["models"]
        model = models["lightgbm"] if "lightgbm" in models else models["previous_week"]
        bounds = row.get("ranges", {}).get("calibrated", [None, None])
        return {"hour": row["lead_hour"], "actual": row["actual_mw"],
                "prediction": model["prediction_mw"], "baseline": models["previous_day"]["prediction_mw"],
                "low": bounds[0], "high": bounds[1]}

    return {"source": path.name if path else "Synthetic Ontario demand profile",
            "runtime": runtime, "model": "LightGBM" if path else "Weekly baseline",
            "days": [{"date": window["origin_est"][:10], "points": [point(row) for row in window["points"]]}
                     for window in windows]}


class Handler(BaseHTTPRequestHandler):
    def send(self, body, mime, status=200):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def reply(self, data, status=200):
        self.send(json.dumps(data, allow_nan=False).encode(), "application/json", status)

    def stream_pricing(self, query):
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            for update in pricing(self.server.options.binary, query):
                self.wfile.write((json.dumps(update, allow_nan=False) + "\n").encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
        except (ValueError, OSError, subprocess.TimeoutExpired) as error:
            self.wfile.write((json.dumps({"error": str(error)}) + "\n").encode())

    def do_GET(self):
        if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
            self.reply({"error": "Use the printed loopback address."}, 403)
            return
        url = urlsplit(self.path)
        assets = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
                  "/style.css": ("style.css", "text/css")}
        if url.path in assets:
            filename, mime = assets[url.path]
            self.send((HERE / filename).read_bytes(), mime)
            return
        options = self.server.options
        query = parse_qs(url.query)
        try:
            if url.path == "/api/price":
                self.stream_pricing(query)
            elif url.path == "/api/scenarios":
                self.reply(scenarios(options.binary, query, options.history))
            elif url.path == "/api/stock":
                self.reply(stock_report(options.stock_report))
            elif url.path == "/api/energy":
                self.reply(energy_report(options.energy_report))
            else:
                self.reply({"error": "Unknown dashboard route."}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (ValueError, KeyError, TypeError, OSError, subprocess.TimeoutExpired) as error:
            self.reply({"error": str(error)}, 400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/mcprice")
    parser.add_argument("--history", type=Path, default=ROOT / "examples/sample_prices.csv")
    parser.add_argument("--stock-report", type=Path)
    parser.add_argument("--energy-report", type=Path)
    parser.add_argument("--synthetic", action="store_true", help="Ignore auto-detected private reports.")
    options = parser.parse_args()
    options.binary, options.history = options.binary.resolve(), options.history.resolve()
    for key, default in (("stock_report", STOCK_REPORT), ("energy_report", ENERGY_REPORT)):
        explicit = getattr(options, key)
        candidate = ROOT / default
        setattr(options, key, explicit.resolve() if explicit else
                candidate if not options.synthetic and candidate.is_file() else None)
    with ThreadingHTTPServer(("127.0.0.1", options.port), Handler) as server:
        server.options = options
        print(f"http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
