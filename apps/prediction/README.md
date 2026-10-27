# Stock prediction experiments

Optional Python experiments: ridge, a ridge/latest-price ensemble, ARIMA, ETS
and LightGBM regression, plus separate logistic/LightGBM direction classifiers.
These benchmark prices or direction at 1/5/20-session horizons. They
do not replace the C++ forecast command or prove a trading edge.

Run from the repository root after the [normal build](../../README.md):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r python/requirements-forecast.txt -r python/requirements-ml.txt
.venv/bin/python python/boosted_forecasts.py --help
```

To obtain daily histories with the optional downloader:

```bash
.venv/bin/python -m pip install -r python/requirements-data.txt
.venv/bin/python python/download_stock_history.py \
  --start 2010-01-01 --end 2026-10-17 \
  --output Notes-dont-commit/market-data-20261017
```

Use a new snapshot folder for each run. The end date is exclusive. This fetches
AAPL/MSFT/GOOGL and SPY/QQQ/XLK, validates prices/OHLCV, and records provenance
and hashes. It stops on failed data, never overwrites snapshots, and removes its
temporary cookie cache. No credentials or paid subscription are needed, but
provider access can be rate-limited. Yahoo data is for permitted personal use;
check [the downloader's data-use notice](https://github.com/ranaroussi/yfinance).
Reported adjustments are not independently verified or point-in-time data.

Prepare at least two daily stock histories and metadata files. Copy
[`forecast_benchmark.template.json`](../../examples/forecast_benchmark.template.json)
into `Notes-dont-commit/`, then replace dates, file paths and hashes using the
[benchmark guide](../../docs/forecast-benchmark.md). The template is not a
ready-to-run real dataset. Keep a genuinely untouched final period.

With that manifest prepared, use new output filenames:

```bash
.venv/bin/python python/boosted_forecasts.py \
  --manifest Notes-dont-commit/benchmark.json \
  --output Notes-dont-commit/boosted-validation.json

.venv/bin/python python/classical_forecasts.py \
  --manifest Notes-dont-commit/benchmark.json \
  --output Notes-dont-commit/classical-validation.json

.venv/bin/python tests/test_boosted_forecasts.py
.venv/bin/python tests/test_classical_forecasts.py
```

Compare those saved predictions with GBM using the
[matched-date runner](../../docs/matched-forecast.md). It checks data quality and
requires identical model dates rather than silently discarding unmatched rows.

The current LightGBM model still uses only historical price-derived features.
A separate [feature exporter](../../docs/stock-features.md) now prepares OHLCV
and SPY/QQQ/XLK inputs for direction experiments; it does not change that model.
The model predicts log returns and maps
them back to prices; those outputs are not expected arithmetic prices or
calibrated probabilities. It runs on CPU; no GPU is required.

The [direction classifiers](../../docs/direction-forecast.md) use the expanded
features with training-only scaling and fixed yearly fits. Their output is
up/not-up probability, not a price forecast. Install `requirements-direction.txt`
and run that guide's command; existing regression commands stay unchanged.
Use `--training-mode pooled` for [shared cross-stock training](../../docs/pooled-direction.md);
per-stock training remains the default, with identical evaluation dates.
The [selection/calibration runner](../../docs/calibrated-direction.md) adds a
small fixed search and past-only sigmoid adjustment without changing either mode.
The [small CPU CNN](../../docs/neural-direction.md) is a separate fixed-seed
sequence experiment, with no pretrained weights or GPU requirement.

Our small 2015–2017 Apple diagnostic did not beat latest price. More features
are hypotheses to test, not promised improvements. Do not tune on the final
test or use future observations in features. Current fundamentals cannot be
backfilled as though they were available years ago.

Details: [LightGBM](../../docs/boosted-forecast.md),
[ridge](../../docs/ridge-forecast.md), [ensemble](../../docs/ensemble-forecast.md),
[classical models](../../docs/classical-forecast.md),
[empirical ranges](../../docs/residual-forecast.md),
[final-test procedure](../../docs/final-forecast.md).
Implementation stays in [`python/`](../../python/).
