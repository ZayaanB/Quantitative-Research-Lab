# Stock prediction experiments

Optional Python models test price/return forecasts and separate up/not-up
probabilities at 1/5/20-session horizons. Code stays in `python/`; pricing
and the C++ GBM command remain separate.

## Setup and data

Run from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r python/requirements-data.txt
.venv/bin/python python/download_stock_history.py \
  --start 2010-01-01 --end 2024-01-01 \
  --output Notes-dont-commit/stock-history
```

Choose dates for your experiment and a new snapshot path. The end date is
exclusive. Defaults fetch AAPL/MSFT/GOOGL plus SPY/QQQ/XLK, validate OHLCV and
record metadata/hashes. Provider failures stop the run; files are not overwritten.
Reported adjustments are not independently verified or point-in-time data.
Observe the provider's [data-use terms](https://github.com/ranaroussi/yfinance).

Prepare at least two stock histories. Copy the
[manifest template](../../examples/forecast_benchmark.template.json) into
`Notes-dont-commit/`, replace paths/dates/hashes, and follow the
[benchmark guide](../../docs/forecast-benchmark.md). Reserve a genuinely untouched
final period and compare [matched prediction rows](../../docs/matched-forecast.md).

## Models and commands

| Workflow | Guide |
| --- | --- |
| Ridge, fixed ensemble, ARIMA/ETS, LightGBM regression and residual ranges | [Price models](../../docs/models.md#running-price-experiments) |
| OHLCV and SPY/QQQ/XLK feature export | [Features](../../docs/stock-features.md) |
| Logistic/LightGBM direction classifiers | [Direction](../../docs/direction-forecast.md) |
| Shared cross-stock training | [Pooling](../../docs/pooled-direction.md) |
| Past-only selection and sigmoid calibration | [Calibration](../../docs/calibrated-direction.md) |
| Optional CPU CNN without pretrained weights | [CNN](../../docs/neural-direction.md) |
| Frozen final evaluation | [Final procedure](../../docs/final-forecast.md) |

Each guide lists its dependencies and tests. Price regression uses its own
price-derived inputs; expanded OHLCV/context features feed direction experiments.
See [current evidence](../../docs/roadmap.md) before planning another evaluation.
