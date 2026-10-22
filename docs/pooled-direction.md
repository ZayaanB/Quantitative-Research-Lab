# Pooled stock direction

Pool the [direction classifiers](direction-forecast.md) across all stocks in a
frozen manifest. Separate training is still the default. Pricing, Greeks and
price regressors are unchanged; neither mode outputs price targets or signals.

Use the stock/context manifests from the [feature guide](stock-features.md).
Run both modes on the same inputs, using fresh filenames:

```bash
.venv/bin/python -m pip install -r python/requirements-direction.txt
.venv/bin/python python/direction_forecasts.py \
  --manifest Notes-dont-commit/benchmark.json \
  --context-manifest Notes-dont-commit/stock-context.json \
  --training-mode separate --output Notes-dont-commit/separate-validation.json

.venv/bin/python python/direction_forecasts.py \
  --manifest Notes-dont-commit/benchmark.json \
  --context-manifest Notes-dont-commit/stock-context.json \
  --training-mode pooled --output Notes-dont-commit/pooled-validation.json

.venv/bin/python tests/test_direction_forecasts.py
```

## What changes

There is one shared logistic fit and one shared LightGBM fit per horizon/year,
not one per stock. Each asset contributes its past training rows using the same
fold-start **date**, not an index copied from another asset. Labels must finish
strictly before that date. Every requested stock needs 40 matured labels;
insufficient history stops the run instead of silently removing a stock.

Training rows are sorted by origin date then symbol and weighted equally.
No ticker identity is added to the 28 features. The logistic scaler sees only
the pooled training matrix; prediction rows never enter its fit. Parameters,
yearly refits, thresholds, origin grids and per-stock historical-up baselines
are unchanged from separate mode. No tuning, calibration or winner selection
is added. Pooled single-class data is explicitly reported as a constant fit.

`shared_fits` records each fit once, with per-asset counts/cutoffs and scaler
statistics. Group folds and prediction rows reference its `fit_id`. Existing
per-point training dates/indices describe that stock's labels;
`pooled_last_training_target` is the latest target used across all stocks.
Input/source hashes, warnings, versions, metrics and runtime remain in the report.

Compare identical asset/horizon groups and origin/target prices, not just headline
percentages. Existing within-stock bootstrap intervals are sensitivity checks;
shared training and correlated stocks do not make an independent larger sample.
There is no pooled cross-stock significance claim or automatic model promotion.

## Broader inputs

For Day 4 we fixed JPM/JNJ/XOM/PG/WMT/CAT/KO before scoring, alongside the saved
AAPL/MSFT/GOOGL inputs. All seven downloads succeeded for 2010–2023. Their hashes
and retrieval metadata are kept in a new ignored snapshot; old files were not
overwritten. The original SPY/QQQ/XLK context snapshot was reused, and 2024 onward
was not scored. Freeze your own list and hashes before comparing modes.

This is a small present-day survivor sample, not a historical index universe or
a point-in-time dataset. Earlier validation labels can enter later yearly fits
only after they mature. XLK is still a research proxy, not a sector assignment
for every stock. A broader sample is a new experiment, not directly comparable
with a three-stock headline percentage.

## October 21 validation

2018–2023, all ten stocks, identical inputs and grids in both modes:

| Horizon | Samples | Separate LightGBM | Pooled LightGBM | Always-up |
| --- | --- | --- | --- | --- |
| 1 session | 15,080 | 50.4% | 51.4% | 52.5% |
| 5 sessions | 3,010 | 51.0% | 53.2% | 55.9% |
| 20 sessions | 750 | 52.1% | 50.4% | 60.7% |

Pooled logistic hit rates were 49.4%, 51.3% and 48.4%, respectively. Both pooled
models had worse Brier scores than historical up frequency at every horizon.
No stock/horizon group showed an accuracy win over always-up across all tested
bootstrap blocks. These descriptive results do not demonstrate an edge.
The default per-stock mode exactly reproduced the saved Day 3 results on the
original stocks. Both new reports and all data remain in the ignored snapshot.
