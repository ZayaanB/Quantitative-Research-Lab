# Stock features

Export 28 daily features without fitting or scoring a model. Existing forecasts
and the C++ pricing engine are unchanged. No optional packages or GPU are needed.

Prepare a stock [benchmark manifest](forecast-benchmark.md) and a separate
[context manifest](../examples/stock_context.template.json) in `Notes-dont-commit/`.
The context manifest uses the same settings, including dates and bootstrap, but
lists exactly SPY, QQQ and XLK. Replace template hashes with your downloaded
CSV/metadata hashes. Paths are relative to each manifest's folder.

Run from the repository root, using a fresh output name:

```bash
python3 python/stock_features.py \
  --manifest Notes-dont-commit/benchmark.json \
  --context-manifest Notes-dont-commit/stock-context.json \
  --output Notes-dont-commit/stock-features.json

ctest --test-dir build -R stock_feature_pipeline --output-on-failure
```

## What goes in

Stock CSVs need `Open`, `High`, `Low`, `Close`, `Volume` and the price column
declared in metadata. Context CSVs only need dates and their selected price.
All stock sessions must match context sessions within the stock's date range.
Missing or extra sessions stop the export; nothing is filled or intersected.
Earlier context history is allowed. The [Day 1 checks](matched-forecast.md)
still apply; invalid inputs block export and save diagnostics.

| Features | Definition at origin i |
| --- | --- |
| 8 existing price features | Selected-price log returns over 1/2/5/10/20 sessions, population standard deviation of 5/20 daily log returns, log price minus its 20-session mean. |
| Intraday range and return | log(High) − log(Low); log(Close) − log(Open). Both use the same row's reported OHLC, not mixed adjusted/raw price levels. |
| Close location | 2 × (Close − Low)/(High − Low) − 1; zero for a flat bar. |
| Volume changes | log1p(Volume_i) minus the previous session or the mean log1p volume over the last 20 sessions. Zero volume stays valid and raises a data warning. |
| 5 per context | Selected-price log returns over 1/5/20 sessions and population standard deviation of 5/20 daily log returns for each ETF. Volatility is not annualized. |

Every rolling window ends at the origin and includes that session. The first
20 stock sessions are explicit warmup. Output includes earlier training history
and validation history, but no rows on or after `holdout_start`, no targets and
no accuracy metrics. `feature_names` gives the order of each row's values.
Hashes, settings, data checks, source revision and runtime accompany the rows.

These are **after-close** inputs: use them only after all daily observations
are published. Daily files do not prove actual publication time. Yahoo's
retrospective adjustments are not point-in-time data; reported volume may have
split effects. SPY/QQQ/XLK are fixed research proxies, not a sector mapping for
every stock. Whole-file hashes and basic price/date checks still read reserved
rows, but features and extra OHLCV diagnostics do not use them.

The [direction experiments](direction-forecast.md) use these inputs. More features
are not evidence of better predictions; that requires matched out-of-sample tests.
