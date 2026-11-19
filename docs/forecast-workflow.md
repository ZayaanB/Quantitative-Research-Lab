# Stock forecast workflow

The C++ CLI reads daily price files and fits historical GBM. Build using the
[root instructions](../README.md#build-and-run); run commands from the repository
root. For models beyond GBM, use the [prediction setup](../apps/prediction/README.md).

## Prepare daily prices

Save provider exports under `Notes-dont-commit/`. The optional
[downloader](../apps/prediction/README.md) creates validated OHLCV snapshots and
metadata. The GBM command uses only the selected price column.

```csv
Date,Adj Close
2024-01-02,100.00
2024-01-03,101.00
2024-01-04,100.50
```

These are format examples, not market observations. Rows must be oldest first,
with unique valid ISO dates and positive finite prices. Extra columns are allowed;
row lengths must match the header. Currency symbols and thousands separators
are invalid.

Use consistent adjustment semantics. Unadjusted `Close` works with
`--price-column Close`, but corporate actions can distort returns. Adjusted
forecasts describe the provider's series, which may differ from quoted prices.
Calendar-gap/weekend checks are heuristic; no rows are filled or dropped.

Forecasting needs three prices; backtesting needs at least
`lookback + horizon + 1`, which gives only one forecast. Supply more history
for evaluation. N training returns require N+1 prices.

## Record provenance

A companion `AAPL.meta` uses these seven keys:

```text
provider=Your data provider
symbol=AAPL
adjustment=Describe split and dividend treatment
retrieved_on=2026-10-07
source_url=https://your-provider.example/history
price_column=Adj Close
frequency=daily
```

Replace all values with export details. Retrieval must not precede the latest
price date. Unknown, duplicate or missing keys, mismatched price columns and
non-daily frequency are rejected. Metadata declares provenance; it does not
verify the provider's adjustment or historical publication timing.

Keep the CSV/metadata hashes, source revision and toolchain version with
experiment artifacts. Use `sha256sum`, or `shasum -a 256` on macOS.

## Forecast and backtest

```bash
./build/mcprice forecast --csv Notes-dont-commit/AAPL.csv \
  --metadata Notes-dont-commit/AAPL.meta --lookback-days 252 --horizon-days 20

./build/mcprice backtest --csv Notes-dont-commit/AAPL.csv \
  --lookback-days 252 --horizon-days 20 --step-days 20 \
  --volatility-model ewma --ewma-decay 0.94 \
  --drift-model shrinkage --drift-shrinkage 0.5 \
  --bootstrap-samples 10000 --bootstrap-seed 42
```

Horizons count trading sessions after the last supplied row, even if stale.
Defaults use historical drift and sample volatility; insufficient history fails.
A step equal to the horizon avoids overlapping target returns, though shared
training windows can still make errors dependent.

Compare with latest price and other baselines on identical rows. Check
[error, probability, range and uncertainty metrics](calculations.md) together;
[CLI options](cli.md#forecast-and-backtest) cover bootstrap sensitivity and CSV export.
For multiple stocks, use the [manifest benchmark](forecast-benchmark.md).
Freeze settings before a final evaluation.

## Troubleshooting

- Missing column: match the header exactly with `--price-column`.
- Invalid dates/prices: correct the source export and investigate gaps.
- Short history: provide more sessions or explicitly choose a smaller workload.
- Stale forecast: inspect the CSV's final date; no live quote is fetched.
- Build/dependency problems: check the [installation instructions](../README.md#build-and-run).

Keep reproducible predictions as CSV/JSON rather than terminal dumps. After a
large run, replace the relevant result summary and environment details instead
of appending another progress note.
