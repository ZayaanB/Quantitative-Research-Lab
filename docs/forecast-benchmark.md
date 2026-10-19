# Multi-stock accuracy benchmark

The runner calls the C++ engine with historical drift, sample volatility, and
252 sessions/year. It evaluates 1, 5, and 20-session horizons with matching
steps, so target return windows do not overlap. It does not tune models.

## Freeze the experiment

1. Export daily histories and metadata for at least two stocks. Follow the
   [data workflow](forecast-workflow.md); keep private exports outside Git.
2. Copy `examples/forecast_benchmark.template.json` to
   `Notes-dont-commit/benchmark.json`. Paths are relative to that manifest.
3. Set the ticker list, lookback, and chronological split dates before testing.
   Use enough history before validation and enough sessions in both periods
   for the 20-session horizon. Every asset/horizon must have eligible forecasts.
4. Replace both checksum placeholders per stock with the output of
   `sha256sum FILE.csv FILE.meta` (macOS: `shasum -a 256`).

The manifest accepts the template's fields (`bootstrap` is optional). Changed input bytes,
duplicate tickers, invalid metadata, or empty evaluation groups fail the run.
Provenance is declared by you; hashes establish file identity, not data quality.

## Run validation

Requires Python 3.9+; no third-party Python packages are needed.

```bash
python3 python/benchmark_forecasts.py \
  --manifest Notes-dont-commit/benchmark.json \
  --engine build/mcprice \
  --output Notes-dont-commit/validation.json
```

Validation is the default. Only rows before `holdout_start` reach the engine.
Both origin and target must be inside the evaluation period; boundary-crossing
forecasts are excluded. Origins remain anchored to the engine's first eligible
origin, not reset at each split. Previously observed holdout prices can enter
later training windows during a holdout run, as in a live rolling forecast.

The runner validates and hashes the whole source file, but does not score
holdout targets in validation mode. It snapshots the executable and input files
for each run. JSON records the manifest, input/executable hashes, platform,
per-stock metrics, and full-precision forecast rows. It never overwrites an
existing output and does not pool dollar errors across stocks.

## Compare configurations (Day 35)

```bash
python3 python/compare_forecasts.py \
  --manifest Notes-dont-commit/benchmark.json --engine build/mcprice \
  --output Notes-dont-commit/comparison.json
```

This validation-only runner compares 18 fixed configurations: 63/126/252-session
lookbacks, sample/EWMA volatility, and zero/historical/50%-shrunk drift. EWMA
decay stays at 0.94. Allow at least 252 sessions before validation.
All candidates use common origins, sampled every horizon sessions, with no
holdout targets supplied to the engine. The score equally weights stock/horizon
MAE ratios against latest-price forecasts. Latest price itself can win; ties
favor it, then the documented candidate order. Perfect-baseline groups are
excluded when both errors are zero; a candidate failing one is ineligible.
If all groups are flat, latest price wins. JSON includes every trial and date.

This is model selection, not an independent accuracy test. No bootstrap
significance is claimed after searching candidates. Freeze the chosen procedure
before evaluating untouched data. The original benchmark command still tests
historical/sample with its manifest lookback; it does not load this selection.
Research and suggested next experiments are in [model research](forecast-model-research.md).
The separate [ridge return model](ridge-forecast.md) is a validation-only Day 36 experiment.
Day 37 adds a [fixed 50/50 ensemble](ensemble-forecast.md) with paired comparisons.
Day 38 adds [out-of-sample residual ranges](residual-forecast.md) and coverage checks.
Day 39 provides the [frozen final-holdout workflow](final-forecast.md).
Optional [ARIMA/ETS experiments](classical-forecast.md) remain validation-only challengers.
The [LightGBM return experiment](boosted-forecast.md) adds a fixed nonlinear challenger.

To compare saved point-model reports on exactly the same dates as GBM, use the
[matched comparison](matched-forecast.md). It validates inputs and reports gaps
without silently intersecting model samples or touching final-test targets.

## Run the final holdout

Freeze your procedure after validation, then run the final test explicitly:

```bash
python3 python/benchmark_forecasts.py \
  --manifest Notes-dont-commit/benchmark.json --engine build/mcprice \
  --phase holdout --output Notes-dont-commit/holdout.json
```

This flag prevents accidental scoring, not deliberate repeated testing.
Editing settings after looking at holdout results makes that period validation
data; you need a new untouched period for a final test.

## Interpret the output

Compare selected-model MAE/RMSE against `latest_price`. Positive relative MAE
improvement favors the selected model. MAPE, direction accuracy, coverage, and
relative improvements are fractions, not percentages. Direction accuracy
excludes unchanged point forecasts and counts an unchanged actual as neither
up nor down. An unavailable statistic is JSON `null`, never zero.

Coverage alone is not enough: compare interval widths too. These are descriptive
results, not proof of a predictive edge. Nonoverlapping
targets can still have dependent errors. The runner uses declared daily cadence,
not an exchange calendar, and cannot remove survivorship or corporate-action
bias from an unsuitable export.

MASE averages each absolute error divided by the mean one-session absolute
price change in that origin's training window. It is scale-free, but its unit
is a past one-session error even for a 20-session forecast: compare model and
baseline MASE at the same horizon. If any origin has a zero scale, the group's
MASE is unavailable rather than silently excluding that origin.

The always-up directional baseline includes every target; unchanged prices are
not rises. Brier baselines use constant 50%, constant 100%, and the frequency of
strict rises over same-horizon windows inside each training set. The trailing
baseline is unavailable if the horizon exceeds the lookback. Compare direction
accuracy with care: a model that abstains can have a smaller denominator.

Paired circular-block bootstrap intervals compare latest-price and selected
absolute errors within each scored group. Configure `bootstrap.samples`,
`bootstrap.seed`, and `bootstrap.block_sizes` before evaluation. Defaults are
2000, 42, and `[1,2,4]`; block lengths count forecast observations. Fewer than
ten complete blocks gives unavailable bounds and an inconclusive label. This
guard is not proof of independence. Check whether conclusions change across
block sizes; do not select the most favorable interval. Python and C++ use
different random generators, so their bootstrap bounds need not match exactly.

For your own analysis, export the engine's individual predictions directly:

```bash
./build/mcprice backtest --csv examples/sample_prices.csv \
  --lookback-days 10 --horizon-days 3 --step-days 3 --format csv
```
