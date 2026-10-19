# Matched forecasts and data checks

This validation-only tool compares saved point predictions with GBM and
latest-price baselines on identical dates. It does not train or tune a model.
Run commands from the repository root after building `mcprice`.

Use your frozen manifest and an existing ridge, ensemble or LightGBM validation
report. No native ML packages are needed to compare saved predictions:

```bash
python3 python/matched_forecasts.py \
  --manifest Notes-dont-commit/benchmark.json \
  --reference Notes-dont-commit/boosted-validation.json \
  --engine build/mcprice \
  --output Notes-dont-commit/matched-validation.json
```

Repeat `--reference FILE` to compare more reports. Their asset/horizon groups
and origin/target grids must match exactly; there is no silent intersection.
The first report declares the frozen start/warmup. From that start, each group
must have a complete non-overlapping grid through the validation boundary.
GBM is generated daily, then sampled at those same declared origins. The output
records how many extra GBM rows were not scored. Nothing is retuned or repaired.

Every report must have the same manifest hash/configuration and validation phase.
Dates, horizons, observed prices and any declared training cutoffs are checked.
Hashes establish identity, not proof that an upstream model was leakage-free.
For a classical-model report, use `--prediction-column ets_level` (or another
explicit point column). Failed/missing predictions block comparison; there is
no fallback or removal of difficult rows.

## Check data without scoring

```bash
python3 python/matched_forecasts.py \
  --manifest Notes-dont-commit/benchmark.json --quality-only \
  --output Notes-dont-commit/history-quality.json
```

The existing loader checks whole-file hashes, positive finite prices, dates and
metadata. Additional diagnostics inspect only history before `holdout_start`:
provided OHLC bounds, numeric columns, share volume, weekend rows and gaps over
four calendar days. Price-only CSVs work; `ohlcv_checked` then reports false.
These checks are not an exchange calendar and do not prove session completeness.
Warnings need review; they do not cause rows to be filled or dropped. Invalid
OHLCV or input hashes block scoring, save the diagnostic report, and return a
nonzero exit code. Invalid reference reports fail without saving partial scores.

Outputs include matched predictions, MAE/RMSE, direction counts, always-up,
paired bootstrap sensitivity, available GBM distribution metrics, provenance,
source/input hashes and runtime. Mean relative MAE equally weights asset/horizon
groups; latest price is 1.0. Both-zero groups are excluded; failure against a
perfect baseline makes that candidate's aggregate unavailable. No probabilities
or intervals are invented for point-only models. This is not final-test evidence.
Use fresh output names: existing reports are never overwritten.

```bash
ctest --test-dir build -R matched_forecast_runner --output-on-failure
```
