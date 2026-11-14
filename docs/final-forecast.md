# Freeze and evaluate the final procedure

Day 39 separates validation from an explicit final holdout attempt. It uses the
[empirical residual workflow](residual-forecast.md), not C++ option valuation.
Python 3.9+ is enough. No extra packages are needed.

Phase 2 Day 10 status (November 9): the original frozen stock exports/procedure
are still absent locally. No stock final test was run. The downloaded 2024+
candidate is not a replacement for that missing experiment. The separate
[energy final workflow](../apps/energy/README.md#final-evaluation) has its own data gate.

## Freeze before scoring

Keep the original stock list, data hashes, lookback and split dates. Generate a
current residual validation report, then choose the primary model explicitly:
Use fresh output filenames; existing files are never overwritten.

```bash
python3 python/residual_forecasts.py \
  --manifest Notes-dont-commit/benchmark.json \
  --output Notes-dont-commit/residual-validation.json

python3 python/final_forecast.py freeze \
  --manifest Notes-dont-commit/benchmark.json \
  --validation Notes-dont-commit/residual-validation.json \
  --model ensemble --output Notes-dont-commit/procedure.json
```

Use `--model latest_price` instead if validation does not justify complexity.
Freezing checks input identities and matching validation/source hashes. It
requires every stock/horizon group and rejects duplicates or split-crossing
rows without scoring holdout targets. Preserve the validation report and procedure file.
The ridge penalty, 50/50 weights, horizons, calibration windows and quantile
method remain fixed in the hashed source. Source, platform or Python-version changes
require renewed validation before freezing, not a silent final-test adjustment.

## One deliberate final attempt

Only run this if the period really is untouched:

```bash
python3 python/final_forecast.py run \
  --manifest Notes-dont-commit/benchmark.json \
  --procedure Notes-dont-commit/procedure.json \
  --confirm-untouched-holdout --output Notes-dont-commit/final-holdout.json
```

Scored origins start at `holdout_start`; targets end no later than `test_end`.
The procedure refits sequentially: earlier holdout closes and matured errors
can enter later forecasts, just as they could after becoming observable live.
No future target enters its own fit or calibration. Warmup and the original
grid anchor are preserved across the split.

The primary score equally weights stock/horizon MAE ratios versus latest price.
Below one is descriptively better; both-zero groups are excluded. A nonzero
error against a perfect baseline makes the aggregate unavailable. All-flat
data also gives an unavailable aggregate. Coverage, width, interval scores and
Brier scores remain secondary checks—not reasons to change the chosen model.
Both models are reported on paired dates; the primary choice is not reselected.
No statistical significance or reliable predictive edge is asserted.

## Guardrails and limits

The runner refuses changed inputs/source/runtime, existing outputs and a second
attempt beside the same procedure. It writes `procedure.json.used` before
evaluation. An evaluation failure leaves that marker: preserve it and diagnose
the failure rather than treating the data as untouched again. An absent output
does not mean no holdout data was accessed.

This is a local accident guard, not tamper-proof enforcement. Copying a procedure,
deleting its marker or using another tool bypasses it. Hashes prove identity,
not that validation is genuine or that the holdout was never viewed. If the
period has already influenced decisions, use a genuinely new period.

The synthetic tests exercise this flow; they do not establish stock accuracy.
Real final results require your original frozen exports. Do not replace missing
files with new stocks or dates after seeing outcomes and call it the same test.
