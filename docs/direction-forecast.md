# Stock direction experiments

Logistic regression and LightGBM classification predict whether the selected
price will rise over 1/5/20 trading sessions. Unchanged prices count as **not up**.
These models do not output price targets, replace the price regressors, or touch
option pricing. Their probabilities are not calibrated or trading signals.

Use the two manifests from the [feature guide](stock-features.md). Run from the
repository root with a new output filename:

```bash
.venv/bin/python -m pip install -r python/requirements-direction.txt
.venv/bin/python python/direction_forecasts.py \
  --manifest Notes-dont-commit/benchmark.json \
  --context-manifest Notes-dont-commit/stock-context.json \
  --output Notes-dont-commit/direction-validation.json

.venv/bin/python tests/test_direction_forecasts.py
```

## Fixed procedure

Both classifiers use the same 28 features and origin/target rows. Origins start
at the first eligible validation session and advance by the horizon. Set
`lookback_days` to at least 80; each fit needs 40 matured labels.

Fit separately per stock/horizon at each calendar year's start, or at
`validation_start` for the first fold. Use up to `lookback_days` preceding
feature origins, keeping only labels whose targets are **strictly before the
fold starts**. Models stay frozen inside that fold. Earlier validation outcomes
can enter the following year's training once observed. No label crosses the
current training/test boundary, and no 2024+ row enters this run when the
manifest reserves 2024 onward.

Logistic regression uses training-only standardization and L2 regularization
with C=1. LightGBM uses a binary objective, 80 rounds, depth 3 and 7 leaves.
Both run on CPU with fixed settings; there is no tuning, class balancing or
calibration. Fit failures stop the run. If a training set contains one class,
both report an explicit constant prediction instead of claiming a native fit.
See the [logistic API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
and [LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html).

## Read the results

Every group compares the models with always-up, training-period up frequency,
and a latest-price tie baseline (always-not-up for this binary task). A probability
of 0.5 counts as up. Reported metrics include accuracy, balanced accuracy,
Brier score, log loss, confusion counts and five reliability bins. Lower Brier
and log loss are better; log loss clips probabilities at 1e-15. Balanced accuracy
is undefined when only one target class appears. A high always-up hit rate can
reflect market drift rather than skill.

Paired circular block bootstraps check accuracy gains over always-up and Brier
reductions over past up frequency. Positive differences favor the model.
Intervals are unavailable below ten effective blocks and are not corrected for
multiple experiments. Pooled metrics are descriptive; assets/dates are correlated.

The output keeps fold cutoffs, scaler statistics, sample counts, input/source
hashes, dependency versions and runtime. Invalid data saves a blocked diagnostic
report; fit/configuration failures save no partial scores. Existing files are
never overwritten. Probability reports are rejected by the point-price matched
runner so they cannot accidentally be treated as dollar forecasts.

This is inspected validation, not final-test evidence. Annual refits also differ
from the price regressor's per-origin fits: a difference is not proof that the
features alone caused an improvement. Final testing, model selection, pooling
and probability calibration remain separate milestones.

## October 20 validation

The saved October 17 snapshot, AAPL/MSFT/GOOGL, 2018–2023, fixed yearly fits:

| Horizon | Samples | Logistic hit rate | LightGBM hit rate | Always-up |
| --- | --- | --- | --- | --- |
| 1 session | 4,524 | 51.2% | 50.8% | 53.4% |
| 5 sessions | 903 | 53.2% | 50.4% | 60.2% |
| 20 sessions | 225 | 54.2% | 53.3% | 61.3% |

Both classifiers also had worse pooled Brier scores than past up frequency at
each horizon. No group showed an accuracy gain over always-up across the tested
bootstrap block sizes. These fixed challengers lost; existing models stay intact.
The full report is kept with the ignored snapshot, not presented as a final test.
