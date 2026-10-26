# Direction selection and calibration

This opt-in experiment selects a pooled direction model using past data, then
fits a probability adjustment on a separate past year. Existing classifiers,
forecasts, pricing and Greeks stay unchanged. It does not produce price targets.

Use the same frozen manifests as [pooled direction](pooled-direction.md), with
enough history for three earlier calendar years plus pre-fold training:

```bash
.venv/bin/python -m pip install -r python/requirements-direction.txt
.venv/bin/python python/calibrated_direction.py \
  --manifest Notes-dont-commit/benchmark.json \
  --context-manifest Notes-dont-commit/stock-context.json \
  --output Notes-dont-commit/calibrated-validation.json

.venv/bin/python tests/test_calibrated_direction.py
```

## Fixed chronological procedure

For each horizon and outer test year Y, all stocks share these boundaries:

| Period | Role |
| --- | --- |
| Y−3 and Y−2 | Compare five candidates using out-of-fold Brier scores, equally weighting each asset/year. |
| Y−1 | Fit the chosen candidate's sigmoid adjustment using only its out-of-fold predictions. No reselection here. |
| Y | Refit the chosen base model on matured past labels; score raw and adjusted predictions on the existing test grid. |

The candidates, in exact-tie order, are historical up frequency, logistic C=0.1,
logistic C=1, LightGBM L2=1 and LightGBM L2=10. Other settings stay fixed. The
baseline can win; that is a reported choice, not a hidden fallback.

Each inner fit trains strictly before its year's start. Inner targets must
finish before that year's end; boundary-crossing labels are excluded. Outer
training labels finish before the outer fold starts. The first partial test
year uses the same three preceding full calendar years. Earlier validation
outcomes may enter later fits once observed. No random splitting is used.

Calibration is L2 logistic C=1 on raw probability logits, clipped to
[1e-6, 1−1e-6]. It uses equal row weights, no threshold search, and a fixed 0.5
decision threshold. Each OOF block needs 20 rows and ten distinct origin dates;
each asset still needs 40 matured pre-fold labels. Single-class fits are
explicitly constant. Failed fits stop the run; existing outputs are never
overwritten. Insufficient history is an error, not a skipped stock or year.

## Reading the report

The output retains losing candidate scores, OOF rows and cutoffs, chosen models,
calibrator coefficients, raw/adjusted predictions, source/input hashes and
versions. Both outputs are compared with the same always-up and past-frequency
baselines using hit rate, balanced accuracy, Brier, log loss and reliability bins.
Paired bootstrap checks also compare adjusted probabilities with raw ones.
Lower Brier alone does not prove better calibration: it also reflects predictive
discrimination. See the [calibration reference](https://scikit-learn.org/stable/modules/calibration.html).

The next-year base refit can change its score distribution, so an adjustment
learned on an older model may help or hurt. The sigmoid can reverse a weak
ranking. Correlated stocks and repeated experiments limit the evidence; this is
still inspected validation, not a fresh test or trading edge. Settings are not
retuned after outcomes, and 2024 onward remains reserved by the manifest.

## October 21 validation

Same ten-stock snapshot and 2018–2023 outer grids as Day 4:

| Horizon | Raw Brier | Adjusted Brier | Adjusted hit rate | Always-up |
| --- | --- | --- | --- | --- |
| 1 session | 0.25106 | 0.25063 | 52.2% | 52.5% |
| 5 sessions | 0.25090 | 0.25023 | 55.2% | 55.9% |
| 20 sessions | 0.28277 | 0.25107 | 59.9% | 60.7% |

Historical up frequency won 16/18 outer selections; logistic C=0.1 and LightGBM
L2=10 won one each. This does not establish a model edge. Daily reliability-bin
error rose from 0.00254 to 0.00469 despite the lower Brier score. Existing modes
exactly reproduced Day 4, and full candidate/OOF evidence stays in the ignored
snapshot. Lower probability loss is not proof of useful price prediction.
