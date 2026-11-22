# Ontario demand forecasting

Working: an IESO importer, seasonal baselines, CPU LightGBM and calibrated
demand ranges. All remain retrospective research, not a live forecasting service.
This predicts demand in MW, not electricity prices or stock prices.

## Run it

From the repository root, with Python 3.9+ and the OS time-zone database.
The importer/baselines need no extra packages or GPU. Use new output paths.

```bash
python3 apps/energy/ieso.py --year 2023 \
  --output Notes-dont-commit/energy/ieso-2023

python3 apps/energy/baselines.py backtest \
  --csv Notes-dont-commit/energy/ieso-2023/PUB_Demand_2023.csv \
  --start 2023-01-08 --end 2023-12-31 --retrospective \
  --output Notes-dont-commit/energy/baselines-2023.json

python3 apps/energy/baselines.py forecast \
  --csv Notes-dont-commit/energy/ieso-2023/PUB_Demand_2023.csv \
  --origin 2023-12-31T08:00:00-05:00 --retrospective \
  --output Notes-dont-commit/energy/example-24h.json

python3 tests/test_energy_baselines.py
```

The forecast example creates a historical 24-hour scenario, not a current forecast.
`backtest` uses origin dates at 08:00 EST; `--end` is exclusive. The first week
supplies lag history. The last evaluated origin is December 30 so all targets
fit in the year. For longer histories, pass multiple `--csv` paths in
chronological order, from separate snapshots.

Both models score identical hourly targets. Reports include MAE, RMSE, bias,
WAPE (an error percentage), peak-magnitude error for each 24-hour window,
runtime, predictions, source timestamps and file/code hashes. Peak error is
not peak timing or Toronto calendar-day peak error. Ranges have a separate runner below.

## Time and availability

IESO reports [market time in fixed EST](https://ieso.ca/-/media/Files/IESO/Document-Library/engage/ca/ca-Introduction-to-the-Capacity-Auction.pdf)
(UTC−5), not daylight-saving time. Hour 1 means 00:00–01:00 EST; Hour 24 means
23:00–00:00 the next day. We retain UTC starts/ends and offset-aware
`America/Toronto` calendar times, including the repeated fall hour. A market day
always has 24 rows. Missing, duplicate, unordered or invalid rows fail; none are
filled or dropped. Only `Ontario Demand` is the target, not `Market Demand`.

These annual files are revised history, not point-in-time observations.
`--retrospective` acknowledges that limitation. We assume each market day's
values become available at 08:00 EST the next day. This is a research assumption,
not an IESO publication guarantee; it does not undo later revisions.

For every target hour, the baselines repeat demand 24 or 168 hours earlier.
If that source is not yet available under the assumption, repeat the same season
again until it is. At an 08:00 origin, the daily baseline uses a two-day lag for
8/24 targets. Each prediction records its actual source and repetition count.
Missing available sources fail rather than silently switching models or dates.
The horizon is `[origin, origin + 24 hours)`; no realized horizon loads are used.

## CPU LightGBM

Download 2018–2022 into separate `ieso-2018`, ..., `ieso-2022` snapshots using
the importer above; keep the existing `ieso-2023` snapshot. Then:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r apps/energy/requirements.txt
.venv/bin/python apps/energy/boosted_demand.py --csv \
  Notes-dont-commit/energy/ieso-2018/PUB_Demand_2018.csv \
  Notes-dont-commit/energy/ieso-2019/PUB_Demand_2019.csv \
  Notes-dont-commit/energy/ieso-2020/PUB_Demand_2020.csv \
  Notes-dont-commit/energy/ieso-2021/PUB_Demand_2021.csv \
  Notes-dont-commit/energy/ieso-2022/PUB_Demand_2022.csv \
  Notes-dont-commit/energy/ieso-2023/PUB_Demand_2023.csv \
  --retrospective --output Notes-dont-commit/energy/lightgbm-validation.json
.venv/bin/python tests/test_energy_lightgbm.py
```

One fixed fit uses January 15, 2018–2021 origins and 19 calendar/available-load
features. The first 14 days supply history; the last training window is purged
because its labels cross the boundary. Training labels must also be available
at the fit origin. Validation covers January 1, 2022–December 30, 2023 origins.
No validation labels enter the fit, and there is no tuning or early stopping.

The regressor pools all 24 lead-hour rows, using local/market calendar fields,
available 24/48/168/336-hour seasonal loads and recent day/week summaries.
It uses 200 rounds, learning rate 0.05, 31 leaves, depth 6 and L2 penalty 5,
with one CPU thread. Missing inputs or invalid predictions fail without fallback.
Reports preserve the model text/hash, dependency versions and source timestamps.
`--train-start`, `--train-end`, `--start` and `--end` are configurable, but this
development runner refuses evaluation beyond the declared 2023 boundary.

Validation used 729 matched origins, 17,496 hourly targets and 34,704 training rows.
Preparation/fit took about 6.9 s excluding JSON output (i9-13900H, LightGBM 4.7.0,
NumPy 2.5.3, CPU only).

| Model | MAE (MW) | RMSE (MW) | WAPE | Window peak MAE (MW) |
| --- | --- | --- | --- | --- |
| Daily baseline | 879.34 | 1,180.86 | 5.61% | 802.84 |
| Weekly baseline | 1,095.77 | 1,505.39 | 6.99% | 1,238.53 |
| LightGBM | 645.18 | 869.34 | 4.11% | 676.29 |

LightGBM reduced MAE by 26.6% versus the matched daily baseline. Its mean bias
was −109.60 MW, so it tended to underpredict. This is inspected validation on
revised data with assumed availability, not a live result or a significance test.
The earlier seasonal-baseline sample below uses different dates and is not the comparator.

## Demand ranges

Use the same six snapshots and optional requirements as above:

```bash
.venv/bin/python apps/energy/uncertainty.py --csv \
  Notes-dont-commit/energy/ieso-2018/PUB_Demand_2018.csv \
  Notes-dont-commit/energy/ieso-2019/PUB_Demand_2019.csv \
  Notes-dont-commit/energy/ieso-2020/PUB_Demand_2020.csv \
  Notes-dont-commit/energy/ieso-2021/PUB_Demand_2021.csv \
  Notes-dont-commit/energy/ieso-2022/PUB_Demand_2022.csv \
  Notes-dont-commit/energy/ieso-2023/PUB_Demand_2023.csv \
  --retrospective --output Notes-dont-commit/energy/demand-ranges.json
.venv/bin/python tests/test_energy_uncertainty.py
```

The 5th/95th quantile models use the same training years, features and settings.
A widening-only [conformalized quantile regression](https://arxiv.org/abs/1905.03222)
correction is fitted on 2022 and frozen before 2023 evaluation. Scores are
`max(lower − actual, actual − upper, 0)`; take order statistic
`ceil((n+1) × 0.9)`. Crossing bounds are sorted and counted. Lower bounds cannot
be negative. The December 31 calibration window is purged because it crosses
the boundary; every retained label must be available at the first scoring origin.
The native fits use [LightGBM's quantile objective](https://lightgbm.readthedocs.io/en/stable/Parameters.html#objective).

Ranges used 8,736 calibration rows and 8,736 scoring rows; about 13 s on the
same CPU. The correction widened each side by 312.61 MW. No quantiles crossed.

| 2023 ranges | Coverage | Mean width (MW) | Mean endpoint pinball loss (MW) |
| --- | --- | --- | --- |
| Raw | 83.54% | 2,149.06 | 88.55 |
| Calibrated | 92.38% | 2,774.29 | 86.09 |

The report includes sample counts, coverage, width, pinball and interval score
by target Toronto hour and meteorological season. Calibrated seasonal coverage
was 91.7–93.0%; hourly coverage was 87.6–96.4%. Repeated DST hours remain distinct
observations, grouped under the same local-hour label. LightGBM point predictions
and baselines are unchanged on matched rows; this does not improve point accuracy.

The 90% target is marginal, not a guarantee for each hour/season or all 24 hours
together. Serial dependence, revised data and distribution changes prevent a
standard exchangeability-based live coverage claim. This is inspected validation,
not an untouched final test. Weather is still excluded.

## Final evaluation

The runner freezes the saved point/quantile models, widening correction,
source/runtime, input hashes and the approved 2024-only origins. It does not
refit or recalibrate. Earlier observed final-period loads may enter later lag
features; future loads never enter their own forecasts.

The approved final scope is 2024 only. The original 2024–2025 comparison was never
scored: both 2025 annual variants lack May 1 HE1.
No missing values were filled. The approved grid has 365 origins at 08:00 EST,
January 1–December 30, forecasting 8,760 intervals entirely within 2024.
The first eight hours of January 1 and last sixteen of December 31 are outside
that fixed origin grid; the complete leap-year source was validated.

The one-shot workflow is below. The actual run has already consumed 2024;
reproducing it does not make that period fresh validation for another model.

```bash
.venv/bin/python apps/energy/final_evaluation.py freeze \
  --validation Notes-dont-commit/energy/demand-ranges.json \
  --csv Notes-dont-commit/energy/ieso-2023/PUB_Demand_2023.csv \
  Notes-dont-commit/energy/ieso-2024/PUB_Demand_2024.csv \
  --output Notes-dont-commit/energy/frozen-final-2024.json

.venv/bin/python apps/energy/final_evaluation.py run \
  --procedure Notes-dont-commit/energy/frozen-final-2024.json \
  --confirm-untouched-period --retrospective \
  --output Notes-dont-commit/energy/final-results-2024.json
```

Only confirm an untouched period if that is true. A `.used` marker is written
before scoring and retained after failures; preserve it instead of retrying as
though nothing was accessed. Copying/deleting files can bypass this local guard.
The primary comparison is LightGBM MAE versus the daily baseline, with weekly
results and range metrics retained. Paired circular-bootstrap intervals use
1/7/28 origin-day blocks, 2,000 samples and seed 42; fewer than ten effective
blocks gives no interval. These are sensitivity checks, not proof of a live edge.

Frozen inference took about 4.6 s excluding JSON output (i9-13900H, Linux kernel
7.0.0-43, with the validation dependencies).

| 2024 final model | MAE (MW) | RMSE (MW) | WAPE | Window peak MAE (MW) |
| --- | --- | --- | --- | --- |
| Daily baseline | 863.66 | 1,158.16 | 5.40% | 745.55 |
| Weekly baseline | 1,228.43 | 1,706.44 | 7.68% | 1,385.88 |
| Frozen LightGBM | 688.48 | 943.23 | 4.31% | 731.94 |

MAE was 20.3% below the daily baseline; peak error improved only about 1.8%.
Mean bias was −223.51 MW. The paired daily-baseline MAE reduction was 175.17 MW;
95% intervals for 1/7/28-day blocks were [122.04,230.47], [116.65,230.10] and
[123.82,226.45] MW. This is evidence for this retrospective comparison, not a
universal or live-performance claim, and it is not a stock-price result.
Raw/calibrated range coverage was 83.24%/91.77%; calibrated mean width was
2,882.36 MW and mean endpoint pinball loss 92.89 MW. The reports, models, input
hashes and `.used` marker remain in the ignored energy folder.

The artifact loader rejects unexpected model fields, duplicate keys and
non-finite values. Frozen source/runtime changes require a new explicit freeze.

## Data and next steps

### Hourly demand and peak experiment

Run the matched development comparison with the 2018–2023 snapshots and the
energy requirements above:

```bash
.venv/bin/python apps/energy/demand_experiment.py --csv \
  /path/to/2018.csv /path/to/2019.csv /path/to/2020.csv /path/to/2021.csv \
  /path/to/2022.csv /path/to/2023.csv --retrospective \
  --output Notes-dont-commit/energy/demand-experiment-2023.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tests/test_energy_demand_experiment.py
```

The experiment fits through 2021, uses 2022 error curves to estimate peak
probabilities and ranges, and evaluates 364 matched 08:00 EST origins in 2023
(8,736 hours). Boundary windows are purged before fitting and calibration.
The calendar features flag Ontario public holidays and adjacent local days.
Employer-specific observed days are not inferred. The blend weights are fixed at
50/50. Peak-hour error is the absolute difference between lead hours, using
the earliest maximum in a tie.

| Model | Hourly MAE (MW) | Bias (MW) | Peak MAE (MW) | Peak-hour MAE (h) |
| --- | ---: | ---: | ---: | ---: |
| Existing LightGBM | 620.20 | −129.44 | 668.36 | 1.253 |
| LightGBM with holidays | 610.49 | −133.20 | 657.61 | 1.269 |
| Additive model | 631.14 | −175.71 | 686.27 | 1.206 |
| Blend | 596.63 | −154.46 | 655.59 | 1.225 |

The blend's empirical 90% peak interval covered 90.66% with mean width
2,619 MW. Only three windows reached the exploratory 22,000 MW threshold.
A paired circular bootstrap (2,000 draws, seven-day blocks, seed 42)
gave a 95% interval of 2 to 48 MW for the blend's hourly MAE reduction
and −11 to 38 MW for its peak MAE reduction.
These are inspected retrospective development results. They use revised
load data and assumed availability. No real weather or operator score is
included in the table.

#### Archived weather and IESO forecasts

Pass `--weather` a CSV with this header. Add optional location columns ending
in `_c` after `toronto_c`. Celsius temperatures apply at the UTC start of each
target hour. Supply 24 rows per origin, all from the same archived forecast
issuance. Keep the raw runs and record their source and release evidence.
The loader checks the supplied issuance and availability timestamps against
the origin. It cannot independently establish when historical data was
public. Use forecast runs, such as the [NCAR GFS archive](https://gdex.ucar.edu/datasets/d084001/),
and keep reanalysis and realized temperature out of the inputs.

```text
origin_utc,target_start_utc,issued_at_utc,available_at_utc,source,toronto_c
```

The weather ablation adds hourly temperatures and forecast-window extremes.
The additive model uses piecewise-linear temperature effects and interactions
with hour of day. Every model is compared on the same origins. The report
includes separate metrics for summer, winter, and forecast-defined heat and cold
when temperatures are supplied.

Use `adequacy_input.py --manifest snapshots.csv --output operator.csv` to
normalize saved IESO Adequacy3 reports. The manifest columns are
`origin_date_est,available_at_utc,source,xml`. Provide the public source URL
and the local XML path relative to the manifest. Retain the two delivery-day
reports for each origin and record when each exact snapshot was seen or
otherwise proven public. The converter selects the latest eligible issuance
per delivery date and extracts Ontario **Average Demand**, keeping the raw
hash in the provenance. IESO's headline Forecast Ontario Demand can switch
to the highest five-minute value in an hour. Use `--operator operator.csv`
to score the hourly average comparator. See the [IESO field definitions](https://reports-public.ieso.ca/docrefs/helpfile/Adequacy3_h2.pdf).

For a recent operator comparison, keep historical fitting data in `--csv`
and use `--train-end 2022-01-01`, `--recent-csv` with a contiguous recent
demand block (including 14 days of lag history), and explicit 2026-or-later
calibration and scoring dates. The operator file must cover every retained
calibration and scoring hour. IESO publishes a rolling [Adequacy archive](https://www.ieso.ca/Power-Data/Data-Directory#System-Adequacy).
Neither workflow consumes or replaces the frozen final evaluation.

#### Research ideas to test

- Fan and Hyndman's additive model uses forecast temperature from multiple
  sites and lagged demand. Test whether Ottawa, London and a northern site
  improve over Toronto alone, then test prior temperature and humidity where
  release times can be verified. [Paper](https://robjhyndman.com/papers/2010STLF-FinalR1.pdf)
- Rubattu, Maroni and Corani use thermal summaries and discuss detrending
  before fitting boosted trees. Try predicting demand relative to a seasonal
  baseline, and compare fixed and trailing refits on the same origins.
  [Paper](https://arxiv.org/html/2305.05575v1)
- Amara-Ouali et al. model peak magnitude and timing across resolutions.
  Compare a direct window-peak prediction against the maximum of the hourly
  curve, and test whether the residual scenarios improve after grouping by
  season when enough archived windows are available. [Paper](https://arxiv.org/abs/2112.04492)

Source: [IESO annual demand reports](https://reports-public.ieso.ca/public/Demand/),
described in the [data directory](https://www.ieso.ca/Power-Data/Data-Directory).
Downloads keep original bytes, creation headers, retrieval time, HTTP metadata,
SHA-256, units and code provenance. TLS verification stays on, requests are bounded,
and existing snapshots/reports are never overwritten. Raw data stays ignored.
Review [IESO's terms](https://www.ieso.ca/Terms-of-Use) before reuse or redistribution.
This information is subject to the Terms of Use set out in the IESO's website (www.ieso.ca).

The standalone baseline diagnostic used January 8–December 30, 2023 origins:
357 origins, 8,568 hourly targets, about 0.29 s on the
development laptop (i9-13900H, Python 3.12, Linux; excluding download/JSON).

| Baseline | MAE (MW) | RMSE (MW) | WAPE |
| --- | --- | --- | --- |
| Daily, with disclosed recursion | 870.55 | 1,170.07 | 5.56% |
| Weekly | 1,034.76 | 1,468.39 | 6.61% |

These are errors, not classification hit rates or `100 − error` accuracy.
The fixed split is training 2018–2021 and validation 2022–2023.
Energy 2024 was consumed by the approved final comparison; 2025 remains unscored
and incomplete. Future changes need a newly frozen evaluation, not claims based
on repeatedly testing 2024. Revised-data results still need
historical vintages or prospective collection before any operational claim.
Weather is excluded until archived forecasts available at each origin exist.
Stock and option-pricing inputs are unchanged.
