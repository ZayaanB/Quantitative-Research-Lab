# Ontario demand forecasting

Working: an IESO hourly-demand importer, seasonal baselines and optional CPU
LightGBM validation. Uncertainty estimates remain [Day 9](../../docs/roadmap.md).
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
not peak timing or Toronto calendar-day peak error. No uncertainty intervals yet.

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

October 30 comparison: 729 matched origins, 17,496 hourly targets; one fit on
34,704 training rows. About 6.9 s including preparation/fit, excluding JSON output,
on the same laptop (LightGBM 4.7.0, NumPy 2.5.3, CPU only).

| Model | MAE (MW) | RMSE (MW) | WAPE | Window peak MAE (MW) |
| --- | --- | --- | --- | --- |
| Daily baseline | 879.34 | 1,180.86 | 5.61% | 802.84 |
| Weekly baseline | 1,095.77 | 1,505.39 | 6.99% | 1,238.53 |
| LightGBM | 645.18 | 869.34 | 4.11% | 676.29 |

LightGBM reduced MAE by 26.6% versus the matched daily baseline. Its mean bias
was −109.60 MW, so it tended to underpredict. This is inspected validation on
revised data with assumed availability, not a live result or a significance test.
The earlier Day 7 sample below uses different dates and is not the comparator.

## Data and next steps

Source: [IESO annual demand reports](https://reports-public.ieso.ca/public/Demand/),
described in the [data directory](https://www.ieso.ca/Power-Data/Data-Directory).
Downloads keep original bytes, creation headers, retrieval time, HTTP metadata,
SHA-256, units and code provenance. TLS verification stays on, requests are bounded,
and existing snapshots/reports are never overwritten. Raw data stays ignored.
Review [IESO's terms](https://www.ieso.ca/Terms-of-Use) before reuse or redistribution.
This information is subject to the Terms of Use set out in the IESO's website (www.ieso.ca).

The October 29, 2026 diagnostic used January 8–December 30, 2023 origins,
fixed before scoring: 357 origins, 8,568 hourly targets, about 0.29 s on the
development laptop (i9-13900H, Python 3.12, Linux; excluding download/JSON).

| Baseline | MAE (MW) | RMSE (MW) | WAPE |
| --- | --- | --- | --- |
| Daily, with disclosed recursion | 870.55 | 1,170.07 | 5.56% |
| Weekly | 1,034.76 | 1,468.39 | 6.61% |

These are errors, not classification hit rates or `100 − error` accuracy.
The fixed split is training 2018–2021 and validation 2022–2023.
Energy 2024–2025 remains an unscored candidate final period;
freeze the complete procedure before using it. Revised-data results still need
historical vintages or prospective collection before any operational claim.
Weather is excluded until archived forecasts available at each origin exist.
Stock and option-pricing inputs are unchanged.
