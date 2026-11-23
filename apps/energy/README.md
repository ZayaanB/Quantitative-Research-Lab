# Ontario demand forecasting

Retrospective 24-hour forecasts of Ontario electricity demand in MW. For current
results see the [weather benchmark](#current-benchmark) and its
[reproducibility record](../../results/README.md#ontario-electricity-demand).

## Quick start

Run from the repository root. The importer and seasonal baselines use the
standard library and the OS time-zone database.

```bash
python3 apps/energy/ieso.py --year 2023 \
  --output Notes-dont-commit/energy/ieso-2023
python3 apps/energy/baselines.py backtest \
  --csv Notes-dont-commit/energy/ieso-2023/PUB_Demand_2023.csv \
  --start 2023-01-08 --end 2023-12-31 --retrospective \
  --output Notes-dont-commit/energy/baselines-2023.json
```

For the LightGBM and additive models, install the energy dependencies in `.venv`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r apps/energy/requirements.txt
```

| Script | Purpose |
| --- | --- |
| `baselines.py forecast` | Historical daily and weekly forecasts at one origin |
| `boosted_demand.py` | Calendar and lagged-demand LightGBM validation |
| `uncertainty.py` | Hourly 5th and 95th quantiles with a widening correction fitted on 2022 |
| `demand_experiment.py` | Matched calendar, weather and additive development comparisons |
| `recent_benchmark.py` | 2026 archived-weather and IESO operator benchmark |

Run a script with `--help` for its arguments. The original LightGBM and quantile
validation use annual IESO CSVs from 2018 through 2023 in chronological order.
Use new output paths. See the [test guide](../../tests/README.md) for native checks.

## Time and availability

IESO uses [fixed EST (UTC−5)](https://ieso.ca/-/media/Files/IESO/Document-Library/engage/ca/ca-Introduction-to-the-Capacity-Auction.pdf).
Hour ending 1 runs from midnight to 01:00. Each origin is 08:00 fixed EST
and each forecast covers the next 24 hours. The peak is the maximum within
that window. Inputs use `Ontario Demand` and require contiguous hourly rows.
The baselines repeat 24- or 168-hour lags, going back another full period when
the most recent lag is unavailable. Calendar features use Toronto local time.

The [annual IESO reports](https://reports-public.ieso.ca/public/Demand/) contain
revised history. The backtests assume each market day's load becomes available
at 08:00 fixed EST the following day. `--retrospective` acknowledges the resulting
limit on operational claims. Downloads retain raw bytes and source metadata
under `Notes-dont-commit/`. See the [IESO terms](https://www.ieso.ca/Terms-of-Use)
before reuse or redistribution. This information is subject to the Terms of Use
set out in the IESO's website (www.ieso.ca).

## Current benchmark

The July 15–November 19, 2026 comparison scored 128 matched windows
(3,072 hours). The default blends combine LightGBM and an additive model
with equal weights.

| Model | Hourly MAE (MW) | WAPE | Bias (MW) | Peak MAE (MW) | Peak timing MAE (h) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Calendar | 654 | 3.97% | −224 | 769 | 0.91 |
| Toronto weather | 563 | 3.42% | −321 | 628 | 0.98 |
| Four-city weather | 564 | 3.42% | −383 | 645 | 0.94 |
| Tuned four-city weather | 568 | 3.45% | −390 | 640 | 0.98 |

Toronto weather reduced hourly MAE by 13.9% (paired 95% interval of 41–146 MW).
Four cities reduced hourly MAE from 794 to 753 MW on the 19 forecast-hot
windows (Toronto forecast ≥28 °C). No windows met the −10 °C cold threshold.
Toronto's nominal 90% peak interval covered 94.5% of windows at a mean width
of 2,556 MW. The broader test included 15 actual 22,000 MW exceedances.

On 44 matched fall windows, IESO averaged **242 MW** hourly and **210 MW**
peak error, versus **473 MW** and **522 MW** for Toronto. IESO is the
operating reference. Use the Toronto forecast for side-by-side research.
The tuned model's higher scoring-period error favors the default settings.

### Reproduce the benchmark

From the root, use the saved demand snapshots and normalized forecast inputs:

```bash
energy=Notes-dont-commit/energy
run="$energy/benchmark-2026-11-22"
.venv/bin/python apps/energy/recent_benchmark.py \
  --demand-2024 "$energy/ieso-2024-20261109/PUB_Demand_2024.csv" \
  --demand-2026 "$run/PUB_Demand_2026.csv" \
  --weather "$run/weather.csv" --operator "$run/operator.json" \
  --output /tmp/ontario-weather-benchmark.json
```

Add `--tune` with a new output path to reproduce the tuning run.

### Archived forecast inputs

The weather CSV uses the header below. Include 24 UTC target-hour starts
from a single forecast issuance per origin. Additional Celsius location
columns end in `_c` and follow `toronto_c`. Preserve the raw forecast and
evidence of when it was available.

```text
origin_utc,target_start_utc,issued_at_utc,available_at_utc,source,toronto_c
```

The recent benchmark uses the archived `operator.json` with issue times,
HTTP Last-Modified timestamps, raw hashes and hourly values. For a
`demand_experiment.py` run, `adequacy_input.py --manifest snapshots.csv
--output operator.csv` converts saved XML to the required CSV. The manifest
header is `origin_date_est,available_at_utc,source,xml`. Provide the source
URL and XML path relative to the manifest, with both delivery dates
represented at each origin. Use Ontario **Average Demand** for the
hourly benchmark. [IESO field definitions](https://reports-public.ieso.ca/docrefs/helpfile/Adequacy3_h2.pdf)

## Final evaluation

**2024 has been consumed.** The frozen retrospective evaluation scored
365 windows (8,760 hours) using models fitted through 2021 and a 2022 range
correction. LightGBM achieved 688.48 MW hourly MAE (4.31% WAPE) against
863.66 MW for the daily baseline. Peak MAE was 731.94 MW and the
calibrated hourly interval covered 91.77%. The 2025 source lacks May 1 hour
ending 1 and remains unscored.

The frozen procedure, result and `.used` marker are preserved in
`Notes-dont-commit/energy/`. Keep the marker with its procedure. Reserve
fresh data for another final comparison. See the [research status](../../docs/roadmap.md)
and [forecast research](../../docs/forecast-model-research.md) for follow-up work.
