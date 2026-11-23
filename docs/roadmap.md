# Research status and next steps

## Current evidence

Stock validation has not established a useful edge over latest-price, always-up
or historical-frequency baselines. Pooled training and the CNN lost to the
relevant baselines; probability calibration improved Brier scores without
establishing better directional predictions. See the
[model guides](models.md) and linked validation tables.

Stock 2018–2023 is inspected validation. Downloaded rows from 2024 onward are
only candidate final data; check prior use before freezing. The original stock
final exports/procedure remain missing. See the [final workflow](final-forecast.md).

Energy uses fixed 2018–2021 models, a 2022 range correction and inspected
2023 validation. The approved 2024-only final comparison reused saved artifacts:
MAE improved against the daily baseline, with calibrated coverage near its
90% target. **2024 is consumed.** The 2025 source lacks May 1 HE1 and remains
unscored. See the [results and assumptions](../apps/energy/README.md#final-evaluation).

## Next experiments

Predeclare each procedure and reserve fresh final data.

The [current Ontario benchmark](../apps/energy/README.md#current-benchmark)
found that Toronto weather cut hourly MAE from 654 MW to 563 MW. IESO's
hourly MAE was 242 MW versus 473 MW for Toronto on 44 matched fall windows.
Tuning raised scoring-period hourly MAE. The 2026 scoring period is now
inspected. Reserve a new prospective period for operational claims.

1. Capture per-origin weather releases, IESO forecasts and demand vintages
   through both cold and hot conditions.
2. Compare trailing refits and seasonal residual targets to address
   underprediction. Test an IESO residual correction with fresh paired data.
3. Compare direct window-peak forecasts and adaptive ranges using timing,
   coverage, width and interval score.

Rationale: [energy research priorities](forecast-model-research.md#energy-research-priorities).

After a benchmark or large test, replace the relevant result table and environment
details; revise this status when conclusions or blockers change. Keep raw artifacts
separate and avoid appending dated progress entries.
