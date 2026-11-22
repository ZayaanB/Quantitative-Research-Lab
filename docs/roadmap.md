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

The matched [energy development experiment](../apps/energy/README.md#hourly-demand-and-peak-experiment)
tested Ontario holidays and a fixed additive blend. The blend lowered 2023
hourly MAE from 620 MW to 597 MW and window-peak MAE from 668 MW to 656 MW.
Its mean underprediction increased. Operator XML and issued weather inputs
are supported, pending a matched historical or prospective comparison.

1. Validate reporting vintages and backtest archived weather and IESO forecasts.
2. Compare past-only rolling refits and seasonal-residual targets; recalibrate changes.
3. Test an additive/spline control and a fixed blend before increasing complexity.
4. Evaluate delayed-label adaptive ranges using coverage, width and pinball loss.

Rationale: [energy research priorities](forecast-model-research.md#energy-research-priorities).

After a benchmark or large test, replace the relevant result table and environment
details; revise this status when conclusions or blockers change. Keep raw artifacts
separate and avoid appending dated progress entries.
