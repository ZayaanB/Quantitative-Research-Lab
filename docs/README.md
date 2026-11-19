# Documentation

Start with the [project overview](../README.md), then choose a guide:

| Guide | Covers |
| --- | --- |
| [Calculations explained](calculations.md) | Instruments, inputs, Greeks, pricing statistics, forecast metrics and how to interpret them |
| [CLI](cli.md) | Commands, flags, defaults, inputs and outputs |
| [Models](models.md) | Pricing models, stock scenarios, prediction experiments and energy models |
| [Research status and next steps](roadmap.md) | Implemented work, current evidence and proposed experiments |

## Technical methods

- [Architecture](architecture.md) and [mathematics](mathematics.md).
- [Forecast research](forecast-model-research.md).
- [Ontario demand data, commands and results](../apps/energy/README.md).

## Stock workflows

- [Prepare data, forecast and backtest](forecast-workflow.md).
- [Multi-stock benchmark and manifests](forecast-benchmark.md).
- [Matched-date comparison and data checks](matched-forecast.md).
- [OHLCV and market-context features](stock-features.md).
- [Logistic and LightGBM direction classifiers](direction-forecast.md).
- [Pooled cross-stock training](pooled-direction.md).
- [Past-only model selection and calibration](calibrated-direction.md).
- [Optional CPU CNN](neural-direction.md).
- [Stock price models and empirical ranges](models.md#running-price-experiments).
- [Frozen final evaluation](final-forecast.md).

Private datasets and experiment outputs stay in `Notes-dont-commit/`.
