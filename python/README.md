# Python tools

Run scripts from the repository root. These files share evaluation utilities;
keep their import paths stable. C++ option pricing has no Python dependency.

| Area | Entry points |
| --- | --- |
| Data | `download_stock_history.py` |
| GBM evaluation | `benchmark_forecasts.py`, `compare_forecasts.py` |
| Matched comparison and data checks | `matched_forecasts.py`, `forecast_quality.py` |
| Stock feature export | `stock_features.py` ([guide](../docs/stock-features.md)) |
| Stock direction | `direction_forecasts.py` ([per-stock](../docs/direction-forecast.md), [pooled](../docs/pooled-direction.md)) |
| Direction selection/calibration | `calibrated_direction.py` ([guide](../docs/calibrated-direction.md)) |
| Stock models | `ridge_forecasts.py`, `ensemble_forecasts.py`, `classical_forecasts.py`, `boosted_forecasts.py` |
| Ranges and final test | `residual_forecasts.py`, `final_forecast.py` |
| Pricing checks and plots | `validate_black_scholes.py`, `plot_*.py` |

Create `.venv` once. Install only the requirements for the tool you are using:

- `requirements.txt`: numerical checks and plotting.
- `requirements-forecast.txt`: optional ARIMA/ETS models.
- `requirements-ml.txt`: optional LightGBM model.
- `requirements-direction.txt`: optional LightGBM and scikit-learn classifiers.
- `requirements-data.txt`: optional Yahoo history download.

See [stock forecasts](../apps/forecast/README.md) and
[prediction experiments](../apps/prediction/README.md) for exact commands.
No GPU is required; scikit-learn is only needed for direction models. Do not pass hourly
energy data through the daily positive-stock-price contract.
