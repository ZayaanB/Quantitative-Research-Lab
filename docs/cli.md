# Command-line interface

The CLI is a program run in a terminal with a command and named options.
Use the [build instructions](../README.md#build-and-run), then run from the
repository root. `./build/mcprice --help` lists all C++ options and defaults.

## Commands

| Command | Purpose | Output |
| --- | --- | --- |
| `price` | Value European calls/puts or arithmetic Asian calls | Analytical value where supported, Monte Carlo statistics and optional Greeks |
| `greeks` | European call/put sensitivities | Delta, Gamma, Vega, bumps and runtime |
| `forecast` | Fit historical GBM to a daily CSV | Growth/volatility, mean and median price, 95% range and up probability |
| `backtest` | Repeated past-only fits at historical origins | Baseline comparisons, errors, probabilities, ranges and uncertainty; optional CSV rows |

## Pricing and Greeks

```bash
./build/mcprice price --type call --method both \
  --spot 100 --strike 100 --rate 0.05 --volatility 0.20 --maturity 1 \
  --paths 1000000 --threads 4 --seed 42 --antithetic --greeks

./build/mcprice price --type asian-call --steps 252 --paths 100000 --antithetic

./build/mcprice greeks --type put --spot-bump 1 --vol-bump 0.01 \
  --paths 1000000 --antithetic
```

| Option | Meaning / default |
| --- | --- |
| `--type` | `call`, `put` or `asian-call`; default call |
| `--method` | `mc`, `analytical` or `both`; default both for European pricing; Asian uses MC |
| `--spot`, `--strike` | Current and exercise prices; both default 100 |
| `--rate`, `--volatility` | Annual decimal inputs; defaults 0.05 and 0.20 |
| `--maturity` | Years; default 1 |
| `--paths`, `--threads` | Total trajectories and workers; defaults 1,000,000 and 1 |
| `--steps` | Asian monitoring count; default 1; choose for the intended contract |
| `--batch-size` | Observations per worker batch; default automatic |
| `--seed` | Random seed; default 42 |
| `--antithetic` | Pair shocks; requires even paths |
| `--greeks` | Add Greeks to European price with MC or both methods |
| `--spot-bump`, `--vol-bump` | Absolute bumps; defaults 1% of spot and 0.01 volatility |

`greeks` uses Monte Carlo and does not accept `--method`. It runs five
valuations; paths apply to each. Vega is per one volatility percentage point.
See [calculation examples](calculations.md#greeks) for interpreting the output.

## Forecast and backtest

```bash
./build/mcprice forecast --csv examples/sample_prices.csv \
  --metadata examples/sample_prices.meta --lookback-days 10 --horizon-days 20

./build/mcprice backtest --csv examples/sample_prices.csv \
  --lookback-days 10 --horizon-days 3 --step-days 3

./build/mcprice backtest --csv examples/sample_prices.csv \
  --lookback-days 10 --horizon-days 3 --step-days 3 --format csv
```

Examples use synthetic prices. Real histories need chronological `Date` and
positive `Adj Close` values, or another selected price column. Optional metadata
declares provenance and adjustment semantics.

| Option | Meaning / default |
| --- | --- |
| `--csv` | Required history |
| `--price-column` | Price column; default Adj Close |
| `--metadata` | Optional key=value provenance file |
| `--lookback-days` | Returns per fit: full history for forecast, 252 for backtest; N returns need N+1 prices |
| `--horizon-days` | Sessions from origin to target; default 20 |
| `--trading-days` | Annualization convention; default 252 |
| `--volatility-model` | sample or ewma; default sample |
| `--ewma-decay` | Recent-weight decay in (0,1); default 0.94 |
| `--drift-model` | historical, zero or shrinkage; default historical |
| `--drift-shrinkage` | Fraction removed in [0,1]; default 0.5 when used |
| `--step-days` | Backtest spacing between origins; default 1 |
| `--format` | Backtest text summary or full-precision csv; default text |
| `--bootstrap-samples`, `--bootstrap-seed` | Backtest resamples and seed; defaults 10,000 and 42 |
| `--bootstrap-block-size` | Forecast observations per block; default overlap heuristic |
| `--bootstrap-sensitivity` | Also compare doubled and quadrupled blocks |

A step shorter than horizon gives overlapping targets. Interpret uncertainty
with that dependence in mind. See [metrics](calculations.md) and
[data preparation](forecast-workflow.md).

## Python and energy tools

Python runners are separate scripts rather than mcprice subcommands. Most stock
experiments read a manifest of assets, hashes, dates and settings and save JSON.
Templates in examples need real paths and hashes. Install documented optional
requirements in `.venv`.

```bash
python3 python/ridge_forecasts.py --help
python3 python/stock_features.py --help
python3 apps/energy/baselines.py --help
```

Install only the requirements needed for your task:

| Requirements file | Purpose |
| --- | --- |
| `python/requirements.txt` | Numerical validation and plots |
| `python/requirements-data.txt` | Yahoo history downloader |
| `python/requirements-forecast.txt` | ARIMA/ETS via statsmodels |
| `python/requirements-ml.txt` | LightGBM return regression |
| `python/requirements-direction.txt` | LightGBM/scikit-learn direction models |
| `python/requirements-neural.txt` | NumPy/PyTorch CNN; follow the [CPU install guide](neural-direction.md) |
| `apps/energy/requirements.txt` | Energy LightGBM and ranges |

The [prediction setup](../apps/prediction/README.md) covers downloads and manifests;
[models](models.md#running-price-experiments) provides price-model commands.
Other entry points are `benchmark_forecasts.py`/`compare_forecasts.py` for GBM
comparisons, `matched_forecasts.py` for matched scoring, `stock_features.py`
for features, and the direction/calibration/CNN runners linked in the
[documentation index](README.md). `validate_black_scholes.py` and `plot_*.py`
handle numerical checks and plots. Shared imports stay in `python/`.

Use the [energy guide](../apps/energy/README.md) for hourly data and commands.
Use new snapshot and output paths to preserve runs.
