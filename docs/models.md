# Models and forecasting

The project separates risk-neutral option valuation, historical stock
forecasting and hourly Ontario demand forecasting. Their targets differ.

## Option valuation

**Black-Scholes** directly computes European call/put values. **Risk-neutral
GBM** simulates positive prices with expected growth equal to the supplied rate.
European Monte Carlo uses exact terminal draws; Asian simulation advances at
its monitoring times. Both discount payoffs at that rate. European analytical
prices check simulation; Asian calls have no implemented analytical reference.

Pricing assumes constant volatility, zero dividends and frictionless markets.
Greeks revalue European instruments with bumped inputs. More paths improve
sampling precision rather than the realism of the assumptions.

## Historical stock GBM

The C++ forecast fits daily log returns. Volatility uses sample SD or **EWMA**,
which weights recent squared deviations more heavily. Default decay is 0.94;
lower decay reacts faster. Drift retains historical growth, sets expected price
growth to zero, or shrinks historical drift toward zero by a chosen fraction.

The fitted lognormal distribution supplies mean/median prices, ranges and up
probabilities. Walk-forward evaluation refits through each historical origin.
It omits news, jumps and fitted-parameter uncertainty.
See the [workflow](forecast-workflow.md).

## Stock baselines and price models

| Method | Simple explanation | Role |
| --- | --- | --- |
| Latest price / zero expected growth | Predict the origin price again | Primary point baseline; zero-drift GBM still provides a distribution |
| Momentum | Extend the recent log-price move | Simple continuation comparison |
| Mean reversion | Move toward a trailing mean log price | Simple return-to-recent-level comparison |
| Ridge | Linear future-log-return regression penalizing large slopes | Small regularized challenger using recent return features |
| Fixed ensemble | Average ridge price and latest price, 50/50 | Test reduced reliance on ridge |
| ARIMA(1,1,0) | Model log-price changes using a previous change, without drift | Classical time-series challenger |
| ETS level / damped trend | Smooth a level, optionally a trend that fades with horizon | Classical exponential-smoothing challengers |
| LightGBM regression | Combine small decision trees to predict log returns | Nonlinear price-derived-feature challenger |
| Empirical residual distribution | Shift matured past forecast errors around a new forecast | Ranges and up probabilities without changing the point model |

Momentum and mean reversion use up to 20 trailing sessions, capped by lookback.
Exponentiated predicted log returns/log prices are not automatically expected
arithmetic prices. Stock LightGBM regression uses its own price features; the
expanded OHLCV/context export feeds direction experiments.

### Running price experiments

Prepare a [benchmark manifest](forecast-benchmark.md) with real paths, hashes and
chronological splits. Run from the repository root with new output filenames:

```bash
python3 python/ridge_forecasts.py --manifest Notes-dont-commit/benchmark.json --output Notes-dont-commit/ridge-validation.json
python3 python/ensemble_forecasts.py --manifest Notes-dont-commit/benchmark.json --output Notes-dont-commit/ensemble-validation.json
python3 python/residual_forecasts.py --manifest Notes-dont-commit/benchmark.json --output Notes-dont-commit/residual-validation.json

python3 -m venv .venv
.venv/bin/python -m pip install -r python/requirements-forecast.txt -r python/requirements-ml.txt
.venv/bin/python python/classical_forecasts.py --manifest Notes-dont-commit/benchmark.json --output Notes-dont-commit/classical-validation.json
.venv/bin/python python/boosted_forecasts.py --manifest Notes-dont-commit/benchmark.json --output Notes-dont-commit/boosted-validation.json
```

These runners score validation at 1/5/20-session horizons with steps equal to
the horizon. Compare saved predictions using the [matched-date runner](matched-forecast.md),
because warmup and origin grids can differ. Reports retain point predictions,
training counts, errors, direction counts, runtime and source/input hashes.
Holdout rows do not enter training or validation scoring.

### Ridge and ensemble

Ridge uses the latest log return and trailing 5/20-session mean returns.
Each origin refits on examples beginning in its lookback, using only labels
matured by the origin. Allow a 20-return feature warmup and at least 20 matured
labels. Training-only centering/population scaling uses one for constant-feature
scales. Slopes have fixed squared penalty 1.0; the intercept is unpenalized.
No extra packages are needed.

The ensemble averages ridge and latest price in price space with fixed weights.
Its `ridge_predicted_log_return` field describes the ridge component, not the
blend. It usually retains ridge's direction even when price errors change.
The equal-weight stock/horizon MAE-ratio ranking favors latest price, then
ensemble, then ridge on ties. Both-zero groups are excluded; an error against
a perfect baseline disqualifies the candidate.

### ARIMA and exponential smoothing

The optional statsmodels runner requires at least a 30-session lookback.
At every origin it fits ARIMA(1,1,0) without drift, ETS(A,N,N) level-only, and
ETS(A,Ad,N) with damped additive trend. Inputs are log prices centered on the
last close. Flat windows return latest price; optimizer iterations are capped
at 200. There is no order or seasonal search. Failed fits disqualify the
candidate globally, with failures retained rather than silently dropped.

### LightGBM return regression

This CPU runner needs at least a 60-session lookback, 20-session feature warmup
and 40 matured labels. Its eight inputs are 1/2/5/10/20-session log returns,
5/20-return population volatility and distance from the trailing 20-close
mean log price.

Fixed settings are L1 log-return loss, 80 rounds, learning rate 0.03, depth 3,
seven leaves, minimum 20 rows/leaf, L2 penalty 1.0 and 63 bins. It uses one thread,
seed 42, deterministic mode and column-wise processing, without early stopping
or tuning. Feature gains are descriptive, not evidence of predictive skill.
The equal-weight MAE-ratio ranking favors latest price on ties.

### Empirical residual ranges

After a previous target is observed, record `log(actual)-log(forecast)`.
Keep up to 60 matured errors and require 20. At a 20-session horizon this can
need roughly 400 extra sessions after the first fit. For 80/90/95% ranges,
interpolate equal-tail quantiles at `probability*(count-1)`, exponentiate and
multiply by the new forecast. Do not recenter the errors; they may reflect bias.
The unchanged point forecast need not lie inside the resulting range.

Strict empirical exceedance frequencies provide up probabilities without
smoothing. Latest price gets its own residual distribution on the same rows.
Reports include coverage, width, interval score, Brier score and calibration
counts/dates. These are empirical ranges with limited tail resolution, not
guaranteed coverage. The [final workflow](final-forecast.md) freezes this procedure.

Native verification commands are in [tests](../tests/README.md).

## Stock direction models

These output the probability of a strictly higher target price. At probability
0.5 or above, the predicted class is up.

| Method | Simple explanation | Role |
| --- | --- | --- |
| Always-up | Predict a rise every time | Hit-rate baseline |
| Historical up frequency | Frequency of matured up outcomes in training | Probability baseline |
| Logistic regression | Convert a weighted feature sum into probability | Regularized linear classifier with training-only scaling |
| LightGBM classifier | Combine decision trees into an up probability | Nonlinear classifier |
| Optional CPU CNN | Learn short sequence patterns with a small convolutional network | Fixed-seed local neural challenger, without pretrained weights |

Inputs describe past returns, volatility, OHLC ranges, volume and SPY/QQQ/XLK
context. Per-stock fitting is default; pooled training shares examples across
stocks while splitting all assets by time. Fixed yearly folds use labels that
mature before the fitting boundary.

Selection uses a small predeclared search in earlier chronological folds.
Sigmoid calibration adjusts confidence using past out-of-fold predictions.
Selection can choose a baseline; better calibration need not mean a better hit
rate. See [direction](direction-forecast.md), [pooling](pooled-direction.md),
[selection/calibration](calibrated-direction.md) and [CNN](neural-direction.md).

Current inspected stock validation has not established a reliable edge over
the relevant baselines. Final evaluation still lacks its frozen stock exports.

## Ontario demand models

The independent energy module predicts 24 hourly Ontario demand values in MW
from an 08:00 fixed-EST origin. It retains UTC intervals and Toronto calendar
fields. Annual IESO files are revised; next-day 08:00 EST availability is assumed.

**Seasonal baselines** repeat 24- or 168-hour-old loads. If unreleased, they step
back another season until available, recording the actual source. Missing
available observations fail.

**CPU LightGBM** pools lead-hour rows using 19 calendar/available-load features.
One fixed point fit uses 2018–2021. Separate 5th/95th quantile fits and a frozen
2022 widening correction supply ranges scored in 2023 validation. Weather is
excluded. The approved 2024 final evaluation reused models/correction without
refitting; 2024 is now consumed. Incomplete 2025 remains unscored.
See [energy commands and results](../apps/energy/README.md).

## Choosing a method

Choose the target, then compare baselines and challengers on identical dates.
Fit features, scaling, selection and calibration using past information only.
Keep failed fits and inconclusive results. Assess error size, probabilities,
range coverage/width, uncertainty and runtime using the
[calculation reference](calculations.md). More complexity is a hypothesis to test.
