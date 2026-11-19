# Calculations explained

This reference explains implemented calculations and how to use them to choose
settings, assess sensitivity and compare research methods. Examples are
illustrations, not measured results. See [commands](cli.md), [models](models.md)
and [formulas](mathematics.md).

## Instruments and inputs

An instrument defines the cash payment, or payoff, being valued.

| Instrument | Payoff at expiry | Example with strike 100 | Use |
| --- | --- | --- | --- |
| European call | `max(final price - strike, 0)` | Final price 110 gives 10 | Study exposure above a fixed level at expiry |
| European put | `max(strike - final price, 0)` | Final price 90 gives 10 | Study exposure below a fixed level at expiry |
| Arithmetic Asian call | `max(average monitored price - strike, 0)` | Average price 105 gives 5 | Study a payoff tied to an average |

Asian monitoring is at equally spaced times `jT/M`, for `j=1,...,M`: today
is excluded and maturity included. European exercise is at expiry. Values
are per unit of underlying; no contract multiplier is applied.

| Input | Simple meaning | Use |
| --- | --- | --- |
| Spot | Underlying price now | Starting level for the model |
| Strike | Payoff's reference price | Defines where the instrument begins paying |
| Maturity | Time to expiry in years | Sets how long uncertainty accumulates |
| Rate | Flat continuously compounded annual rate | Discounts payoffs and sets risk-neutral growth |
| Volatility | Annualized size of random returns | Explore variability assumptions; `0.20` means 20% |
| Discount factor | `exp(-rate * maturity)` | Converts an expiry payoff into present value |
| Paths / steps | Simulated trajectories / Asian monitoring points | Paths control sampling precision; steps define monitoring |

Spot and strike must be positive. Finite negative rates, zero volatility and
zero maturity are valid. Pricing assumes zero dividends, constant volatility
and frictionless markets without transaction costs. Inputs are supplied
assumptions rather than automatically retrieved quotes.

## Pricing and simulation statistics

Monte Carlo generates possible prices, calculates and discounts each payoff,
then averages. European simulation needs only the final price. Asian simulation
keeps a running price and average instead of storing a paths-by-steps matrix.

| Output or method | Meaning | Use |
| --- | --- | --- |
| Monte Carlo price | Mean discounted payoff | Estimate model value and compare with a reference |
| Analytical price | Direct Black-Scholes European value | Check simulation under identical inputs |
| Absolute pricing error | Absolute difference from analytical value | Judge relative to SE rather than expecting exact equality |
| Independent observations | Samples used to estimate uncertainty | Interpret precision using this count |
| Sample variance | Spread of payoff observations, using divisor `n-1` | Compare estimator variability under the same workload |
| Standard deviation (SD) | Square root of variance | Express spread in original units |
| Standard error (SE) | Uncertainty in the mean, `SD/sqrt(n)` | Decide whether more simulation is needed |
| 95% sampling interval | Price plus/minus `1.96 * SE` | Assess sampling precision; a normal approximation |
| Runtime / paths per second | Elapsed time / trajectory throughput | Compare speed and precision on matched hardware/workloads |
| Antithetic variates | Average discounted payoffs driven by paired shocks `Z` and `-Z` | Test noise reduction for a fixed trajectory count |

Antithetic counts are total trajectories and must be even. One pair average is
one independent observation: 1,000 paired trajectories give 500 observations.
Below two observations, sample variance, SE and the interval are undefined.
Roughly four times as many independent observations halves SE. Sampling precision
does not measure error in model assumptions or inputs.

Welford updates and mergeable worker summaries compute statistics without
retaining all payoffs. Each worker owns a random generator. Reproduction on the
same toolchain requires the same seed, threads and full configuration; changing
threads or standard libraries can change draws. Benchmarks also investigate
scaling, variance reduction and memory/profile behavior; environment records
identify the workload and hardware behind their measurements.

| Benchmark metric | Meaning | Use |
| --- | --- | --- |
| Estimator variance | SE squared, or sample variance divided by independent observation count | Compare precision of estimated prices rather than payoff spread alone |
| Variance-reduction ratio | Standard estimator variance / antithetic estimator variance | Above 1 means pairing reduced sampling variance for the compared workload |
| Speedup | One-thread runtime / measured runtime | Assess benefit from additional workers |
| Parallel efficiency | Speedup / thread count | Assess how much ideal linear scaling is retained |
| Median runtime | Middle elapsed time across repeated measurements | Reduce sensitivity to an unusually fast or slow run |
| Relative reference error | Absolute error / absolute analytical value | Compare pricing or Greek error scales; unstable near a zero reference |
| Convergence slope | Fitted slope of log SE against log path count | Check the expected Monte Carlo trend near -0.5 |

The CLI displays throughput in millions of paths per second. Compare speed
alongside SE: a faster run can still have worse precision. Memory/profile
records describe resource use rather than forecast skill.

## Greeks

The engine bumps an input and reprices. Reusing random streams across valuations
reduces noise in the differences.

| Greek | Meaning and units | Example | Decision it supports |
| --- | --- | --- | --- |
| Delta | Option-price change per one-unit spot change | Delta 0.6: spot rising 1 changes value by about 0.6 locally | Compare sensitivity to small spot moves |
| Gamma | Delta change per one-unit spot change | Gamma 0.02: spot rising 1 changes Delta by about 0.02 locally | Judge when constant Delta becomes inadequate |
| Vega | Option-price change per one volatility percentage point | Vega 0.4: volatility rising from 20% to 21% changes value by about 0.4 | Compare sensitivity to volatility assumptions |

A local spot approximation is `change in value ≈ Delta*dS + 0.5*Gamma*dS²`.
For volatility changes expressed in percentage points, add `Vega*change in points`.
Large changes require repricing.

Default bumps are 1% of spot and `0.01` absolute volatility. Central differences
are used except near zero volatility, where Vega uses a second-order forward
difference. Smaller bumps reduce some approximation error but can amplify noise.
Compare bump settings to assess stability. Expiry payoff kinks are not smooth.

Implemented Greeks are European call/put Delta, Gamma and Vega. Asian Greeks,
Theta, Rho and Greek confidence intervals are not implemented. These sensitivities
describe model values rather than predict the stock's direction.

## Historical stock distributions

| Calculation | Meaning | Use |
| --- | --- | --- |
| Simple return | `future/current - 1` | Express a move relative to its starting level |
| Log return | `log(next/current)` | Combine multi-session changes by addition |
| Mean daily log return | Average historical log return | Estimate fitted log-growth tendency |
| Daily volatility | Standard deviation of daily log returns | Describe historical variability |
| Annualized volatility | Daily volatility times square root of trading days per year | Compare on an annual scale |
| Annualized GBM drift | Trading days per year times `(mean daily log return + daily variance/2)` | Describe expected price growth, distinct from mean log growth |
| Expected price | Fitted distribution's arithmetic mean | Summarize average future price |
| Median price | Fitted distribution's middle outcome | Summarize a typical outcome; the lognormal upper tail raises the mean |
| 80/90/95% model ranges | Central quantiles of the fitted distribution | Explore dispersion and check subsequent coverage |
| Probability above current | Fitted probability that target price exceeds origin price | Evaluate probabilistic forecasts |

The single forecast displays the 95% range; backtest rows also contain 80/90%
ranges. Horizons count trading sessions after the latest supplied row, even
when that row is stale. Historical drift can be retained, set to zero or shrunk;
forecasting never automatically changes risk-neutral pricing inputs.

## Features and fitted quantities

Features are inputs available at the forecast origin. The [feature guide](stock-features.md)
defines the exact 28-column stock export.

| Quantity | Meaning | Use |
| --- | --- | --- |
| Trailing returns / momentum | Price change over prior sessions | Test whether recent movement helps forecast later movement |
| Rolling volatility | SD of recent log returns | Describe variability; exported feature volatility uses population SD and is not annualized |
| Distance from mean log price | Log price minus its trailing average | Describe deviation from a recent level |
| Intraday range / return | Log high minus log low / log close minus log open | Describe movement using consistent OHLC prices |
| Close location | Close within low-to-high range, scaled to `[-1,1]`; flat bars use zero | Distinguish closes near daily extremes |
| Log-volume changes | Changes in `log(1+volume)` against prior/trailing levels | Describe reported activity changes |
| SPY/QQQ/XLK context | Return and volatility measures for fixed market proxies | Test whether broader market history helps |
| Training mean / scale | Feature centering and SD fitted only on training rows | Make inputs comparable without future information |
| Predicted log return | Regression output mapped to `current*exp(prediction)` | Produce a point price, not automatically an expected arithmetic price |
| Residual | Actual minus forecast, or log actual minus log forecast in the residual runner | Study bias and build ranges from matured out-of-sample errors |

Energy inputs include calendar fields, available demand lags and recent load
summaries. MW is demand power, not energy in MWh. Weather is currently excluded.

## Point-forecast errors

Here error means `forecast - actual`. Price errors use the selected price
units; energy errors use MW. Compare identical assets, origins, targets and data.

| Metric | Meaning | Use |
| --- | --- | --- |
| MAE | Mean absolute error | Typical error size; lower is better |
| RMSE | Square root of mean squared error | Emphasize large misses; lower is better |
| MAPE | Mean absolute error divided by actual, per observation | Percentage error for positive stock prices; `100-MAPE` is not accuracy |
| MASE | Mean absolute error scaled by training-window naive one-step error | Compare with historical variability; below 1 beats that scale, not necessarily the horizon-matched baseline |
| Relative MAE | Model MAE / baseline MAE | Below 1 favors the model |
| Absolute MAE improvement | Baseline MAE minus model MAE | Positive means smaller model errors |
| Relative MAE improvement | Absolute improvement / baseline MAE | Report error reduction; 20% less error is not 80% accuracy |
| Energy bias | Mean signed error | Negative is underprediction, positive overprediction; cancellation can conceal large errors |
| Energy WAPE | Total absolute error / total actual demand, as percent | Compare aggregate relative error; differs from averaging hourly percentages |
| Window peak MAE | Mean absolute difference between predicted and actual maxima per 24-hour window | Assess peak magnitude separately from hourly error; does not measure peak timing |

MASE is unavailable with zero training scale; relative improvement is unavailable
with zero baseline error. Rankings handle perfect baselines explicitly.
Equal-weight stock/horizon MAE ratios avoid one high-price stock dominating.

## Direction and probabilities

Classifiers predict up at probability 0.5 or above. Actual classes are strictly
up and not-up, so unchanged outcomes belong to not-up. Point-price direction
reports exclude unchanged forecasts and require matching positive/negative moves;
an unchanged actual is neither direction.

| Metric | Meaning | Use |
| --- | --- | --- |
| Directional accuracy / hit rate | Fraction of scored direction predictions matching outcomes | Compare with always-up on the same rows, with counts |
| Balanced accuracy | Average of up-class and not-up-class hit rates | Check both classes when one dominates; undefined with only one actual class |
| Confusion counts | True up, true not-up, false up, false not-up | Identify dominant error types |
| Brier score | Mean squared difference between probability and 0/1 outcome | Lower is better; compare with probability baselines |
| Log loss | Average negative log probability assigned to the actual class | Lower is better; strongly penalizes confident mistakes; extreme probabilities are clipped |
| Reliability/calibration buckets | Mean predicted probability versus observed up rate in each band | Check whether predictions near 60% rise about 60% of the time |
| Expected calibration error | Count-weighted absolute bucket gaps | Summarize reliability; lower is better, but depends on bins and sample size |
| Accuracy gain / Brier reduction | Model accuracy minus baseline accuracy / baseline Brier minus model Brier | Positive favors the challenger |

A constant 50% probability has Brier score 0.25. Always-up can have a high hit
rate in a rising sample without distinguishing dates. Historical up frequency
is another probability baseline. Calibration alone does not prove useful
discrimination; assess several metrics together.

## Ranges

| Metric or quantity | Meaning | Use |
| --- | --- | --- |
| Coverage | Fraction of outcomes within a range | Compare with nominal target, such as 90% |
| Mean width | Average upper minus lower endpoint | Prefer narrower ranges when coverage remains adequate |
| Interval score | Width plus a penalty for misses beyond endpoints | Compare sharpness and misses together; lower is better |
| Pinball loss | Asymmetric error for a chosen quantile | Check lower/upper energy quantiles; lower is better |
| Calibration residual count | Matured past errors used for a range | Assess how much tail information is available |
| Energy widening correction | Past-calibration quantile of `max(lower-actual, actual-upper, 0)` | Expand endpoints to improve marginal coverage; report width too |
| Quantile crossings | Raw lower prediction exceeds upper | Diagnose inconsistency; energy orders endpoints before scoring |

For a central range with total tail probability `alpha`, interval score is
`width + (2/alpha)*distance outside`, with no outside penalty on a hit.
Pinball loss at quantile `q`, with `u=actual-predicted`, is `q*u` when
`u>=0`, otherwise `(q-1)*u`.

Energy's 5th/95th quantiles use 2018–2021 fits, a frozen 2022 correction, inspected
2023 scoring and the approved 2024 final evaluation. Its 90% target is marginal
across observations, not coverage for every hour/season or all 24 hours together.
Stock residual ranges use the most recent 20–60 matured errors. Both workflows
require observed coverage checks.

## Uncertainty, data checks and research decisions

| Output | Meaning | Use |
| --- | --- | --- |
| Wilson interval | Approximate 95% uncertainty interval for a hit rate | Read alongside counts; independent-trial assumptions can be weak for dependent forecasts |
| Paired block-bootstrap interval | Resample adjacent blocks of matched forecast differences | A wholly positive gain interval supports improvement in that comparison; crossing zero is inconclusive |
| Block-size sensitivity | Repeat with longer blocks | Check dependence assumptions affect the conclusion |
| Sample/origin/class counts | Numbers of forecasts, dates and outcome classes | Check comparability and sample size |
| Recursive energy hours | Targets whose seasonal source moved back more than one season for availability | Understand what the baseline knew |
| Gap/weekend/duplicate checks | Diagnostics on dates and cadence | Investigate inputs before interpreting scores |
| Hashes, retrieval dates, versions and runtime | Identity and context of data, models, source and execution | Reproduce results and detect changes |

Guarded bootstrap comparisons require ten effective blocks. Block resampling
helps represent dependence but does not prove independence or correct for many
repeated experiments. Stock calendar checks are heuristic. Energy annual files
are revised and next-day 08:00 EST availability is assumed.

Use outputs in this order:

1. Define the question: option value, sensitivity, price, direction or demand.
2. Validate inputs, timestamps and information available at each origin.
3. Check pricing precision, or compare forecasts with a simple baseline.
4. Examine error size, large misses, probabilities and ranges together.
5. Check counts and paired uncertainty before choosing a more complex method.
6. Freeze settings before final evaluation; a scored final period is consumed.

These decisions select research methods and numerical settings. The project
does not convert metrics into trading signals or execute decisions.
