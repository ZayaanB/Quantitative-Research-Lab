# Forecast research

Research references and proposed experiments. Current implementation and evaluation
status are in the [roadmap](roadmap.md); model procedures are in the [model guide](models.md).

## Energy research priorities

The approved 2024 final comparison reduced MAE by 20.3% versus the daily baseline,
but bias remained −224 MW and peak improvement was small. That period is consumed;
the proposals below must not be tuned on it and presented as another clean final test.

Prioritize inputs and controlled updates before model complexity:

1. **Weather and holidays.** [Ben Taieb & Hyndman](https://robjhyndman.com/publications/kaggleloadforecasting/)
   use temperature, calendar/holiday effects and past demand in hourly additive
   boosted models. For us, first test Ontario holiday/adjacent-day features;
   weather requires archived forecasts and documented publication latency.
2. **Trend/residual targets and past-only refits.** The
   [Rubattu–Maroni–Corani paper](https://arxiv.org/html/2305.05575v1)
   reports improvements from detrending, feature engineering and temporal
   reconciliation. Test changes separately, not its full heavy 30,000-round
   competition recipe. Our frozen fit stops in 2021; a scheduled rolling fit is
   a reasonable hypothesis, not an established improvement.
3. **An additive/spline challenger and simple blend.** This supplies a different
   inductive bias from LightGBM. [Forecast-combination evidence](https://otexts.com/fpp3/practical.html#forecast-combinations)
   supports testing a simple average before fitting complex blend weights.
4. **Adaptive uncertainty, not more point-accuracy claims.**
   [Gibbs & Candès](https://proceedings.neurips.cc/paper_files/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html)
   motivate online calibration under shift. Updates must wait for demand labels
   to become available; long-run coverage tracking is not per-hour or simultaneous coverage.

[Open-Meteo Single Runs](https://open-meteo.com/en/docs/single-runs-api) preserves
forecast runs, unlike a stitched historical series. Its stated coverage starts
in March 2024 for ECMWF IFS and April 2026 for other models, so it cannot simply
populate our old 2018–2021 training contract. Initialization time alone does not
prove release by our origin: verify publication latency, units, locations and
terms before use. A new weather-enabled experiment needs its own chronological
split and genuinely available forecasts, not ERA5/future realized temperatures.

These are CPU-first candidates; no evidence here establishes a best algorithm
for Ontario or Apple, or justifies buying hardware. Existing stock results do not
show a reliable price-prediction edge, and energy results do not change that.

## Papers checked

| Paper | Actual finding | What we can reasonably take from it |
| --- | --- | --- |
| [Fischer & Krauss, LSTM (2017 working paper; published 2018)](https://www.iwf.rw.fau.de/files/2015/12/11-2017.pdf) | 54.3% for selected top/bottom stocks outperforming the cross-sectional median; returns deteriorate in 2010–2015. | Evidence for pooled sequence learning, not 54.3% tomorrow-up accuracy on Apple. |
| [Gu, Kelly & Xiu, 2020](https://dachxiu.chicagobooth.edu/download/ML.pdf) | Trees/neural nets improve monthly return prediction; NN3 reaches 0.40% out-of-sample R-squared in the pooled sample. | Tiny predictive signal can matter across a large panel. Needs rich characteristics and careful timing. |
| [Jiang, Kelly & Xiu, 2023](https://doi.org/10.1111/jofi.13268) | CNNs learn from scaled OHLC/volume windows; train/validate in 1993–2000, test in 2001–2019. Their image-scaled logistic control is also useful. | Test normalization and simple classification before assuming a large network is necessary. Portfolio Sharpe is not direction accuracy. |
| [Namdari & Durrani, 2021](https://doi.org/10.1007/s43069-021-00071-2) | Reports 70.36% average directional accuracy across five discretized trend classes with technical/fundamental inputs. | Not our binary daily-up task. Fundamental availability and preprocessing would need an independent reproduction. |
| [Zhang, Zohren & Roberts, DeepLOB](https://arxiv.org/html/1808.03668v6) | 84.47% on FI-2010 setup 2, horizon 10; labels are up/stationary/down from smoothed order-book mid-prices. | Credible different task, not daily-close forecasting. Needs order-book data; the paper itself notes FI-2010 is too short for a realistic standalone evaluation. |
| [Pretrained models for financial returns, 2026 preprint](https://arxiv.org/html/2606.27100v1) | Foundation models often beat locally trained neural alternatives, but gains over random walk are sparse. Raw significant comparisons are exploratory. | Chronos/TimesFM are challengers, not proven universal stock predictors. Small test samples and pretraining overlap remain concerns. |
| [Rubattu, Maroni & Corani, 2023](https://arxiv.org/html/2305.05575v1) | Load competition: lagged features reduce qualification error; probabilistic LightGBM and temporal hierarchies improve uncertainty scores. | Supports a separate energy-demand experiment, not stock accuracy claims. Competition horizons differ from our proposed next-24-hour task. |

The GKX authors studied almost 30,000 stocks and 94 firm characteristics, plus
macroeconomic interactions and industry information. That is very different from
our few hundred trailing labels and eight price-derived inputs. Their
[public resources](https://dachxiu.chicagobooth.edu/) include data and simulation
examples; reproducing source characteristics can require licensed databases.
Do not execute third-party notebooks or unpickle downloads blindly.

The CNN paper also reports useful one-dimensional CNN and logistic alternatives.
Its randomly selected training/validation images are confined to the early
sample, with a later chronological test. We should still use purged temporal
validation in our own overlapping-window experiments.

## Price-model comparison

The fixed-settings comparison used AAPL/MSFT/GOOGL, a 252-return lookback,
2018–2023 validation, and 1/5/20-session targets. C++ dense forecast rows were
filtered to exactly the LightGBM origin/target dates before comparison. No
model settings were tuned from these results. The 2024 onward rows were
downloaded but not fitted or scored; reserve them for a separately frozen final
procedure. The original stock final exports remain missing.

GBM's equal-weight asset/horizon mean MAE ratio was 1.00067; LightGBM's was
1.06126, where unchanged-price baseline is 1.0. Neither demonstrates a useful
edge. Always-up beat both on aggregate direction at every horizon. Confidence
intervals must account for dependence; current block choices are heuristic and
multiple model searches are not corrected.

Results from different periods, lookbacks and adjustment snapshots are not directly
comparable. The OHLCV/context features feed the separate direction experiments;
the price regressor remains price-only. Provider adjustments and exchange-session
completeness are not independently verified. Historical adjusted prices do not
supply point-in-time fundamentals.
