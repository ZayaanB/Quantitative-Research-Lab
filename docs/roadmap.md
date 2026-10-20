# Phase 2 roadmap

Days are logical milestones, not guaranteed calendar days or exactly one commit.
Each should leave a buildable, tested project. Start at Day 1 when requested;
this plan does not mean the future features are already implemented.

## Starting point

Working: option pricing, European Greeks CLI, stock GBM, walk-forward evaluation,
ridge/ensemble, ARIMA/ETS, LightGBM regression and a validated history downloader.
Energy has a guide but no importer or runner. Stock classification and neural
models have not been built.

The October 17 download contains AAPL/MSFT/GOOGL and SPY/QQQ/XLK daily OHLCV.
The fixed 2018–2023 comparison found no useful stock edge. Keep those outcomes;
do not tune a model until that same diagnostic looks good.

## Daily milestones

Day 1 completed October 19, 2026: [matched comparison and data checks](matched-forecast.md).
Existing model predictions and the reserved final period were preserved.

Day 2 completed October 20, 2026: [past-only stock features](stock-features.md).
OHLCV and SPY/QQQ/XLK features are ready; no new model was fitted or scored.

| Day | Build | Completion check |
| --- | --- | --- |
| 1 | Reusable matched-date comparison and data-quality checks | All models use identical origin/target rows; report bad sessions and hash inputs. Preserve the current baseline results. |
| 2 | Stock feature pipeline | Add observed OHLC ranges, volume changes, SPY/QQQ/XLK returns and volatility. Future-data mutation tests leave earlier features unchanged. |
| 3 | Direction models | Add regularized logistic regression and LightGBM classification, separate from return regression. Compare with always-up, balanced accuracy and Brier/log loss. |
| 4 | Pooled stock training | Share training across available assets and expand the universe if suitable data is available. Split all assets by date; purge labels across boundaries and disclose survivor bias. |
| 5 | Controlled selection and calibration | Use a small predeclared search inside chronological development folds. Fit calibration only on past out-of-fold predictions; report accuracy/error tradeoffs and losing models. |
| 6 | One optional neural challenger | Benchmark a small normalized 1D CNN, or LSTM if justified, on the same folds. Measure memory/runtime; retain it only if it adds evidence. No foundation-model training. |
| 7 | Energy data and baselines | Build a separate Ontario hourly-demand importer, availability rules and previous-day/week forecasts. Test timestamps, units, missing hours and DST. |
| 8 | Energy LightGBM | Predict the next 24 hours using calendar and available lagged demand. Score hourly MAE/RMSE and daily peak error against both baselines. |
| 9 | Energy uncertainty and robust evaluation | Add past-only quantile calibration; report coverage, width and pinball loss by season/hour. Add weather only if archived forecasts are available. |
| 10 | Frozen final evaluation | Lock each domain's procedure and compare with baselines on its own untouched period. Report paired uncertainty and failures; no test-driven retuning. |
| 11 | Audit and handoff | Run full/native tests and representative samples, check unused code and dependency boundaries, finish concise run guides and record reproducible results. |

The stock and energy tracks remain independent. Days 7–9 do not mean energy is
proven better; Day 6 is a bounded experiment, not a required model upgrade.
If a step needs unavailable data or new authority, report that rather than
fabricating inputs or quietly expanding scope.

## Evaluation rules

Keep three questions separate: numerical option-pricing correctness, stock
price/direction skill, and energy-demand forecast quality. Never compare their
percentages as though they were the same metric.

The downloaded stock rows from 2024 onward are a candidate final period, not a
completed final test. Check that no prior experiment used them before freezing
the procedure. Foundation-model pretraining can invalidate that candidate;
use genuinely post-checkpoint data when needed. Energy needs its own split.
After a final period is scored, it stays consumed even if a model loses.

No milestone promises 60%, 70% or any other accuracy. A challenger that loses to
a baseline is a valid research outcome, not a reason to hide the result.

## Organization

See the [tool guides](../apps/README.md), [methods index](README.md) and
[development rules](../AGENTS.md). Existing source paths and `mcprice` commands
remain stable. New energy code belongs in its own module; shared code stays
shared. A project/repository rename is a separate user choice.
