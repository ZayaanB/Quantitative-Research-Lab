# Development guide

## Project scope

This is a quantitative research workspace: C++20 derivatives pricing and Greeks,
historical stock scenarios, experimental stock prediction, and a separate
energy-demand forecasting project. Stock/energy forecasting is explicitly in
scope. It must not silently change risk-neutral option-pricing inputs.

No brokerage integration, execution, portfolio management or trading signals.
Forecasts are research outputs, not demonstrated trading advice. Do not claim
an accuracy improvement unless matched out-of-sample results support it.

At the start of a new interactive project conversation, ask once:
"Is this a learning project where you'd like me to use $learn-while-building?"
If yes, read and follow the skill throughout the conversation. If the preference
is already explicit, do not ask again. Silence is not agreement. Skip this check
in unattended runs unless learning mode was explicitly requested.

## Repository map

- `include/mc/`, `src/`: public C++ types, pricing, statistics, stock GBM and CLI.
- `python/`: shared evaluation utilities, stock experiments, downloads and plots.
- `apps/`: per-tool README entry points; existing shared code is not duplicated.
- `apps/energy/`: dedicated IESO hourly-demand importer and seasonal baseline runner.
- `tests/`: C++ and Python regression tests, including offline downloader tests.
- `examples/`: small synthetic fixtures and manifest templates, never real results.
- `benchmarks/`, `results/`: measured performance and reproducibility evidence.
- `docs/`: methods, workflow and the current [roadmap](docs/roadmap.md).
- `Notes-dont-commit/`: ignored raw datasets, manifests and private experiment outputs.

Do not move modules merely for appearance: Python scripts currently share imports
from `python/`, and CMake names existing source paths. Explain any necessary
architecture change first, update all callers and test it. Keep one authoritative
implementation and preserve established commands unless a migration is requested.

## Pricing and numerical contract

- Zero dividends, flat continuously compounded rates, constant volatility,
  frictionless markets and risk-neutral GBM. No transaction costs or discrete dividends.
- Validate finite inputs: positive spot/strike, nonnegative volatility/maturity,
  positive paths/threads and positive steps for path simulation. Negative rates,
  zero volatility and zero maturity are valid.
- European call/put MC prices must be checked against analytical Black-Scholes
  with statistical tolerances. Report price standard error and a 95% sampling
  interval; explain when fewer than two independent observations make them undefined.
- Antithetic path counts are total trajectories and must be even. One pair's
  average discounted payoff is one independent statistical observation.
- Asian monitoring uses jT/M for j=1,...,M, excluding today and including maturity.
  Keep incremental paths/averages; do not store a paths-by-steps matrix.
- Greek bumps are configurable: default spot bump 1% of spot, volatility bump 0.01.
  Vega is per one volatility percentage point, not per unit volatility.
- Current Greek support is European Delta/Gamma/Vega. No Asian Greeks, Theta,
  Rho or Greek confidence intervals; never imply those are implemented.
- Reproduce identical full configurations, including threads, on the same
  implementation/toolchain. `std::normal_distribution` is implementation-dependent;
  cross-thread/platform results need statistical consistency, not identical bits.
- Retain streaming/mergeable statistics, thread-local RNG and no hot-loop locks.
  Use C++20, RAII, const correctness and focused value types; avoid owning raw pointers.
- Keep core builds portable on Linux/macOS and CPU-only by default. GPU research
  dependencies are optional. Profile before optimizing and record benchmark hardware.
- Do not add pricing stretch features (Heston, barriers, CUDA, American options,
  control variates, Sobol) without a specific request.

## Forecasting and ML contract

- Stock targets use trading sessions; energy uses explicitly defined hourly
  timestamps, units, local calendar rules and reporting availability.
- Pricing and forecasts use different measures. Never present a risk-neutral
  simulation or option Greek as a stock-direction predictor.
- Fit features, scalers, tuning and calibration on training data only. Training
  labels must mature by the forecast origin and not cross fold boundaries.
- Split by time across all assets; no random train/test shuffling of financial
  histories. Predeclare windows and settings, then use a frozen final procedure.
- Compare identical assets, origins, targets and datasets. Check latest-price and
  always-up stock baselines; use previous-day/week baselines for hourly demand.
- Report price/return errors, direction, calibration/range metrics where supported,
  sample counts, runtime and uncertainty. Do not label `100 - MAPE` as accuracy.
- Preserve failures and inconclusive results. Do not silently drop difficult rows,
  change requested workloads or fall back to another model without disclosure.
- A downloaded dataset is not automatically point-in-time or survivorship-free.
  Fundamentals and weather must be available at the origin; future realized
  weather and retroactively backfilled statements are not valid predictors.
- Check foundation-model licenses, checkpoint dates and possible pretraining overlap.
  A reserved local period is not clean if a pretrained model already saw it.
- As of the October 17, 2026 experiment, 2018–2023 is inspected validation.
  2024 onward is downloaded but unscored in that comparison; treat it only as a
  candidate final period, not a completed test or replacement for the old missing holdout.
- No model family is assumed best. Keep baseline results when challengers lose.
  Implement only the requested milestone; the optional CNN is experimental.
- Energy uses fixed EST hour-ending data, UTC intervals and Toronto calendar times.
  Annual IESO files are revised, not point-in-time. The baseline's next-day 08:00
  EST availability rule is assumed, not verified historical publication timing.
  Disclose seasonal recursion for unreleased lags; never fill missing available rows.

## Safety and repository hygiene

Use `apply_patch` for edits and `rg` for searches. Preserve existing uncommitted
work; do not reset, stash, commit, push or delete user data without authorization.
Keep commits coherent, buildable and tested when commits are requested.
Name new branches `feat/zayaan/<description>`. After implementation milestones,
provide checkout, explicit staging, commit and push commands; do not run them unless asked.

Keep optional dependencies in a virtual environment; do not require system-wide
installs or CUDA. Never commit credentials, downloaded histories, caches or local
builds. Never ask users to paste API keys in chat. Retain TLS checks and respect
provider limits; do not bypass challenges or run downloaded scripts blindly.
Use new snapshot/report paths, validate data and record provider, adjustment
semantics, retrieval dates, input hashes, source revision and dependency versions.

Remove only temporary files created for the current work. Preserve datasets and
results needed to reproduce claims. Do not call old benchmarks dead code simply
because they are not part of the normal CLI. Keep docs brief and comments to
necessary one-line explanations. Do not create a new audit report unless requested.

## Verification

Before completing a code change:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

For changes touching optional models, also run native tests after installing their
documented requirements in `.venv`:

```bash
.venv/bin/python tests/test_classical_forecasts.py
.venv/bin/python tests/test_boosted_forecasts.py
```

Default CTest may skip optional model cases; a skip is not a tested native fit.
For direction-model changes, install `python/requirements-direction.txt` in
`.venv` and also run `.venv/bin/python tests/test_direction_forecasts.py`.
For selection/calibration changes, also run `.venv/bin/python tests/test_calibrated_direction.py`.
For CNN changes, install the documented CPU-only requirements and also run
`.venv/bin/python tests/test_neural_direction.py`; default CTest may skip native fits.
For energy changes, also run `python3 tests/test_energy_baselines.py` (standard library only).
For documentation-only changes, check local links, commands and `git diff --check`.
Report what actually ran, failures/skips and remaining limitations. No test suite
can justify a claim that all bugs are impossible.
