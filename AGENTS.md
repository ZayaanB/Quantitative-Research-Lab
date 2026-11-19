# Development guide

## Scope

C++20 option pricing/Greeks, stock research and separate energy-demand forecasting.
Keep pricing and forecasting inputs separate. No brokerage integration,
execution or trading signals.

Ask once at the start of an interactive conversation:
“Is this a learning project where you'd like me to use $learn-while-building?”
Use the skill only if requested; skip when preference is explicit or unattended.

## Working rules

- Follow the [repository map](README.md#repository-layout). Preserve shared source
  paths and established commands; don't duplicate implementations.
- Use `apply_patch` for edits and `rg` for searches. Preserve uncommitted work.
  Don't reset, stash, commit, push or delete user data without authorization.
- Name branches `feat/zayaan/<description>`. After implementation, provide checkout,
  explicit staging, commit and push commands; don't run them unasked.
- Keep docs concise and comments limited to non-obvious behavior. Replace stale
  content instead of appending progress notes; no routine session/audit reports.
- Use `.venv` for optional dependencies; keep CPU-only defaults and TLS checks.
  Never commit credentials, downloaded data, caches or builds.
- Preserve datasets, saved models, results, provenance and final-attempt markers.
  Record source/retrieval details, hashes, code revision and dependencies; use
  new paths for experiment artifacts.

## Numerical and research rules

- Preserve [pricing conventions](docs/mathematics.md): validated finite inputs,
  valid zero volatility/expiry and negative rates, paired antithetic statistics,
  Asian monitoring excluding today, and Vega per volatility percentage point.
- Support European Delta/Gamma/Vega only. Check European Monte Carlo against
  analytical Black-Scholes with statistical tolerances. Sampling error is not model error.
- Keep streaming statistics, thread-local RNG and portable C++20. Reproduction
  requires the same full configuration and toolchain.
- Split data by time. Features, labels, scaling, tuning and calibration must use
  only information available at the origin. Compare matched targets and baselines.
- Report errors, coverage, counts and uncertainty; never call `100-MAPE` accuracy.
  Preserve failures; don't silently fill rows, trim workloads or switch models.
- Energy uses fixed EST hour-ending data and assumed next-day 08:00 availability.
  Preserve timezone, availability and boundary-purge checks.
- Read [research status](docs/roadmap.md) before evaluation. Consumed final periods
  stay consumed; revised history is not automatically point-in-time.
- Add features only within the requested scope; accuracy claims require matched
  out-of-sample evidence.

## Verify and finish

- Run the [required tests](tests/README.md) after code changes, including affected
  native optional-model suites; skipped fits aren't tested fits. For docs-only
  changes, validate links, shell examples and `git diff --check`.
- Use disposable builds outside the repo, cleanup traps and
  `PYTHONDONTWRITEBYTECODE=1`. Remove task-created builds, binaries, caches,
  scratch files and raw logs on success or failure. Retain them only if requested.
- Fix new warnings/errors where possible; report unresolved failures without
  suppressing diagnostics. Don't store warning/error dumps or routine test logs
  in Markdown. Preserve pre-existing files and durable research evidence.
- Check `git status --short` and `git diff --check`; summarize changes, verification,
  remaining limitations and any task-created artifacts left behind.
- After large runs, replace current tables in the component guide and metadata in
  `results/README.md` or `results/final/environment.md` (workload/counts,
  metrics, runtime/hardware, versions, limitations). Update `docs/roadmap.md` if
  conclusions or blockers change; don't append dated progress notes.
