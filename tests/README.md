# Tests

Run from the repository root in a disposable build:

```bash
(
  set -e
  export PYTHONDONTWRITEBYTECODE=1
  verification_build="$(mktemp -d /tmp/mc-options-engine-build.XXXXXX)"
  trap 'rm -rf -- "$verification_build"' EXIT
  cmake -S . -B "$verification_build"
  cmake --build "$verification_build"
  ctest --test-dir "$verification_build" --output-on-failure
)
```

Disable bytecode caches for native Python checks too. Remove task-created scratch
files and raw logs; keep reproducible research artifacts separate.

Run the offline repository check with `python3 tests/check_repository.py`.
CI checks local Markdown files/headings and rejects tracked generated or ignored
files. External URLs are skipped; raw benchmark CSVs remain allowed. New untracked
Markdown is checked locally, while pending deletions are excluded.

Catch2 covers the C++ engine, numerical boundaries and CLI parsing. CTest also
runs Python evaluation checks and end-to-end CLI workflows when Python is
available. Downloader tests use offline fixtures, never live provider access.
Stock calendars/price files and IESO report fixtures are shared in
`stock_fixtures.py` and `energy_fixtures.py`; keep assertions in the individual suites.

Optional model cases may skip without native dependencies. Exercise real fits
in the virtual environment:

```bash
.venv/bin/python -m pip install -r python/requirements-forecast.txt -r python/requirements-ml.txt
.venv/bin/python tests/test_classical_forecasts.py
.venv/bin/python tests/test_boosted_forecasts.py
```

Quantitative tests compare with analytical values or meaningful baselines.
Future-data mutation tests check for leakage. A passing suite supports the
implementation; it does not establish predictive accuracy or absence of all bugs.

Energy importer/baseline tests need only the standard library and OS zone data:
`python3 tests/test_energy_baselines.py`. They cover DST, hour endings, missing
data, availability, recursive seasonal sources, provenance and offline downloads.
For the CPU energy model, install `apps/energy/requirements.txt` in `.venv` and
run `.venv/bin/python tests/test_energy_lightgbm.py` to exercise native fits too.
For demand ranges, also run `.venv/bin/python tests/test_energy_uncertainty.py`;
it checks rank/metrics, boundary purges, unchanged point fits and future-label mutation.
For the final workflow, run `.venv/bin/python tests/test_energy_final.py`.
Synthetic fixtures test frozen inference, hash/missing-row guards, paired intervals
and a retained used marker after failure; they do not score the real final period.

Linux CI runs native classical/boosted stock and energy tests after installing
their optional requirements. Core CTest runs on Linux/macOS. For direction,
calibration and CNN native tests, follow their [guides](../docs/README.md).
