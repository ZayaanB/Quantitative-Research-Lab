# Tests

From the repository root:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Catch2 covers the C++ engine, numerical boundaries and CLI parsing. CTest also
runs Python evaluation checks and end-to-end CLI workflows when Python is
available. Downloader tests use offline fixtures, never live provider access.

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
