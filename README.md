# Monte Carlo Options Engine

C++20 option pricing and Greeks, with separate stock-prediction experiments
and separate hourly-demand baselines.
No execution or brokerage integration. Forecasts are research outputs, not
demonstrated trading signals.

## Build and test

You need CMake 3.20+ and a C++20 compiler. Catch2 is downloaded at configure time
if it is not installed. Linux and macOS are the initial targets.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

## Pick a tool

All commands run from the repository root. Each folder has its own quick start.

| Area | What it does | Status |
| --- | --- | --- |
| [Options engine](apps/options/README.md) | European/Asian pricing, sampling error and runtime | Implemented |
| [Greek calculator](apps/greeks/README.md) | European Delta, Gamma and Vega | Implemented |
| [Stock forecast](apps/forecast/README.md) | Historical GBM scenarios and walk-forward checks | Implemented |
| [Stock prediction](apps/prediction/README.md) | Ridge, ensemble, ARIMA/ETS and LightGBM experiments | Experimental |
| [Energy forecast](apps/energy/README.md) | Ontario hourly-demand importer and seasonal baselines | Retrospective experiment |

Forecast and prediction are two workflows, not different promises of accuracy.
Neither has established a reliable stock-price edge over the latest-price
baseline in the current multi-stock validation. Energy remains a separate experiment,
not an assumed accuracy upgrade.

```bash
./build/mcprice price --type call --paths 1000000 --antithetic --greeks
./build/mcprice greeks --type put --paths 1000000 --antithetic
./build/mcprice forecast --csv examples/sample_prices.csv --horizon-days 20
./build/mcprice --help
```

The app folders are documentation entry points; the separate energy importer
and runner also live in `apps/energy/`. Shared implementation stays in
`include/mc/`, `src/` and `python/`; tests stay in `tests/`. Private datasets and
experiment outputs belong in the ignored `Notes-dont-commit/` folder.

Next steps: [Phase 2 daily roadmap](docs/roadmap.md).
Contributor rules: [AGENTS.md](AGENTS.md). More guides: [documentation](docs/README.md),
[Python tools](python/README.md) and [tests](tests/README.md).

## Pricing assumptions

Zero dividends, flat continuously compounded rates, constant volatility,
frictionless markets, no transaction costs and risk-neutral GBM. Negative rates,
zero volatility and zero maturity are supported. Asian monitoring excludes
today and includes maturity. Antithetic paths must be even; each pair is one
independent statistical observation.

Price confidence intervals describe Monte Carlo sampling error, not model risk.
Greeks also have bump error and currently have no confidence intervals. Vega is
per one volatility percentage point. Identical full configurations reproduce on
the same toolchain; `std::normal_distribution` differs across implementations.

## Results and details

Published [benchmarks](results/final/) describe one Intel Core i9-13900H laptop,
not universal performance. See [architecture](docs/architecture.md),
[mathematics](docs/mathematics.md) and the [forecast research](docs/forecast-model-research.md).

To reproduce benchmark plots:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r python/requirements.txt
./build/mc_final_benchmarks results/final
.venv/bin/python python/plot_final_benchmarks.py
```
