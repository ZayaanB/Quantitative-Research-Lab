# Monte Carlo Options Engine

C++20 option pricing and Greeks, with Python stock research and a separate
Ontario electricity-demand forecasting tool. Everything runs locally on CPU;
you supply market assumptions or historical data files.

## Project parts

| Part | What it does | Guide |
| --- | --- | --- |
| Pricing and Greeks | Black-Scholes and Monte Carlo European call/put values, arithmetic Asian calls, and European Delta/Gamma/Vega | [CLI](docs/cli.md#pricing-and-greeks) |
| Stock scenarios | Historical GBM price distributions and walk-forward baseline comparisons | [Forecast workflow](docs/forecast-workflow.md) |
| Stock prediction | Regression and direction models, pooled training, calibration and experimental CNN | [Prediction README](apps/prediction/README.md) |
| Energy forecasting | IESO hourly demand, seasonal baselines, CPU LightGBM and calibrated ranges | [Energy README](apps/energy/README.md) |

Pricing uses risk-neutral assumptions; historical forecasts use a separate
model and never automatically change pricing inputs. These are research tools
without brokerage integration, execution or portfolio management. Evaluation
compares errors, probabilities and ranges against baselines; see
[current evidence and remaining work](docs/roadmap.md).

## Build and run

Use CMake 3.20+ and a C++20 compiler on Linux or macOS. CMake downloads Catch2
if it is not installed. Run from the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
./build/mcprice price --type call --antithetic --greeks
./build/mcprice forecast --csv examples/sample_prices.csv --horizon-days 20
./build/mcprice --help
```

The example history is synthetic. C++ pricing needs no Python packages;
optional Python dependencies go in `.venv`. Component guides contain setup,
commands and model details.

## Repository layout

| Location | Contents |
| --- | --- |
| `include/mc/` | Public C++ types and interfaces |
| `src/` | Pricing, Greeks, models, statistics, forecasting and CLI implementation |
| `python/` | Shared stock experiments, downloads, evaluation and plotting |
| `apps/prediction/` | Stock-prediction setup README |
| `apps/energy/` | Independent demand-forecasting code and README |
| [docs/](docs/README.md) | CLI, calculations, models, architecture and research guides |
| [tests/](tests/README.md) | C++/Python regression tests and verification commands |
| `examples/` | Synthetic fixtures and manifest templates |
| `benchmarks/` | Performance and numerical experiments |
| [results/](results/README.md) | Benchmark CSVs, summaries and reproducibility metadata |
| `Notes-dont-commit/` | Ignored private datasets, saved models and research outputs |
| `.github/workflows/` | Continuous integration |

Start with [calculations explained](docs/calculations.md) for plain-language
metrics, [models](docs/models.md) for algorithms, or [mathematics](docs/mathematics.md)
for formulas. Contributor instructions are in [AGENTS.md](AGENTS.md).
