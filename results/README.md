# Benchmark results

The five CSVs in [final/](final/) are the reference suite: convergence, scaling,
variance reduction, instrument performance and Greek accuracy. Its
[environment record](final/environment.md) describes the workload and toolchain.

Generate plots from the benchmark CSVs:

```bash
.venv/bin/python python/plot_final_benchmarks.py --output-dir /tmp/mc-final-plots
.venv/bin/python python/plot_convergence.py --output-dir /tmp/mc-convergence-plots
.venv/bin/python python/plot_scaling.py --output-dir /tmp/mc-scaling-plots
```

## Ontario electricity demand

See the [current benchmark](../apps/energy/README.md#current-benchmark) for scores.

| Run partition | Dates | Origins | Scored hours |
| --- | --- | ---: | ---: |
| Training | March 15–December 29, 2024 and January 15–May 30, 2026 | 426 | — |
| Calibration | June 1–July 13, 2026 | 43 | — |
| Scoring | July 15–November 19, 2026 | 128 | 3,072 |
| IESO comparison | October 7–November 19, 2026 | 44 | 1,056 |

The benchmark uses archived ECMWF IFS 00 UTC forecasts for Toronto Pearson,
Ottawa, London and Thunder Bay with an assumed 12-hour release lag. June 23
uses the previous day's 18 UTC cycle because the 00 UTC run had gaps.
Revised demand uses the assumed next-day cutoff. Both the IESO XML creation
and HTTP Last-Modified timestamps had to be at or before the forecast cutoff.
Eighty-four scoring origins lacked both eligible delivery-date reports.
[Weather archive](https://open-meteo.com/en/docs/single-runs-api) and
[IESO Adequacy definitions](https://reports-public.ieso.ca/docrefs/helpfile/Adequacy3_h2.pdf).

The default and tuning runs took about 4 and 5 seconds on an Intel
i9-13900H with Python 3.12.3, NumPy 2.5.3, SciPy 1.18.1 and LightGBM 4.7.0.
The 95% intervals use 2,000 paired circular resamples of seven-day blocks
with seed 42. The operator comparison is descriptive. Normalized inputs,
provider evidence, model artifacts and matched predictions are preserved
locally in `Notes-dont-commit/energy/benchmark-2026-11-22/`.

## Focused experiments

| Files | Purpose | Recorded |
| --- | --- | --- |
| `scaling.csv` | 1M/5M/10M-path scaling across 1/2/4/8 threads | 2026-09-19 |
| `antithetic.csv` | Standard/paired sampling at 1M/5M paths and 1/2/4 threads | 2026-09-20 |
| `greeks_variance.csv` | Common versus independent random streams for Delta bumps | 2026-09-22 |
| Memory measurements below | Process-level peak RSS for terminal/path engines | 2026-09-27 |
| `thread_overhead.csv` and profile below | Worker startup cost and instruction profile | 2026-09-28 |
| `hot_loop_before.csv`, `hot_loop_after.csv` | Matched optimization comparison | 2026-09-28 |

Shared environment: Intel i9-13900H (14 cores/20 threads), 31 GiB RAM, Linux Mint
22.3/kernel 7.0.0-31-generic, GCC 14.3.0, CMake 3.28.3, C++20 Release
(`-O3 -DNDEBUG`). Scaling plots used Python 3.12.3/Matplotlib 3.11.2. Timings
were not CPU-pinned or isolated from thermal/background effects.

### Scaling and antithetic sampling

Both price the one-year ATM European call, `S=K=100`, `r=5%`, `sigma=20%`, seed 42.
Each configuration runs three times after a 100K-path warmup; medians determine
throughput and speedup. Pair averages are independent observations. Compare
estimator variance (`SE²`), not raw payoff variance: pairing reduced it about 2x
for these workloads.

### Greek variance

Delta uses a spot bump of 1 for the same ATM call. Each method has 100
replications, with two 50K-path prices per replication. Reusing seeds across
bumps gave variance `6.6856e-06`, versus `0.0028051` for independent seeds
(419.6x reduction). This depends on payoff, bump and RNG configuration.

### Memory

Measured with `/usr/bin/time -v`, eight threads, seed 42 and batches of 16,384
observations. RSS includes the whole process and varies by OS/allocator.

| Workload | Peak RSS | Runtime |
| --- | ---: | ---: |
| European call, 10M paths | 4,124 KiB | 0.116 s |
| Asian call, 1M antithetic paths, 252 steps | 4,124 KiB | 1.133 s |

### Profiling and optimization

Callgrind (Valgrind 3.22.0/libstdc++) profiled a two-million-path, single-thread
European call. `perf` was unavailable (`kernel.perf_event_paranoid=4`). Of
481,576,245 instructions, normal generation used 46.42%, terminal evolution
29.07%, Welford updates 9.14%, payoff 4.57% and discount access 1.25%.
These are instruction shares, not elapsed-time shares.

| Thread-overhead workload | 1 thread | 8 threads |
| --- | ---: | ---: |
| 8 paths | 2.628 us | 396.367 us |
| 1,000 paths | 43.340 us | 179.755 us |
| 1,000,000 paths | 42.391 ms | 10.633 ms |

Inlining small methods and caching path/discount constants preserved prices,
SE and RNG streams. Instructions fell to 449,576,174 (-6.645%).

| Threads | Before throughput | After throughput | Change |
| --- | ---: | ---: | ---: |
| 1 | 23.619M paths/s | 24.379M paths/s | +3.218% |
| 8 | 112.697M paths/s | 129.211M paths/s | +14.654% |
