# Local research dashboard

Build the C++ engine once from the repository root, then start the dashboard.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build build --target mcprice -j 4
python3 apps/demo/serve.py
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). Use Ctrl+C to stop the server.
The dashboard serves assets and data on loopback. It needs Python 3.9+, the
system time-zone database, and a built `mcprice` executable. There are no
dashboard packages, remote assets, or data downloads.

- **Options & Greeks** streams seven complete native runs with increasing path
  counts. The final checkpoint uses the full requested path count. The figures
  include native engine time and throughput at each checkpoint and total wall
  time across all checkpoints and European Greek revaluations. The Asian paths
  are separate illustrative draws under the same assumptions.
- **Stock Scenarios** runs the CLI against the supplied stock history. Its fan
  chart shows pointwise 95% model intervals. The backtest reveals actual target
  prices and compares GBM against the latest-price baseline.
- **Stock Prediction** replays a saved calibrated direction report. With no such
  report, it runs the repository's ridge model on deterministic synthetic history
  and replays those price forecasts.
- **Ontario Demand** replays saved LightGBM validation forecasts against actual
  demand and the daily baseline, with calibrated ranges where available. With no
  report, it runs the repository's daily and weekly baselines on synthetic demand.

The stock and energy tabs automatically look for the local validation artifacts
named in `serve.py`. Specify an alternate calibrated stock report or energy
uncertainty report with `--stock-report` and `--energy-report`. Use
`--synthetic` to skip auto-detection, `--history` to supply another chronological
`Date,Adj Close` CSV, and `--binary` or `--port` to override the launch defaults.
Paths are resolved on the local machine. Replay speed controls the reveal rate.
