# Tools

Start from the [root build instructions](../README.md). Run every command from
the repository root, not from these folders.

- [Options](options/README.md): risk-neutral derivative pricing.
- [Greeks](greeks/README.md): European option sensitivities.
- [Forecast](forecast/README.md): historical stock-price distributions.
- [Prediction](prediction/README.md): optional statistical and ML experiments.
- [Energy](energy/README.md): Ontario hourly-demand importer and seasonal baselines.

These folders group user-facing documentation; the independent energy Python
module also lives here. They do not duplicate shared code or introduce separate
builds. Stock and energy forecasts must not silently
change the option engine's risk-neutral inputs.

Future work is split into the [Phase 2 daily roadmap](../docs/roadmap.md).
