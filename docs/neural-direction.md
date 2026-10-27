# Small CNN direction experiment

One optional CPU-only challenger: a 1D CNN over 20 observed sessions of the
existing 28 features. It predicts up/not-up probabilities, not price targets.
Pricing, Greeks, fixed classifiers and the selection/calibration runner are unchanged.

## Run

Use the same frozen manifests as [pooled direction](pooled-direction.md).
Create `.venv` if needed and install the ordinary model requirements first.
For Linux, use the CPU wheel index to avoid downloading CUDA libraries:

```bash
.venv/bin/python -m pip install -r python/requirements-direction.txt
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  -r python/requirements-neural.txt

.venv/bin/python python/neural_direction.py \
  --manifest Notes-dont-commit/benchmark.json \
  --context-manifest Notes-dont-commit/stock-context.json \
  --output Notes-dont-commit/cnn-validation.json

.venv/bin/python tests/test_neural_direction.py
```

On macOS, install `requirements-neural.txt` without the Linux CPU index. The
runner still uses CPU explicitly. Use a Python version supported by your
PyTorch wheel; the measured run uses Python 3.12. See the
[official installer](https://pytorch.org/get-started/locally/). Your separate
RTX 5060 Ti is not needed or benchmarked here. No pretrained checkpoints are loaded.

## Fixed experiment

The network has 889 parameters: two 8-channel, kernel-3 convolutions with ReLU,
global averaging and one output logit. Training is 20 epochs of AdamW, learning
rate 0.001, weight decay 0.001, batch size 128, seed 42 and one CPU thread.
Rows remain ordered by date then symbol. No architecture search, shuffling,
early stopping, calibration or threshold tuning is added.

Each window ends at its origin; convolutions see only that fully observed
window. Yearly folds, training labels and test origins match the fixed pooled
runner. Labels must finish before the fold starts. Channel means/population
standard deviations come only from training windows; repeated sessions receive
their window-occurrence weight. Zero variance uses scale one.

Extra sequence history is required before the earliest training origin. If it
is missing, the run fails rather than shifting dates or trimming the workload.
Single-class training is explicitly constant; other fit failures stop the run.
Existing output files are never overwritten. The manifest's reserved period
is excluded from features and scores, though basic whole-file checks still read it.

The report includes probabilities, baselines, balanced accuracy, Brier/log loss,
reliability bins, bootstrap checks, cutoffs, normalization statistics, hashes,
library versions and hardware. RSS is process-lifetime peak memory including
Python, data and framework imports, not isolated network or GPU usage. Input
and parameter byte counts exclude optimizer state, activations and overhead.

This is a small, survivor-selected, inspected validation sample. Sequences and
stocks are dependent; probabilities are uncalibrated. Same-platform repeatability
is tested, but [PyTorch does not guarantee identical results across releases/platforms](https://docs.pytorch.org/docs/stable/notes/randomness.html).
No result automatically replaces a baseline or justifies scaling neural training.

## October 26 validation

Same ten-stock snapshot, 2018–2023 folds and dates as Day 4:

| Horizon | Samples | CNN hit rate | Always-up | CNN Brier | Past-frequency Brier |
| --- | --- | --- | --- | --- | --- |
| 1 session | 15,080 | 50.5% | 52.5% | 0.2653 | 0.2511 |
| 5 sessions | 3,010 | 52.6% | 55.9% | 0.2821 | 0.2509 |
| 20 sessions | 750 | 50.9% | 60.7% | 0.3143 | 0.2636 |

No stock/horizon had an accuracy gain across all tested bootstrap blocks.
This CNN is retained only to reproduce the negative experiment, not as a promoted
model. These results do not support expanding neural training on this setup.

The verified run took 40.5 seconds on an Intel i9-13900H (14 physical/20 logical
cores, one training thread), Linux, Python 3.12, PyTorch 2.14.1+cpu and NumPy
2.5.3. The OS exposed 31.0 GiB RAM. Peak process RSS at evaluation end, before
JSON serialization, was about 462 MiB; the largest training input was 5.62 MB
and parameters occupied 3,556 bytes. Hardware/build details are in the private
report. These local development timings are not universal performance claims.
