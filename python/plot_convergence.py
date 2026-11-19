"""Plot Monte Carlo price and error convergence from the C++ experiment CSV."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mc-options-engine-matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ConvergenceData:
    paths: np.ndarray
    monte_carlo_prices: np.ndarray
    analytical_prices: np.ndarray
    absolute_errors: np.ndarray
    standard_errors: np.ndarray


def read_convergence(path: Path) -> ConvergenceData:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"no convergence rows found in {path}")

    paths = np.asarray([int(row["paths"]) for row in rows], dtype=np.int64)
    if np.any(paths <= 0) or np.any(np.diff(paths) <= 0):
        raise ValueError("path counts must be positive and strictly increasing")

    return ConvergenceData(
        paths=paths,
        monte_carlo_prices=np.asarray([float(row["mc_price"]) for row in rows]),
        analytical_prices=np.asarray([float(row["analytical_price"]) for row in rows]),
        absolute_errors=np.asarray([float(row["absolute_error"]) for row in rows]),
        standard_errors=np.asarray([float(row["standard_error"]) for row in rows]),
    )


def plot_prices(data: ConvergenceData, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.errorbar(
        data.paths,
        data.monte_carlo_prices,
        yerr=1.96 * data.standard_errors,
        marker="o",
        capsize=4,
        label="Monte Carlo estimate with 95% CI",
    )
    axis.axhline(
        data.analytical_prices[0],
        color="black",
        linestyle="--",
        label="Analytical Black-Scholes",
    )
    axis.set_xscale("log")
    axis.set_xlabel("Simulated paths")
    axis.set_ylabel("European call price")
    axis.set_title("Monte Carlo price convergence")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_errors(data: ConvergenceData, output_path: Path) -> None:
    positive_floor = np.finfo(float).tiny
    observed_errors = np.maximum(data.absolute_errors, positive_floor)
    reference = data.standard_errors[0] * np.sqrt(data.paths[0] / data.paths)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.loglog(data.paths, observed_errors, marker="o", label="Absolute pricing error")
    axis.loglog(data.paths, data.standard_errors, marker="s", label="Estimated standard error")
    axis.loglog(data.paths, reference, linestyle="--", label=r"$N^{-1/2}$ reference")
    axis.set_xlabel("Simulated paths")
    axis.set_ylabel("Error")
    axis.set_title("Monte Carlo error convergence")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/final/convergence.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = read_convergence(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    price_path = args.output_dir / "convergence_price.png"
    error_path = args.output_dir / "convergence_error.png"
    plot_prices(data, price_path)
    plot_errors(data, error_path)
    fitted_slope = np.polyfit(np.log(data.paths), np.log(data.standard_errors), 1)[0]
    print(f"Wrote {price_path}")
    print(f"Wrote {error_path}")
    print(f"Fitted standard-error slope: {fitted_slope:.4f} (ideal: -0.5000)")


if __name__ == "__main__":
    main()
