"""Independent Black-Scholes and Monte Carlo validation implementation."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class Inputs:
    spot: float
    strike: float
    rate: float
    volatility: float
    maturity: float

    def validate(self) -> None:
        values = {
            "spot": self.spot,
            "strike": self.strike,
            "rate": self.rate,
            "volatility": self.volatility,
            "maturity": self.maturity,
        }
        for name, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.spot <= 0.0:
            raise ValueError("spot must be positive")
        if self.strike <= 0.0:
            raise ValueError("strike must be positive")
        if self.volatility < 0.0:
            raise ValueError("volatility must be nonnegative")
        if self.maturity < 0.0:
            raise ValueError("maturity must be nonnegative")


@dataclass(frozen=True)
class PricePair:
    call: float
    put: float


@dataclass(frozen=True)
class MonteCarloEstimate:
    prices: PricePair
    call_standard_error: float
    put_standard_error: float


def analytical_prices(inputs: Inputs) -> PricePair:
    inputs.validate()
    if inputs.maturity == 0.0:
        return PricePair(
            call=max(inputs.spot - inputs.strike, 0.0),
            put=max(inputs.strike - inputs.spot, 0.0),
        )

    discounted_strike = inputs.strike * math.exp(-inputs.rate * inputs.maturity)
    if inputs.volatility == 0.0:
        return PricePair(
            call=max(inputs.spot - discounted_strike, 0.0),
            put=max(discounted_strike - inputs.spot, 0.0),
        )

    volatility_time = inputs.volatility * math.sqrt(inputs.maturity)
    d1 = (
        math.log(inputs.spot)
        - math.log(inputs.strike)
        + (inputs.rate + 0.5 * inputs.volatility**2) * inputs.maturity
    ) / volatility_time
    d2 = d1 - volatility_time
    return PricePair(
        call=inputs.spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2),
        put=discounted_strike * norm.cdf(-d2) - inputs.spot * norm.cdf(-d1),
    )


def monte_carlo_prices(inputs: Inputs, paths: int, seed: int) -> MonteCarloEstimate:
    inputs.validate()
    if paths < 2:
        raise ValueError("Python validation requires at least two paths")

    generator = np.random.default_rng(seed)
    standard_normals = generator.standard_normal(paths)
    drift = (inputs.rate - 0.5 * inputs.volatility**2) * inputs.maturity
    diffusion = inputs.volatility * math.sqrt(inputs.maturity)
    terminal_prices = inputs.spot * np.exp(drift + diffusion * standard_normals)
    discount_factor = math.exp(-inputs.rate * inputs.maturity)
    call_payoffs = discount_factor * np.maximum(terminal_prices - inputs.strike, 0.0)
    put_payoffs = discount_factor * np.maximum(inputs.strike - terminal_prices, 0.0)

    return MonteCarloEstimate(
        prices=PricePair(call=float(np.mean(call_payoffs)), put=float(np.mean(put_payoffs))),
        call_standard_error=float(np.std(call_payoffs, ddof=1) / math.sqrt(paths)),
        put_standard_error=float(np.std(put_payoffs, ddof=1) / math.sqrt(paths)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spot", type=float, default=100.0)
    parser.add_argument("--strike", type=float, default=100.0)
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--volatility", type=float, default=0.20)
    parser.add_argument("--maturity", type=float, default=1.0)
    parser.add_argument("--paths", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalized_error(error: float, standard_error: float) -> str:
    if standard_error == 0.0:
        return "0.000 SE" if error == 0.0 else "undefined (zero SE)"
    return f"{error / standard_error:.3f} SE"


def main() -> None:
    args = parse_args()
    inputs = Inputs(
        spot=args.spot,
        strike=args.strike,
        rate=args.rate,
        volatility=args.volatility,
        maturity=args.maturity,
    )
    analytical = analytical_prices(inputs)
    monte_carlo = monte_carlo_prices(inputs, args.paths, args.seed)

    call_error = monte_carlo.prices.call - analytical.call
    put_error = monte_carlo.prices.put - analytical.put
    print("Independent Python validation")
    print("=============================")
    print(f"Paths:                    {args.paths:,}")
    print(f"Analytical call:          {analytical.call:.8f}")
    print(f"Monte Carlo call:         {monte_carlo.prices.call:.8f}")
    print(f"Call standard error:      {monte_carlo.call_standard_error:.8f}")
    print(
        f"Call normalized error:    {normalized_error(call_error, monte_carlo.call_standard_error)}"
    )
    print(f"Analytical put:           {analytical.put:.8f}")
    print(f"Monte Carlo put:          {monte_carlo.prices.put:.8f}")
    print(f"Put standard error:       {monte_carlo.put_standard_error:.8f}")
    print(f"Put normalized error:     {normalized_error(put_error, monte_carlo.put_standard_error)}")


if __name__ == "__main__":
    main()
