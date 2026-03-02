from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from models import Option, OptionType


@dataclass
class MonteCarloResult:
    price:           float          # MC price estimate
    std_error:       float
    ci_lower:        float          # confidence interval lower bound
    ci_upper:        float          # confidence interval upper bound
    bs_price:        float          # analytical Black-Scholes price
    abs_error:       float          # abs(MC - BS)
    rel_error_pct:   float          # abs(MC - BS) / BS * 100
    n_simulations:   int            # number of paths used
    antithetic:      bool           # antithetic variates usage
    convergence_data: Optional[dict] = field(default=None, repr=False)

    def __repr__(self) -> str:
        return (
            f"MonteCarloResult(\n"
            f"  MC price  = {self.price:.6f}\n"
            f"  BS price  = {self.bs_price:.6f}\n"
            f"  Abs error = {self.abs_error:.6f}\n"
            f"  95% CI    = [{self.ci_lower:.6f}, {self.ci_upper:.6f}]\n"
            f"  Std error = {self.std_error:.6f}\n"
            f"  N sims    = {self.n_simulations:,}\n"
            f")"
        )


class MonteCarlopricer:

    def __init__(
        self,
        n_simulations: int = 100_000,
        seed: Optional[int] = 42,
        antithetic: bool = True,
    ) -> None:
        self.n_simulations = n_simulations
        self.seed          = seed
        self.antithetic    = antithetic

    def _simulate_terminal_prices(self, option: Option) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        S, K, T, sigma, r = (
            option.spot, option.strike, option.maturity,
            option.volatility, option.risk_free_rate,
        )
        drift      = (r - 0.5 * sigma ** 2) * T
        diffusion  = sigma * np.sqrt(T)

        if self.antithetic:
            n_half = self.n_simulations // 2
            Z      = rng.standard_normal(n_half)
            Z_all  = np.concatenate([Z, -Z])
        else:
            Z_all = rng.standard_normal(self.n_simulations)

        return S * np.exp(drift + diffusion * Z_all)

    def price(self, option: Option) -> MonteCarloResult:
        S_T      = self._simulate_terminal_prices(option)
        K        = option.strike
        df       = option._df

        if option.option_type == OptionType.CALL:
            payoffs = np.maximum(S_T - K, 0.0)
        else:
            payoffs = np.maximum(K - S_T, 0.0)

        discounted = df * payoffs
        mc_price   = float(np.mean(discounted))
        std_err    = float(np.std(discounted, ddof=1) / np.sqrt(len(discounted)))

        bs_price   = option.price()
        abs_err    = abs(mc_price - bs_price)
        rel_err    = (abs_err / bs_price * 100) if bs_price > 1e-10 else np.nan

        return MonteCarloResult(
            price         = mc_price,
            std_error     = std_err,
            ci_lower      = mc_price - 1.96 * std_err,
            ci_upper      = mc_price + 1.96 * std_err,
            bs_price      = bs_price,
            abs_error     = abs_err,
            rel_error_pct = rel_err,
            n_simulations = self.n_simulations,
            antithetic    = self.antithetic,
        )

    def convergence_study(
        self,
        option: Option,
        n_steps: int = 20,
        n_min: int = 100,
        n_max: Optional[int] = None,
    ) -> dict:
        if n_max is None:
            n_max = self.n_simulations

        n_values   = np.unique(np.logspace(
            np.log10(n_min), np.log10(n_max), n_steps
        ).astype(int))

        mc_prices  = []
        std_errors = []

        for n in n_values:
            pricer = MonteCarlopricer(n_simulations=n, seed=self.seed,
                                      antithetic=self.antithetic)
            result = pricer.price(option)
            mc_prices.append(result.price)
            std_errors.append(result.std_error)

        return {
            "n_sims":     n_values.tolist(),
            "mc_prices":  mc_prices,
            "std_errors": std_errors,
            "bs_price":   option.price(),
        }