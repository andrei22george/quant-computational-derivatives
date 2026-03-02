from __future__ import annotations

import numpy as np
from typing import Optional, Tuple
import warnings

from models import Option, OptionType


_VOL_MIN   = 1e-6
_VOL_MAX   = 10.0       # extreme upper bound
_MAX_ITER  = 100
_TOL_PRICE = 1e-8       # convergence criterion: price residual


def _bs_initial_guess(option_price: float, S: float, T: float) -> float:
    guess = np.sqrt(2.0 * np.pi / T) * (option_price / S)
    return float(np.clip(guess, 0.01, 5.0))


def _newton_raphson(
    option: Option,
    target_price: float,
    sigma0: float,
    tol: float = _TOL_PRICE,
    max_iter: int = _MAX_ITER,
) -> Tuple[Optional[float], int, str]:
    sigma = sigma0

    for i in range(max_iter):
        trial = Option(
            spot=option.spot, strike=option.strike, maturity=option.maturity,
            volatility=sigma, risk_free_rate=option.risk_free_rate,
            option_type=option.option_type,
        )
        price_diff = trial.price() - target_price

        vega = trial.spot * _phi(trial.d1()) * np.sqrt(trial.maturity)

        if abs(price_diff) < tol:
            return sigma, i + 1, "converged"

        if vega < 1e-10: 
            return None, i + 1, "failed_zero_vega"

        sigma -= price_diff / vega
        sigma  = float(np.clip(sigma, _VOL_MIN, _VOL_MAX))

    return None, max_iter, "failed_max_iter"


def _bisection(
    option: Option,
    target_price: float,
    sigma_lo: float = _VOL_MIN,
    sigma_hi: float = _VOL_MAX,
    tol: float = _TOL_PRICE,
    max_iter: int = 200,
) -> Tuple[Optional[float], int, str]:
    def residual(sigma: float) -> float:
        trial = Option(
            spot=option.spot, strike=option.strike, maturity=option.maturity,
            volatility=sigma, risk_free_rate=option.risk_free_rate,
            option_type=option.option_type,
        )
        return trial.price() - target_price

    f_lo = residual(sigma_lo)
    f_hi = residual(sigma_hi)

    if f_lo * f_hi > 0:
        return None, 0, "failed_no_bracket"

    for i in range(max_iter):
        sigma_mid = 0.5 * (sigma_lo + sigma_hi)
        f_mid = residual(sigma_mid)

        if abs(f_mid) < tol or (sigma_hi - sigma_lo) < _VOL_MIN:
            return sigma_mid, i + 1, "converged"

        if f_lo * f_mid < 0:
            sigma_hi = sigma_mid
            f_hi     = f_mid
        else:
            sigma_lo = sigma_mid
            f_lo     = f_mid

    return 0.5 * (sigma_lo + sigma_hi), max_iter, "converged_approx"


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    maturity: float,
    risk_free_rate: float,
    option_type: OptionType,
    tol: float = 1e-8,
    max_iter: int = 100,
    return_diagnostics: bool = False,
) -> float | dict:
    df = np.exp(-risk_free_rate * maturity)
    if option_type == OptionType.CALL:
        lb = max(spot - strike * df, 0.0)
        ub = spot
    else:
        lb = max(strike * df - spot, 0.0)
        ub = strike * df

    if market_price < lb - tol or market_price > ub + tol:
        raise ValueError(
            f"market_price={market_price:.4f} violates arbitrage bounds "
            f"[{lb:.4f}, {ub:.4f}] for {option_type.value}"
        )

    template = Option(
        spot=spot, strike=strike, maturity=maturity,
        volatility=0.20,  # placeholder
        risk_free_rate=risk_free_rate, option_type=option_type,
    )

    sigma0 = _bs_initial_guess(market_price, spot, maturity)

    sigma_nr, iters_nr, status_nr = _newton_raphson(
        template, market_price, sigma0, tol=tol, max_iter=max_iter
    )

    if status_nr == "converged":
        if return_diagnostics:
            return {"iv": sigma_nr, "method": "newton_raphson",
                    "iterations": iters_nr, "status": status_nr}
        return sigma_nr

    warnings.warn(
        f"Newton-Raphson did not converge ({status_nr}), falling back to bisection.",
        RuntimeWarning, stacklevel=2,
    )
    sigma_bs, iters_bs, status_bs = _bisection(template, market_price, tol=tol)

    if sigma_bs is not None and "converged" in status_bs:
        if return_diagnostics:
            return {"iv": sigma_bs,
                    "method": "bisection",
                    "iterations": iters_nr + iters_bs,
                    "status": status_bs,
                    "newton_status": status_nr}
        return sigma_bs

    raise ValueError(
        f"Implied volatility solver failed. "
        f"Newton: {status_nr}. Bisection: {status_bs}. "
        f"Check that market_price={market_price:.4f} is valid."
    )


def _phi(x: float) -> float:
    return np.exp(-0.5 * x ** 2) / np.sqrt(2.0 * np.pi)