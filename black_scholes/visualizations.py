from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, List

from models import Option, OptionType
from greeks import Greeks
from monte_carlo import MonteCarloResult


COLORS   = ["#2563eb", "#dc2626", "#16a34a", "#ea580c", "#7c3aed"]
FONTSIZE = 11

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.size": FONTSIZE,
})


def plot_price_vs_spot(
    base_option: Option,
    spot_range: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    if spot_range is None:
        K = base_option.strike
        spot_range = np.linspace(0.5 * K, 1.5 * K, 300)

    call_prices = []
    put_prices  = []
    call_intr   = []
    put_intr    = []

    for S in spot_range:
        c = Option(S, base_option.strike, base_option.maturity,
                   base_option.volatility, base_option.risk_free_rate, OptionType.CALL)
        p = Option(S, base_option.strike, base_option.maturity,
                   base_option.volatility, base_option.risk_free_rate, OptionType.PUT)
        call_prices.append(c.call_price())
        put_prices.append(p.put_price())
        call_intr.append(c.intrinsic_value())
        put_intr.append(p.intrinsic_value())

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        created_fig = True
    else:
        fig = ax.get_figure()
        created_fig = False

    ax.plot(spot_range, call_prices, color=COLORS[0], lw=2, label="Call (BS)")
    ax.plot(spot_range, put_prices,  color=COLORS[1], lw=2, label="Put (BS)")
    ax.plot(spot_range, call_intr,   color=COLORS[0], lw=1, ls="--",
            alpha=0.5, label="Call intrinsic")
    ax.plot(spot_range, put_intr,    color=COLORS[1], lw=1, ls="--",
            alpha=0.5, label="Put intrinsic")
    ax.axvline(base_option.strike, color="gray", ls=":", lw=1, label=f"K={base_option.strike}")

    ax.set_xlabel("Spot Price S")
    ax.set_ylabel("Option Price")
    ax.set_title(
        f"Black-Scholes Price vs Spot  "
        f"(K={base_option.strike}, T={base_option.maturity}y, "
        f"σ={base_option.volatility:.0%}, r={base_option.risk_free_rate:.0%})"
    )
    ax.legend()
    ax.set_xlim(spot_range[0], spot_range[-1])
    ax.set_ylim(bottom=0)
    if created_fig:
        fig.tight_layout()
    return fig


def plot_delta_vs_spot(
    base_option: Option,
    maturities: Optional[List[float]] = None,
    spot_range: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    if maturities is None:
        maturities = [0.08, 0.25, 0.5, 1.0, 2.0]
    if spot_range is None:
        K = base_option.strike
        spot_range = np.linspace(0.5 * K, 1.5 * K, 300)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        created_fig = True
    else:
        fig = ax.get_figure()
        created_fig = False

    for T, color in zip(maturities, COLORS):
        deltas = []
        for S in spot_range:
            opt = Option(S, base_option.strike, T, base_option.volatility,
                         base_option.risk_free_rate,
                         base_option.option_type)
            deltas.append(Greeks(opt).delta())
        ax.plot(spot_range, deltas, color=color, lw=2, label=f"T={T:.2f}y")

    ax.axvline(base_option.strike, color="gray", ls=":", lw=1)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5, label="Δ=0.5 (ATM)")
    ax.set_xlabel("Spot Price S")
    ax.set_ylabel("Delta (Δ)")
    ax.set_title(
        f"Call Delta vs Spot for Multiple Maturities  "
        f"(K={base_option.strike}, σ={base_option.volatility:.0%})"
    )
    ax.legend(title="Maturity")
    ax.set_xlim(spot_range[0], spot_range[-1])
    ax.set_ylim(-0.05, 1.05)
    if created_fig:
        fig.tight_layout()
    return fig


def plot_vega_vs_vol(
    base_option: Option,
    vol_range: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    if vol_range is None:
        vol_range = np.linspace(0.05, 0.80, 200)

    vegas = []
    for sigma in vol_range:
        opt = Option(base_option.spot, base_option.strike, base_option.maturity,
                     sigma, base_option.risk_free_rate, base_option.option_type)
        vegas.append(Greeks(opt).vega())

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        created_fig = True
    else:
        fig = ax.get_figure()
        created_fig = False

    ax.plot(vol_range * 100, vegas, color=COLORS[2], lw=2)
    ax.axvline(base_option.volatility * 100, color="gray", ls=":", lw=1,
               label=f"Base σ={base_option.volatility:.0%}")
    ax.set_xlabel("Implied Volatility (%)")
    ax.set_ylabel("Vega (per 1% vol move)")
    ax.set_title(
        f"Vega vs Implied Volatility  "
        f"(S={base_option.spot}, K={base_option.strike}, T={base_option.maturity}y)"
    )
    ax.legend()
    ax.set_xlim(vol_range[0] * 100, vol_range[-1] * 100)
    if created_fig:
        fig.tight_layout()
    return fig


def plot_mc_convergence(
    convergence_data: dict,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    n_sims        = convergence_data["n_sims"]
    mc_prices     = convergence_data["mc_prices"]
    std_errors    = np.array(convergence_data["std_errors"])
    bs_price      = convergence_data["bs_price"]
    mc_prices_arr = np.array(mc_prices)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        created_fig = True
    else:
        fig = ax.get_figure()
        created_fig = False

    ax.semilogx(n_sims, mc_prices_arr, "o-", color=COLORS[0],
                lw=1.5, ms=4, label="MC price")
    ax.fill_between(
        n_sims,
        mc_prices_arr - 1.96 * std_errors,
        mc_prices_arr + 1.96 * std_errors,
        alpha=0.2, color=COLORS[0], label="95% CI"
    )
    ax.axhline(bs_price, color=COLORS[1], lw=2, ls="--", label=f"BS price = {bs_price:.4f}")

    ax.set_xlabel("Number of Simulations (log scale)")
    ax.set_ylabel("Option Price")
    ax.set_title("Monte Carlo Convergence to Black-Scholes Price")
    ax.legend()
    if created_fig:
        fig.tight_layout()
    return fig

def plot_dashboard(
    option: Option,
    mc_convergence_data: Optional[dict] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig = plt.figure(figsize=(16, 10), layout="constrained")  
    gs  = gridspec.GridSpec(2, 2, figure=fig)                

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    plot_price_vs_spot(option, ax=ax1)
    plot_delta_vs_spot(option, ax=ax2)
    plot_vega_vs_vol(option, ax=ax3)

    if mc_convergence_data is not None:
        plot_mc_convergence(mc_convergence_data, ax=ax4)
    else:
        T_range = np.linspace(0.01, 2.0, 200)
        thetas  = []
        for T in T_range:
            opt = Option(option.spot, option.strike, T, option.volatility,
                         option.risk_free_rate, option.option_type)
            thetas.append(Greeks(opt).theta())
        ax4.plot(T_range, thetas, color=COLORS[3], lw=2)
        ax4.set_xlabel("Time to Maturity (years)")
        ax4.set_ylabel("Theta (per calendar day)")
        ax4.set_title("Theta Decay vs Time to Maturity")
        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)

    fig.suptitle(
        f"Black-Scholes Analytics Dashboard  —  "
        f"S={option.spot}, K={option.strike}, "
        f"σ={option.volatility:.0%}, r={option.risk_free_rate:.0%}",
        fontsize=14, fontweight="bold",
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig