# Black-Scholes Options Pricing Engine

This repository contains a mathematically rigorous implementation of the
Black-Scholes model using pure Python (numpy/scipy only).

## Underlying Theory

### 1. Geometric Brownian Motion

The model assumes the underlying asset price follows:

$$dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$$

Under the **risk-neutral measure** Q (from Girsanov's theorem, μ → r):

$$S_T = S_0 \exp\!\left[\left(r - \tfrac{\sigma^2}{2}\right)T + \sigma\sqrt{T}\,Z\right], \quad Z \sim \mathcal{N}(0,1)$$

### 2. Closed-Form Solution

$$C = S\,\Phi(d_1) - K e^{-rT}\,\Phi(d_2)$$
$$P = K e^{-rT}\,\Phi(-d_2) - S\,\Phi(-d_1)$$

$$d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

- __Φ(d₂)__: risk-neutral probability of exercise Q(S_T > K)
- **S·Φ(d₁)**: present value of the asset conditional on exercise

## Project Structure

```ini
black_scholes/
├── models.py         — Option dataclass, BS pricing formulas
├── greeks.py         — Delta, Gamma, Vega, Theta, Rho + FD validation
├── implied_vol.py    — Newton-Raphson + bisection IV solver
├── monte_carlo.py    — GBM simulation, antithetic variates, convergence
├── visualizations.py — 4-panel analytics dashboard
└── tests/
    ├── test_models.py
    ├── test_greeks.py
    └── test_implied_vol.py
```

## Installation

For easier replication, create and activate a Conda environment:

```ini
conda create -n black_scholes_env python=3.12 -y
conda activate black_scholes_env
```

Install the required packages using pip:

```bash
pip install numpy scipy matplotlib pytest
```

## Quick Start

```python
from models import Option, OptionType
from greeks import Greeks
from implied_vol import implied_volatility
from monte_carlo import MonteCarlopricing

# Price an ATM call
opt = Option(spot=100, strike=100, maturity=1.0,
             volatility=0.20, risk_free_rate=0.05,
             option_type=OptionType.CALL)

print(opt.summary())

# Greeks
g = Greeks(opt)
print(g.all_greeks())

# Implied volatility
iv = implied_volatility(market_price=10.45, spot=100, strike=100,
                         maturity=1.0, risk_free_rate=0.05,
                         option_type=OptionType.CALL)
print(f"Implied Vol: {iv:.2%}")

# Monte Carlo validation
from monte_carlo import MonteCarlopricing
pricer = MonteCarlopricing(n_simulations=500_000, seed=42)
result = pricer.price(opt)
print(result)
```

## Run Tests

In order to run the tests, simply uese pytest and the tests path:

```bash
pytest tests/ -v
```

## Model Assumptions and Limitations

The following assumptions are to be taken into account for this implementation:

| Project assumption | In practice | Impact of the assumption |
|---|---|---|
| Constant σ | Volatility smile / surface | Misprices OTM options |
| GBM log-normal returns | Fat tails, jumps (Merton) | Underprices tail risk |
| Constant r | Stochastic rates (Hull-White) | Material for long-dated options |
| No dividends | Discrete dividends common | Use dividend-adjusted BS |
| No transaction costs | Friction exists | Limits delta-hedging frequency |
| European exercise | American options exist | American > European (puts) |

## References

1. Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. Journal of Political Economy, 81(3), 637–654. https://doi.org/10.1086/260062
2. Hull, J. C. (2022). Options, futures, and other derivatives (11th ed.). Pearson.
3. Shreve, S. E. (2004). Stochastic Calculus for Finance II: Continuous-time models. Springer.
4. Gatheral, J. (2006). The volatility surface: A practitioner's guide. John Wiley & Sons.