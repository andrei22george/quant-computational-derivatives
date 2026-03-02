from models import Option, OptionType
from monte_carlo import MonteCarlopricer
from visualizations import (
    plot_price_vs_spot,
    plot_delta_vs_spot,
    plot_vega_vs_vol,
    plot_mc_convergence,
    plot_dashboard,
)
import matplotlib.pyplot as plt

# define a sample option here
# can modify the parameters to test different scenarios
opt = Option(
    spot=100,
    strike=100,
    maturity=1.0,
    volatility=0.20,
    risk_free_rate=0.05,
    option_type=OptionType.CALL,
)

pricer = MonteCarlopricer(n_simulations=100_000, seed=42, antithetic=True)
conv   = pricer.convergence_study(opt, n_steps=30, n_min=100)

plot_price_vs_spot(opt)
plt.show()

plot_delta_vs_spot(opt)
plt.show()

plot_vega_vs_vol(opt)
plt.show()

plot_mc_convergence(conv)
plt.show()

plot_dashboard(opt, mc_convergence_data=conv)
plt.show()