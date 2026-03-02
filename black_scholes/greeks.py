from __future__ import annotations

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Dict

from models import Option, OptionType


@dataclass
class Greeks:
    option: Option

    def _phi_d1(self) -> float:
        return norm.pdf(self.option.d1())

    def delta(self) -> float:
        d1 = self.option.d1()
        if self.option.option_type == OptionType.CALL:
            return norm.cdf(d1)
        return norm.cdf(d1) - 1.0

    def gamma(self) -> float:
        S, sigma, T = self.option.spot, self.option.volatility, self.option.maturity
        return self._phi_d1() / (S * sigma * np.sqrt(T))
    
    def vega(self) -> float:
        S, T = self.option.spot, self.option.maturity
        raw_vega = S * self._phi_d1() * np.sqrt(T)
        return raw_vega / 100.0 

    def vega_raw(self) -> float:
        return self.option.spot * self._phi_d1() * np.sqrt(self.option.maturity)

    def theta(self) -> float:
        S, K, T, sigma, r = (
            self.option.spot, self.option.strike, self.option.maturity,
            self.option.volatility, self.option.risk_free_rate,
        )
        d1, d2 = self.option.d1(), self.option.d2()
        df = self.option._df

        decay_term = -(S * self._phi_d1() * sigma) / (2.0 * np.sqrt(T))

        if self.option.option_type == OptionType.CALL:
            rate_term = -r * K * df * norm.cdf(d2)
        else:
            rate_term = +r * K * df * norm.cdf(-d2)

        annualised_theta = decay_term + rate_term
        return annualised_theta / 365.0 

    def theta_annualised(self) -> float:
        return self.theta() * 365.0

    def rho(self) -> float:
        K, T, r = self.option.strike, self.option.maturity, self.option.risk_free_rate
        d2 = self.option.d2()
        df = self.option._df

        if self.option.option_type == OptionType.CALL:
            raw_rho = K * T * df * norm.cdf(d2)
        else:
            raw_rho = -K * T * df * norm.cdf(-d2)

        return raw_rho / 100.0  

    def all_greeks(self) -> Dict[str, float]:
        return {
            "delta": round(self.delta(), 6),
            "gamma": round(self.gamma(), 6),
            "vega":  round(self.vega(), 6),
            "theta": round(self.theta(), 6),
            "rho":   round(self.rho(), 6),
        }

    def fd_delta(self, bump: float = 0.01) -> float:
        opt = self.option
        up   = Option(opt.spot + bump, opt.strike, opt.maturity,
                      opt.volatility, opt.risk_free_rate, opt.option_type)
        down = Option(opt.spot - bump, opt.strike, opt.maturity,
                      opt.volatility, opt.risk_free_rate, opt.option_type)
        return (up.price() - down.price()) / (2 * bump)

    def fd_gamma(self, bump: float = 0.01) -> float:
        opt = self.option
        up   = Option(opt.spot + bump, opt.strike, opt.maturity,
                      opt.volatility, opt.risk_free_rate, opt.option_type)
        down = Option(opt.spot - bump, opt.strike, opt.maturity,
                      opt.volatility, opt.risk_free_rate, opt.option_type)
        return (up.price() - 2 * opt.price() + down.price()) / bump ** 2

    def fd_vega(self, bump: float = 0.001) -> float:
        opt = self.option
        up   = Option(opt.spot, opt.strike, opt.maturity,
                      opt.volatility + bump, opt.risk_free_rate, opt.option_type)
        down = Option(opt.spot, opt.strike, opt.maturity,
                      opt.volatility - bump, opt.risk_free_rate, opt.option_type)
        return (up.price() - down.price()) / (2 * bump * 100)   # per 1%

    def fd_theta(self, bump: float = 1/365) -> float:
        opt  = self.option
        down = Option(opt.spot, opt.strike, opt.maturity - bump,
                      opt.volatility, opt.risk_free_rate, opt.option_type)
        return (down.price() - opt.price()) / bump / 365.0 * bump

    def fd_rho(self, bump: float = 0.0001) -> float:
        opt = self.option
        up   = Option(opt.spot, opt.strike, opt.maturity,
                      opt.volatility, opt.risk_free_rate + bump, opt.option_type)
        down = Option(opt.spot, opt.strike, opt.maturity,
                      opt.volatility, opt.risk_free_rate - bump, opt.option_type)
        return (up.price() - down.price()) / (2 * bump * 100)   # per 1%