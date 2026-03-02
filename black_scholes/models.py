from __future__ import annotations

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from enum import Enum


class OptionType(str, Enum):
    CALL = "call"
    PUT  = "put"


@dataclass
class Option:
    spot:           float
    strike:         float
    maturity:       float
    volatility:     float
    risk_free_rate: float
    option_type:    OptionType

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError(f"spot must be > 0, got {self.spot}")
        if self.strike <= 0:
            raise ValueError(f"strike must be > 0, got {self.strike}")
        if self.maturity <= 0:
            raise ValueError(f"maturity must be > 0, got {self.maturity}")
        if self.volatility <= 0:
            raise ValueError(f"volatility must be > 0, got {self.volatility}")
        if not isinstance(self.option_type, OptionType):
            self.option_type = OptionType(str(self.option_type).lower())

    def d1(self) -> float:
        return (
            np.log(self.spot / self.strike)
            + (self.risk_free_rate + 0.5 * self.volatility ** 2) * self.maturity
        ) / (self.volatility * np.sqrt(self.maturity))

    def d2(self) -> float:
        return self.d1() - self.volatility * np.sqrt(self.maturity)

    @property
    def _df(self) -> float:
        return np.exp(-self.risk_free_rate * self.maturity)
    
    def call_price(self) -> float:
        d1, d2 = self.d1(), self.d2()
        return self.spot * norm.cdf(d1) - self.strike * self._df * norm.cdf(d2)

    def put_price(self) -> float:
        d1, d2 = self.d1(), self.d2()
        return self.strike * self._df * norm.cdf(-d2) - self.spot * norm.cdf(-d1)

    def price(self) -> float:
        return self.call_price() if self.option_type == OptionType.CALL else self.put_price()
    
    def intrinsic_value(self) -> float:
        if self.option_type == OptionType.CALL:
            return max(self.spot - self.strike, 0.0)
        return max(self.strike - self.spot, 0.0)

    def time_value(self) -> float:
        return self.price() - self.intrinsic_value()

    def moneyness(self) -> str:
        ratio = self.spot / self.strike
        if abs(ratio - 1.0) < 0.01:
            return "ATM"
        if self.option_type == OptionType.CALL:
            return "ITM" if ratio > 1.0 else "OTM"
        return "ITM" if ratio < 1.0 else "OTM"

    def put_call_parity_residual(self) -> float:
        return self.call_price() - self.put_price() - (self.spot - self.strike * self._df)

    def summary(self) -> dict:
        return {
            "type":            self.option_type.value,
            "spot":            self.spot,
            "strike":          self.strike,
            "maturity":        self.maturity,
            "volatility":      self.volatility,
            "risk_free_rate":  self.risk_free_rate,
            "d1":              round(self.d1(), 6),
            "d2":              round(self.d2(), 6),
            "call_price":      round(self.call_price(), 6),
            "put_price":       round(self.put_price(), 6),
            "intrinsic_value": round(self.intrinsic_value(), 6),
            "time_value":      round(self.time_value(), 6),
            "moneyness":       self.moneyness(),
            "parity_residual": abs(round(self.put_call_parity_residual(), 12)),
        }

    def __repr__(self) -> str:
        return (
            f"Option({self.option_type.value.upper()} "
            f"S={self.spot} K={self.strike} T={self.maturity} "
            f"σ={self.volatility:.1%} r={self.risk_free_rate:.1%})"
        )