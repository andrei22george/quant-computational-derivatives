import pytest
import numpy as np
from models import Option, OptionType
from implied_vol import implied_volatility


class TestImpliedVolRoundTrip:

    @pytest.mark.parametrize("S,K,T,sigma,r,opt_type", [
        # ATM calls/puts
        (100, 100, 1.0, 0.20, 0.05, OptionType.CALL),
        (100, 100, 1.0, 0.20, 0.05, OptionType.PUT),
        # OTM call
        (100, 110, 0.5, 0.25, 0.03, OptionType.CALL),
        # ITM put
        (100, 110, 0.5, 0.25, 0.03, OptionType.PUT),
        # high volatility
        (100, 100, 0.25, 0.80, 0.05, OptionType.CALL),
        # low volatility, long dated
        (100, 100, 2.0,  0.05, 0.05, OptionType.CALL),
        # zero rate
        (100, 100, 1.0, 0.30, 0.00, OptionType.PUT),
    ])
    def test_round_trip(self, S, K, T, sigma, r, opt_type):
        opt    = Option(S, K, T, sigma, r, opt_type)
        price  = opt.price()
        iv     = implied_volatility(price, S, K, T, r, opt_type)
        assert abs(iv - sigma) < 1e-6, \
            f"IV round-trip failed: input={sigma:.4f}, recovered={iv:.4f}"


class TestImpliedVolDiagnostics:

    def test_newton_converges_atm(self):
        opt    = Option(100, 100, 1.0, 0.20, 0.05, OptionType.CALL)
        result = implied_volatility(opt.price(), 100, 100, 1.0, 0.05,
                                    OptionType.CALL, return_diagnostics=True)
        assert result["method"] == "newton_raphson"
        assert abs(result["iv"] - 0.20) < 1e-6

    def test_arbitrage_violation_raises(self):
        with pytest.raises(ValueError, match="arbitrage"):
            implied_volatility(150.0, 100, 100, 1.0, 0.05, OptionType.CALL)


class TestNumericalStability:

    def test_deep_otm_call(self):
        opt   = Option(50, 100, 0.5, 0.20, 0.05, OptionType.CALL)
        price = opt.price()
        if price > 1e-6:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                iv = implied_volatility(price, 50, 100, 0.5, 0.05, OptionType.CALL)
            assert abs(iv - 0.20) < 1e-4

    def test_high_vol(self):
        opt   = Option(100, 100, 1.0, 1.50, 0.05, OptionType.CALL)
        price = opt.price()
        iv    = implied_volatility(price, 100, 100, 1.0, 0.05, OptionType.CALL)
        assert abs(iv - 1.50) < 1e-5