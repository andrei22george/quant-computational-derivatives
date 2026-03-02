import pytest
import numpy as np
from models import Option, OptionType


@pytest.fixture
def atm_call():
    return Option(100, 100, 1.0, 0.20, 0.05, OptionType.CALL)

@pytest.fixture
def atm_put():
    return Option(100, 100, 1.0, 0.20, 0.05, OptionType.PUT)


class TestKnownPrices:
    
    def test_hull_example_call(self):
        opt = Option(42, 40, 0.5, 0.20, 0.10, OptionType.CALL)
        assert abs(opt.call_price() - 4.76) < 0.01

    def test_hull_example_put(self):
        opt = Option(42, 40, 0.5, 0.20, 0.10, OptionType.PUT)
        expected_call = Option(42, 40, 0.5, 0.20, 0.10, OptionType.CALL).call_price()
        expected_put  = expected_call - 42 + 40 * np.exp(-0.10 * 0.5)
        assert abs(opt.put_price() - expected_put) < 1e-8

    def test_atm_call_approx(self):
        opt = Option(100, 100, 1.0, 0.20, 0.0, OptionType.CALL)  # r=0
        approx = opt.spot * opt.volatility * np.sqrt(opt.maturity / (2 * np.pi))
        assert abs(opt.call_price() - approx) < 0.10 # this approximation is pretty rough, but should be resonable


class TestPutCallParity:

    @pytest.mark.parametrize("S,K,T,sigma,r", [
        (100, 100, 1.0, 0.20, 0.05),
        (100, 110, 0.5, 0.30, 0.02),
        (50,  55,  2.0, 0.15, 0.08),
        (200, 180, 0.25, 0.40, 0.01),
        (100, 100, 0.1, 0.80, 0.00),   # high vol, short maturity
    ])
    def test_parity(self, S, K, T, sigma, r):
        call = Option(S, K, T, sigma, r, OptionType.CALL)
        put  = Option(S, K, T, sigma, r, OptionType.PUT)
        residual = call.call_price() - put.put_price() - (S - K * np.exp(-r * T))
        assert abs(residual) < 1e-10, f"Parity violated: residual={residual:.2e}"


class TestBoundaryConditions:

    def test_deep_itm_call_approaches_forward(self):
        opt = Option(200, 100, 1.0, 0.20, 0.05, OptionType.CALL)
        forward_intrinsic = 200 - 100 * np.exp(-0.05)
        assert abs(opt.call_price() - forward_intrinsic) < 0.10

    def test_deep_otm_call_near_zero(self):
        opt = Option(50, 200, 1.0, 0.20, 0.05, OptionType.CALL)
        assert opt.call_price() < 0.01

    def test_deep_itm_put_near_zero(self):
        opt = Option(200, 50, 1.0, 0.20, 0.05, OptionType.PUT)
        assert opt.put_price() < 0.01

    def test_price_positive(self):
        for S in [80, 100, 120]:
            for opt_type in [OptionType.CALL, OptionType.PUT]:
                opt = Option(S, 100, 1.0, 0.20, 0.05, opt_type)
                assert opt.price() >= 0.0

    def test_call_bounded_above_by_spot(self):
        opt = Option(100, 50, 1.0, 0.50, 0.05, OptionType.CALL)
        assert opt.call_price() <= opt.spot + 1e-10

    def test_put_bounded_above_by_pv_strike(self):
        opt = Option(100, 100, 1.0, 0.20, 0.05, OptionType.PUT)
        assert opt.put_price() <= 100 * np.exp(-0.05) + 1e-10


class TestInputValidation:

    def test_negative_spot_raises(self):
        with pytest.raises(ValueError, match="spot"):
            Option(-1, 100, 1.0, 0.20, 0.05, OptionType.CALL)

    def test_zero_maturity_raises(self):
        with pytest.raises(ValueError, match="maturity"):
            Option(100, 100, 0.0, 0.20, 0.05, OptionType.CALL)

    def test_negative_vol_raises(self):
        with pytest.raises(ValueError, match="volatility"):
            Option(100, 100, 1.0, -0.20, 0.05, OptionType.CALL)


class TestHelpers:

    def test_atm_moneyness(self):
        opt = Option(100, 100, 1.0, 0.20, 0.05, OptionType.CALL)
        assert opt.moneyness() == "ATM"

    def test_itm_call_moneyness(self):
        opt = Option(110, 100, 1.0, 0.20, 0.05, OptionType.CALL)
        assert opt.moneyness() == "ITM"

    def test_otm_call_moneyness(self):
        opt = Option(90, 100, 1.0, 0.20, 0.05, OptionType.CALL)
        assert opt.moneyness() == "OTM"

    def test_time_value_nonneg(self):
        for S in [80, 100, 120]:
            opt = Option(S, 100, 1.0, 0.20, 0.05, OptionType.CALL)
            assert opt.time_value() >= -1e-10