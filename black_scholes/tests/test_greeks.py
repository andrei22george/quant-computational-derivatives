import pytest
import numpy as np
from models import Option, OptionType
from greeks import Greeks


@pytest.fixture
def call():
    return Option(100, 100, 1.0, 0.20, 0.05, OptionType.CALL)

@pytest.fixture
def put():
    return Option(100, 100, 1.0, 0.20, 0.05, OptionType.PUT)


class TestDelta:
    def test_call_delta_range(self, call):
        d = Greeks(call).delta()
        assert 0.0 < d < 1.0

    def test_put_delta_range(self, put):
        d = Greeks(put).delta()
        assert -1.0 < d < 0.0

    def test_call_put_delta_relation(self, call, put):
        assert abs(Greeks(call).delta() - Greeks(put).delta() - 1.0) < 1e-10

    @pytest.mark.parametrize("S", [80, 90, 100, 110, 120])
    def test_delta_vs_fd(self, S):
        opt = Option(S, 100, 1.0, 0.20, 0.05, OptionType.CALL)
        g   = Greeks(opt)
        assert abs(g.delta() - g.fd_delta()) < 1e-5


class TestGamma:
    def test_gamma_positive(self, call):
        assert Greeks(call).gamma() > 0

    def test_call_put_gamma_equal(self, call, put):
        assert abs(Greeks(call).gamma() - Greeks(put).gamma()) < 1e-12

    @pytest.mark.parametrize("S", [80, 90, 100, 110, 120])
    def test_gamma_vs_fd(self, S):
        opt = Option(S, 100, 1.0, 0.20, 0.05, OptionType.CALL)
        g   = Greeks(opt)
        assert abs(g.gamma() - g.fd_gamma()) < 1e-4

    def test_gamma_highest_atm(self):
        atm  = Greeks(Option(100, 100, 1.0, 0.20, 0.05, OptionType.CALL)).gamma()
        itm  = Greeks(Option(130, 100, 1.0, 0.20, 0.05, OptionType.CALL)).gamma()
        otm  = Greeks(Option(70,  100, 1.0, 0.20, 0.05, OptionType.CALL)).gamma()
        assert atm > itm and atm > otm


class TestVega:
    def test_vega_positive(self, call):
        assert Greeks(call).vega() > 0

    def test_call_put_vega_equal(self, call, put):
        assert abs(Greeks(call).vega() - Greeks(put).vega()) < 1e-12

    @pytest.mark.parametrize("S", [80, 100, 120])
    def test_vega_vs_fd(self, S):
        opt = Option(S, 100, 1.0, 0.20, 0.05, OptionType.CALL)
        g   = Greeks(opt)
        assert abs(g.vega() - g.fd_vega()) < 1e-4


class TestTheta:
    def test_call_theta_negative_atm(self, call):
        assert Greeks(call).theta() < 0

    def test_theta_larger_near_expiry(self):
        long_dated  = abs(Greeks(Option(100, 100, 2.0, 0.20, 0.05, OptionType.CALL)).theta())
        short_dated = abs(Greeks(Option(100, 100, 0.1, 0.20, 0.05, OptionType.CALL)).theta())
        assert short_dated > long_dated


class TestRho:
    def test_call_rho_positive(self, call):
        assert Greeks(call).rho() > 0

    def test_put_rho_negative(self, put):
        assert Greeks(put).rho() < 0