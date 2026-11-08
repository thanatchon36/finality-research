import numpy as np
import pytest

from finality.contagion import build_network, debtrank, equity_loss, multiplier, systemic_ranking


@pytest.fixture(scope="module")
def net():
    return build_network(7)


def test_matches_web_instrument_seed_7(net):
    """Same PRNG and draw order as the FINALITY web page: its published numbers must reappear."""
    assert net.rho1 == pytest.approx(1.1519, abs=1e-4)
    rank = systemic_ranking(net, 0.5, 0.2)
    top = np.argsort(-rank)[:3]
    assert [net.names[i] for i in top] == ["C1", "P1", "C2"]
    assert rank[top] == pytest.approx([3.71, 3.04, 2.93], abs=0.005)
    h = debtrank(net, 0.5, [net.largest()], 0.2)
    assert net.names[net.largest()] == "C5"
    assert equity_loss(net, h[1]) / 1e9 == pytest.approx(26.68, abs=0.005)
    assert equity_loss(net, h[-1]) / 1e9 == pytest.approx(68.95, abs=0.005)
    hv = debtrank(net, 1.2, net.vendor_clients(0.3), 0.2)
    assert multiplier(net, hv) == pytest.approx(32.55, abs=0.005) and int((hv[-1] >= 0.999).sum()) == 13


def test_spectral_radius_matches_eigenvalues(net):
    M = net.k[:, None] * net.W
    assert net.rho1 == pytest.approx(max(abs(np.linalg.eigvals(M))), rel=1e-12)


def test_linear_regime_equals_closed_form(net):
    i = net.largest()
    h = debtrank(net, 0.5, [i], 1e-6, max_rounds=5000, rtol=1e-12)
    h0 = np.zeros(net.n)
    h0[i] = 1e-6
    closed = np.linalg.solve(np.eye(net.n) - net.Lambda(0.5), h0)
    assert np.abs(h[-1] - closed).max() < 1e-8


def test_supercritical_cascade_causes_defaults(net):
    h = debtrank(net, 1.5, [net.largest()], 0.2)
    assert (h[-1] >= 0.999).sum() > 5
    assert multiplier(net, h) > 10
