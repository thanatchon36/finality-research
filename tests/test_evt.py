import numpy as np
import pytest
from scipy.stats import genpareto

from finality.evt import (fit_pot, gpd_loglik, gpd_mle, hill, percentile_ci, return_level, return_period, rgpd,
                          rl_bootstrap, rl_ci_delta, rl_ci_profile, rl_incident)


@pytest.mark.parametrize("xi,sigma", [(0.625, 250e6), (0.2, 1.0), (-0.2, 1.0), (1.2, 5.0)])
def test_mle_recovers_parameters_and_matches_scipy_likelihood(xi, sigma):
    rng = np.random.default_rng(1)
    y = rgpd(xi, sigma, 4000, rng)
    f = gpd_mle(y)
    assert abs(f.xi - xi) < 0.08 and abs(f.sigma / sigma - 1) < 0.1
    c, _, s = genpareto.fit(y, floc=0)
    assert f.loglik >= genpareto.logpdf(y, c, scale=s).sum() - 1e-6
    assert f.loglik == pytest.approx(gpd_loglik(y, f.xi, f.sigma), abs=1e-6)


def test_return_level_matches_scipy_quantile():
    u, lam, xi, s, T = 400e6, 0.5, 0.625, 250e6, 1000
    r = -np.log1p(-1 / T)
    assert float(rl_incident(u, lam, xi, s, T)) == pytest.approx(u + genpareto.isf(r / lam, xi, scale=s), rel=1e-12)


def test_return_period_inverts_return_level():
    rng = np.random.default_rng(2)
    m = fit_pot(400e6 + rgpd(0.5, 2e8, 60, rng), 120, u=400e6)
    x = float(return_level(m, 500))
    assert return_period(m, x) == pytest.approx(500, rel=1e-6)


def test_intervals_bracket_the_estimate_and_profile_is_right_skewed():
    rng = np.random.default_rng(3)
    m = fit_pot(400e6 + rgpd(0.6, 2.5e8, 50, rng), 100, u=400e6)
    x = float(return_level(m, 1000))
    for lo, hi in (rl_ci_delta(m, 1000), rl_ci_profile(m, 1000),
                   tuple(percentile_ci(rl_bootstrap(m, [1000], B=199, rng=rng))[:, 0])):
        assert lo < x < hi
    lo, hi = rl_ci_profile(m, 1000)
    assert np.log(hi / x) > np.log(x / lo)          # heavy tails: more room above than below


def test_delta_and_profile_agree_in_large_samples():
    rng = np.random.default_rng(4)
    m = fit_pot(400e6 + rgpd(0.5, 2.5e8, 3000, rng), 6000, u=400e6)
    d, p = rl_ci_delta(m, 1000), rl_ci_profile(m, 1000)
    assert d[0] == pytest.approx(p[0], rel=0.05) and d[1] == pytest.approx(p[1], rel=0.05)


def test_hill_recovers_pareto_index():
    x = (1 - np.random.default_rng(5).random(40000)) ** (-1 / 1.6)
    assert hill(x, 4000)[1999] == pytest.approx(1.6, abs=0.1)


def test_threshold_lowering_keeps_minimum_exceedances():
    m = fit_pot(np.random.default_rng(6).lognormal(0, 1, 120), 30, u_quantile=0.99)
    assert m.lowered and m.k == 15
