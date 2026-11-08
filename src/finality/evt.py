"""Peaks-over-threshold inference for heavy-tailed losses.

Model. Exceedances of a threshold u arrive as a Poisson process with rate lam per year,
with Generalized Pareto sizes  Gbar(y) = (1 + xi*y/sigma)^(-1/xi).

Return level. The incident-level T-year return level psi_T (exceeded by the worst incident
of a year with probability 1/T) solves lam * Gbar(psi_T - u) = r_T, with r_T = -log(1 - 1/T):

    psi_T = u + (sigma/xi) * ((lam / r_T)^xi - 1)

Annual aggregates. For a compound Poisson sum S with subexponential severities of finite
mean mu, the second-order expansion P(S > x) ~ lam*Fbar(x) + lam^2*mu*f(x) ~ lam*Fbar(x - lam*mu)
gives the aggregate T-year loss as roughly psi_T + E[S]: the year's worst incident plus an
ordinary year's worth of everything else. `PotModel.shift` carries the record's mean annual
loss for that purpose (set it to 0 for incident-level work).

Interval estimators for psi_T, compared in `coverage.py`:
    delta      Wald interval on the log scale from the asymptotic covariance (fast, symmetric)
    profile    likelihood-ratio interval, profiling out (lam, xi) on a dense grid
    bootstrap  parametric bootstrap percentile interval
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2, norm

XI_MIN = -0.45                               # MLE regularity needs xi > -0.5
XI_GRID = np.linspace(XI_MIN, 4.0, 297)      # spacing 0.015 for the profile likelihood
XI_GRID[np.abs(XI_GRID) < 1e-9] = 1e-6
LAM_STEPS = np.linspace(-4.0, 4.0, 9)        # rate grid, in standard errors of log(lam)
RATE_GRID_BUDGET = 4e5                       # profile the rate only while k * |xi grid| * |rate grid| stays below this


# ----------------------------------------------------------------------------- GPD fitting
@dataclass(frozen=True)
class GPDFit:
    xi: float
    sigma: float
    loglik: float
    n: int


def gpd_loglik(y: np.ndarray, xi: float, sigma: float) -> float:
    y = np.asarray(y, float)
    if sigma <= 0:
        return -np.inf
    if abs(xi) < 1e-12:
        return float(-y.size * np.log(sigma) - y.sum() / sigma)
    z = xi * y / sigma
    if np.any(z <= -1):
        return -np.inf
    return float(-y.size * np.log(sigma) - (1 + 1 / xi) * np.log1p(z).sum())


def _profile_theta(y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Grimshaw (1993) profile log-likelihood in theta = xi / sigma, vectorised over theta."""
    n = y.size
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.log1p(np.multiply.outer(theta, y)).sum(axis=-1)
        xi = S / n
        ll = -n * np.log(xi / theta) - S - n
    return np.where(np.isfinite(ll) & (xi > XI_MIN), ll, -np.inf)


def gpd_mle(y: np.ndarray) -> GPDFit:
    """Maximum likelihood for the GPD: global grid over theta, local refinement, exponential check."""
    y = np.asarray(y, float)
    n = y.size
    if n < 3:
        raise ValueError("need at least 3 exceedances")
    ybar, ymax = y.mean(), y.max()
    theta = np.sort(np.concatenate([
        -(1.0 - np.exp(-np.linspace(1e-4, 14.0, 60))) / ymax,   # xi < 0 branch
        np.exp(np.linspace(-12.0, 9.0, 150)) / ybar,            # xi > 0 branch
    ]))
    ll = _profile_theta(y, theta)
    i = int(np.argmax(ll))
    fine = np.linspace(theta[max(i - 1, 0)], theta[min(i + 1, theta.size - 1)], 81)
    fine = fine[fine != 0.0]
    llf = _profile_theta(y, fine)
    j = int(np.argmax(llf))
    th, best = fine[j], llf[j]
    if 0 < j < fine.size - 1 and np.all(np.isfinite(llf[j - 1:j + 2])):
        den = llf[j - 1] - 2 * llf[j] + llf[j + 1]
        if den < 0:
            t2 = th + 0.5 * (llf[j - 1] - llf[j + 1]) / den * (fine[1] - fine[0])
            l2 = _profile_theta(y, np.array([t2]))[0]
            if l2 > best:
                th, best = t2, l2
    ll_exp = -n * np.log(ybar) - n
    if not np.isfinite(best) or ll_exp >= best:
        return GPDFit(0.0, float(ybar), float(ll_exp), n)
    xi = float(np.log1p(th * y).sum() / n)
    return GPDFit(xi, float(xi / th), float(best), n)


def rgpd(xi: float, sigma: float, size: int, rng: np.random.Generator) -> np.ndarray:
    u = rng.random(size)
    if abs(xi) < 1e-12:
        return -sigma * np.log1p(-u)
    return sigma / xi * np.expm1(-xi * np.log1p(-u))


# ----------------------------------------------------------------------------- POT model
@dataclass
class PotModel:
    u: float
    y: np.ndarray          # exceedances over u
    R: float               # record length, years
    fit: GPDFit
    shift: float = 0.0     # added to incident-level levels to get annual-aggregate levels
    lowered: bool = False  # threshold lowered to keep the minimum number of exceedances

    @property
    def k(self) -> int:
        return int(self.y.size)

    @property
    def rate(self) -> float:
        return self.k / self.R

    @property
    def loglik(self) -> float:
        """Poisson-GPD log-likelihood at the MLE (rate MLE is k / R)."""
        return self.k * np.log(self.rate) - self.k + self.fit.loglik


def fit_pot(incidents: np.ndarray, record_years: float, u_quantile: float = 0.95,
            annual: np.ndarray | None = None, min_exc: int = 15, u: float | None = None) -> PotModel:
    """Fit the Poisson-GPD model above u (given, or the u_quantile of incident losses)."""
    x = np.sort(np.asarray(incidents, float))
    lowered = False
    if u is None:
        N = x.size
        k = int(np.floor(N * (1 - u_quantile)))
        if k < min_exc:
            k, lowered = min(min_exc, N - 2), True
        if k < 8:
            raise ValueError("too few incidents in the record to fit a tail")
        u = float(x[N - k - 1])
        y = x[N - k:] - u
    else:
        y = x[x > u] - u
        if y.size < 8:
            raise ValueError("too few exceedances")
    shift = float(np.mean(annual)) if annual is not None else 0.0
    return PotModel(float(u), y, float(record_years), gpd_mle(y), shift, lowered)


def _r(T):
    return -np.log1p(-1.0 / np.asarray(T, float))


def rl_incident(u: float, lam: float, xi: float, sigma: float, T) -> np.ndarray:
    """Incident-level T-year return level; NaN where it falls below the threshold."""
    m = lam / _r(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        lm = np.log(m)
        val = u + (sigma * lm if abs(xi) < 1e-12 else sigma / xi * np.expm1(xi * lm))
    return np.where(m >= 1, val, np.nan)


def return_level(model: PotModel, T) -> np.ndarray:
    return rl_incident(model.u, model.rate, model.fit.xi, model.fit.sigma, T) + model.shift


def return_period(model: PotModel, X: float) -> float:
    """EVT-implied return period (years) of an annual loss of at least X."""
    x = X - model.shift - model.u
    if x <= 0:
        return float("nan")
    xi, s = model.fit.xi, model.fit.sigma
    if abs(xi) < 1e-12:
        tail = np.exp(-x / s)
    else:
        q = 1 + xi * x / s
        tail = q ** (-1 / xi) if q > 0 else 0.0
    rate = model.rate * tail
    return float(1.0 / -np.expm1(-rate)) if rate > 0 else float("inf")


# ----------------------------------------------------------------------------- delta method
def gpd_cov(fit: GPDFit, k: int) -> np.ndarray:
    """Asymptotic covariance of (sigma, xi) MLEs (valid for xi > -0.5)."""
    c = 1 + fit.xi
    return np.array([[2 * fit.sigma ** 2 * c, -fit.sigma * c], [-fit.sigma * c, c * c]]) / k


def rl_ci_delta(model: PotModel, T: float, level: float = 0.95) -> tuple[float, float]:
    z = norm.ppf(0.5 + level / 2)
    u, lam, xi, s, k, R = model.u, model.rate, model.fit.xi, model.fit.sigma, model.k, model.R
    r = float(_r(T))
    m = lam / r
    if m < 1:
        return float("nan"), float("nan")
    if abs(xi) < 1e-12:
        L = np.log(m)
        x, dl, ds, dx = u + s * L, s / lam, L, s * L * L / 2
    else:
        mx = m ** xi
        x = u + s / xi * (mx - 1)
        dl, ds = s * m ** (xi - 1) / r, (mx - 1) / xi
        dx = -s / xi ** 2 * (mx - 1) + s / xi * mx * np.log(m)
    g = np.array([ds, dx])
    v = dl * dl * (lam / R) + g @ gpd_cov(model.fit, k) @ g
    f = np.exp(z * np.sqrt(max(v, 0.0)) / x)
    return float(x / f + model.shift), float(x * f + model.shift)


# ----------------------------------------------------------------------------- profile likelihood
def _parabola_gain(fm: float, f0: float, fp: float) -> float:
    if not (np.isfinite(fm) and np.isfinite(fp)):
        return 0.0
    den = fm - 2 * f0 + fp
    return float(-(fp - fm) ** 2 / (8 * den)) if den < 0 else 0.0


def profile_loglik_rl(psi: float, model: PotModel, T: float, lam_grid: np.ndarray) -> float:
    """max over (lam, xi) of the Poisson-GPD log-likelihood subject to psi_T = psi."""
    d = psi - model.u
    if not d > 0:
        return -np.inf
    y, k, R = model.y, model.k, model.R
    xi = XI_GRID[:, None]
    la = np.log(lam_grid / _r(T))[None, :]
    with np.errstate(all="ignore"):
        B = np.expm1(xi * la) / xi                       # ((lam/r)^xi - 1) / xi
        sig = d / B
        ok = (sig > 0) & (1 + xi * y.max() / sig > 0)
        Z = np.log1p(xi[..., None] * y / sig[..., None]).sum(axis=-1)
        ll = k * np.log(lam_grid)[None, :] - lam_grid[None, :] * R - k * np.log(sig) - (1 + 1 / xi) * Z
    ll = np.where(ok & np.isfinite(ll), ll, -np.inf)
    g, h = np.unravel_index(int(np.argmax(ll)), ll.shape)
    best = ll[g, h]
    if not np.isfinite(best):
        return -np.inf
    G, H = ll.shape
    best += _parabola_gain(ll[g - 1, h] if g > 0 else -np.inf, best, ll[g + 1, h] if g < G - 1 else -np.inf)
    if H > 2:
        best += _parabola_gain(ll[g, h - 1] if h > 0 else -np.inf, ll[g, h], ll[g, h + 1] if h < H - 1 else -np.inf)
    return float(best)


def rl_ci_profile(model: PotModel, T: float, level: float = 0.95, profile_rate: bool = True,
                  max_ratio: float = 1e5) -> tuple[float, float]:
    """Likelihood-ratio interval for the T-year return level.

    The upper limit is +inf when the data cannot exclude return levels up to max_ratio times
    the estimate (the profile is too flat). The rate is profiled on a grid of +-4 standard
    errors while k is small (k <~ 150); for larger k its relative standard error 1/sqrt(k) is
    negligible next to the shape uncertainty, and it is held at its MLE.
    """
    psi_hat = float(rl_incident(model.u, model.rate, model.fit.xi, model.fit.sigma, T))
    if not np.isfinite(psi_hat):
        return float("nan"), float("nan")
    k = model.k
    if profile_rate and k * XI_GRID.size * LAM_STEPS.size <= RATE_GRID_BUDGET:
        lam_grid = model.rate * np.exp(LAM_STEPS / np.sqrt(k))
    else:
        lam_grid = np.array([model.rate])
    crit = chi2.ppf(level, 1) / 2
    ref = max(model.loglik, profile_loglik_rl(psi_hat, model, T, lam_grid))
    f = lambda t: ref - profile_loglik_rl(model.u + np.exp(t), model, T, lam_grid) - crit
    t0 = np.log(psi_hat - model.u)
    if not f(t0) < 0:
        return float("nan"), float("nan")

    def side(direction: float, span: float) -> float | None:
        t_in, step = t0, 0.15
        while abs(t_in - t0) < span:
            t_out = t_in + direction * step
            if f(t_out) > 0:
                a, b = sorted((t_in, t_out))
                return brentq(f, a, b, xtol=1e-4)
            t_in, step = t_out, step * 1.6
        return None

    lo, hi = side(-1.0, 40.0), side(1.0, np.log(max_ratio))
    lo_v = model.u + np.exp(lo) if lo is not None else model.u
    hi_v = model.u + np.exp(hi) if hi is not None else np.inf
    return float(lo_v + model.shift), float(hi_v + model.shift)


# ----------------------------------------------------------------------------- bootstrap
def rl_bootstrap(model: PotModel, Ts, B: int = 199, rng: np.random.Generator | None = None,
                 min_k: int = 8) -> np.ndarray:
    """Parametric bootstrap draws of the return levels at each T, shape (B, len(Ts))."""
    rng = np.random.default_rng() if rng is None else rng
    Ts = np.atleast_1d(np.asarray(Ts, float))
    out = np.full((B, Ts.size), np.nan)
    for b in range(B):
        kb = rng.poisson(model.rate * model.R)
        if kb < min_k:
            continue
        fb = gpd_mle(rgpd(model.fit.xi, model.fit.sigma, kb, rng))
        out[b] = rl_incident(model.u, kb / model.R, fb.xi, fb.sigma, Ts)
    return out + model.shift


def percentile_ci(draws: np.ndarray, level: float = 0.95) -> np.ndarray:
    a = (1 - level) / 2
    return np.nanquantile(draws, [a, 1 - a], axis=0)


# ----------------------------------------------------------------------------- diagnostics
def hill(x: np.ndarray, kmax: int | None = None) -> np.ndarray:
    """Hill estimates alpha_k for k = 1..kmax from the k largest observations (index k-1)."""
    xs = np.sort(np.asarray(x, float))[::-1]
    kmax = min(xs.size - 1, kmax or xs.size - 1)
    lx = np.log(xs[:kmax + 1])
    k = np.arange(1, kmax + 1)
    return 1.0 / (np.cumsum(lx[:kmax]) / k - lx[1:kmax + 1])


def mean_excess(x: np.ndarray, quantiles: np.ndarray, min_count: int = 8) -> tuple[np.ndarray, np.ndarray]:
    xs = np.sort(np.asarray(x, float))
    us, es = [], []
    for q in quantiles:
        u = np.quantile(xs, q)
        exc = xs[xs > u] - u
        if exc.size < min_count:
            break
        us.append(u)
        es.append(exc.mean())
    return np.array(us), np.array(es)
