"""Loss Distribution Approach (LDA) simulator for banking-core cyber losses.

Frequency   N ~ Poisson(lam) loss-causing incidents per year
Severity    X ~ log-normal body (prob 1 - p_tail) or Pareto tail (prob p_tail); an incident
            above `trigger` cascades with prob c_prob, multiplying its loss by 1 + Exp(c_mult - 1)
Aggregate   S = sum of the year's incident losses

The planted tail index of both X and S is alpha = tail_alpha (xi = 1/alpha): Breiman's lemma
keeps the index under multiplication by a light-tailed factor, and subexponentiality keeps it
under compound Poisson summation. This lets the EVT module be checked against a known truth.

All defaults are ILLUSTRATIVE and must be recalibrated (IC3, VCDB, IBM-Ponemon, IMF/BIS).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

USD_M, USD_B = 1e6, 1e9


@dataclass(frozen=True)
class LDAParams:
    lam: float = 4.0
    body_median: float = 30 * USD_M
    body_sigma: float = 1.4
    p_tail: float = 0.05
    tail_xm: float = 200 * USD_M
    tail_alpha: float = 1.6
    trigger: float = 1_000 * USD_M
    c_mult: float = 3.0
    c_prob: float = 0.30

    @property
    def xi_true(self) -> float:
        return 1.0 / self.tail_alpha if self.p_tail > 0 else 0.0


def draw_severity(n: int, p: LDAParams, rng: np.random.Generator) -> np.ndarray:
    tail = rng.random(n) < p.p_tail
    sev = np.empty(n)
    n_tail = int(tail.sum())
    sev[~tail] = rng.lognormal(np.log(p.body_median), p.body_sigma, n - n_tail)
    sev[tail] = p.tail_xm * (1.0 - rng.random(n_tail)) ** (-1.0 / p.tail_alpha)
    cascade = (sev > p.trigger) & (rng.random(n) < p.c_prob)
    sev[cascade] *= 1.0 + rng.exponential(max(p.c_mult - 1.0, 0.0), int(cascade.sum()))
    return sev


@dataclass
class Simulation:
    annual: np.ndarray       # aggregate loss per year
    incidents: np.ndarray    # every incident loss, in chronological order
    counts: np.ndarray       # incidents per year
    params: LDAParams

    def window(self, start: int, years: int) -> tuple[np.ndarray, np.ndarray]:
        """Incident losses and annual losses observed in years [start, start + years)."""
        edges = np.concatenate([[0], np.cumsum(self.counts)])
        return self.incidents[edges[start]:edges[start + years]], self.annual[start:start + years]


def simulate(n_years: int, p: LDAParams, rng: np.random.Generator) -> Simulation:
    counts = rng.poisson(p.lam, n_years)
    sev = draw_severity(int(counts.sum()), p, rng)
    annual = np.bincount(np.repeat(np.arange(n_years), counts), weights=sev, minlength=n_years)
    return Simulation(annual, sev, counts, p)


def simulate_annual(n_years: int, p: LDAParams, rng: np.random.Generator, chunk: int = 250_000) -> np.ndarray:
    """Annual losses only, in chunks, for very long 'truth' runs."""
    out = np.empty(n_years)
    for s in range(0, n_years, chunk):
        m = min(chunk, n_years - s)
        out[s:s + m] = simulate(m, p, rng).annual
    return out


def risk_metrics(annual: np.ndarray, systemic: float, horizon: int = 10) -> dict:
    srt = np.sort(annual)
    var = {q: float(np.quantile(srt, q)) for q in (0.95, 0.99, 0.999)}
    es = {q: float(srt[srt >= var[q]].mean()) for q in (0.99, 0.999)}
    p_sys = float((annual >= systemic).mean())
    return {
        "mean": float(annual.mean()), "median": float(np.median(annual)),
        "VaR95": var[0.95], "VaR99": var[0.99], "VaR999": var[0.999],
        "ES99": es[0.99], "ES999": es[0.999],
        "p_systemic_year": p_sys,
        "return_period_years": 1.0 / p_sys if p_sys > 0 else float("inf"),
        f"p_within_{horizon}y": 1.0 - (1.0 - p_sys) ** horizon,
    }


def annual_return_level(annual_sorted: np.ndarray, T: float) -> float:
    """Empirical T-year annual loss: the (1 - 1/T) quantile."""
    return float(np.quantile(annual_sorted, 1.0 - 1.0 / T))
