"""Monte Carlo study: do nominal 95% intervals for extreme return levels really cover 95%?

Experiment A (well specified). Exceedances are exactly Poisson-GPD above a known threshold,
so any shortfall in coverage is the interval method's fault.

Experiment B (realistic pipeline). Records come from the cyber LDA simulator (log-normal body,
Pareto tail, cascades); the threshold is a percentile of the record's incident losses and the
target is the annual-aggregate T-year loss, estimated as incident level + mean annual loss.
Shortfalls here mix interval error with threshold bias, which is what a practitioner faces.

Each record is independent; seeds come from one SeedSequence, so results are reproducible.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

from .evt import fit_pot, percentile_ci, return_level, rgpd, rl_bootstrap, rl_ci_delta, rl_ci_profile, rl_incident
from .lda import LDAParams, simulate

METHODS = ("delta", "profile", "bootstrap")


@dataclass
class GPDDesign:
    u: float = 400e6
    sigma: float = 250e6
    xi: float = 0.625
    rate: float = 0.5          # exceedances per year

    def truth(self, T: float) -> float:
        return float(rl_incident(self.u, self.rate, self.xi, self.sigma, T))


@dataclass
class LDADesign:
    params: LDAParams = field(default_factory=LDAParams)
    u_quantile: dict = field(default_factory=lambda: {30: 0.85, 100: 0.95, 1000: 0.98})


def _intervals(model, Ts, B, rng) -> dict:
    """Point estimate and the three intervals at each T, keyed 'method|T' (JSON friendly)."""
    out = {"k": model.k}
    boot = percentile_ci(rl_bootstrap(model, Ts, B=B, rng=rng)) if B else None
    for i, T in enumerate(Ts):
        out[f"point|{T}"] = float(return_level(model, T))
        out[f"delta|{T}"] = list(rl_ci_delta(model, T))
        out[f"profile|{T}"] = list(rl_ci_profile(model, T))
        if B:
            out[f"bootstrap|{T}"] = [float(boot[0, i]), float(boot[1, i])]
    return out


def run_design(design, record_years: int, Ts, reps: int, B: int, seed) -> list:
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(reps):
        try:
            if isinstance(design, GPDDesign):
                k = rng.poisson(design.rate * record_years)
                y = rgpd(design.xi, design.sigma, k, rng)
                model = fit_pot(design.u + y, record_years, u=design.u)
            else:
                sim = simulate(record_years, design.params, rng)
                model = fit_pot(sim.incidents, record_years, u_quantile=design.u_quantile[record_years],
                                annual=sim.annual)
            results.append(_intervals(model, Ts, B, rng))
        except ValueError:
            results.append(None)
    return results


def summarise(results: list, truth: dict, Ts) -> list[dict]:
    rows = []
    ok = [r for r in results if r is not None]
    for T in Ts:
        tv = truth[T]
        pts = np.array([r[f"point|{T}"] for r in ok])
        for m in METHODS:
            iv = [r[f"{m}|{T}"] for r in ok if f"{m}|{T}" in r]
            iv = np.array([v for v in iv if np.isfinite(v[0])])
            if iv.size == 0:
                continue
            lo, hi = iv[:, 0], iv[:, 1]
            cover = (lo <= tv) & (tv <= hi)
            p = float(cover.mean())
            rows.append({
                "T": T, "method": m, "n": int(len(iv)), "coverage": p,
                "mc_se": float(np.sqrt(p * (1 - p) / len(iv))),
                "miss_low": float((hi < tv).mean()),        # truth above the interval: underestimated risk
                "miss_high": float((lo > tv).mean()),
                "median_width_ratio": float(np.median(hi / lo)),
                "share_unbounded": float(np.isinf(hi).mean()),
                "median_point_over_truth": float(np.nanmedian(pts / tv)),
                "failed_records": int(len(results) - len(ok)),
            })
    return rows


def coverage_study(design, record_lengths, Ts, truth: dict, reps: int, B: int, seed,
                   label: str, log=print, cache_dir: str | None = None,
                   deadline: float | None = None) -> tuple[list[dict], bool]:
    """Run (or load from cache) one cell per record length. Returns (rows, complete).

    Raw intervals are cached per cell under a key built from the design, settings and seed, so an
    interrupted study resumes where it stopped; `deadline` (a time.time() value) stops cleanly
    between cells.
    """
    rows, complete = [], True
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    children = ss.spawn(len(record_lengths))
    for R, child in zip(record_lengths, children):
        key = hashlib.md5(f"{design!r}|{R}|{tuple(Ts)}|{reps}|{B}|{child.entropy}|{child.spawn_key}".encode()).hexdigest()[:12]
        path = os.path.join(cache_dir, f"{label}_{R}y_{key}.json") if cache_dir else None
        t = time.time()
        if path and os.path.exists(path):
            with open(path) as f:
                res = json.load(f)
            log(f"  {label}: {R:>5}-year records loaded from cache")
        elif deadline is not None and t > deadline:
            complete = False
            log(f"  {label}: {R:>5}-year records not run yet (time budget reached)")
            continue
        else:
            res = run_design(design, R, Ts, reps, B, child)
            if path:
                os.makedirs(cache_dir, exist_ok=True)
                with open(path, "w") as f:
                    json.dump(res, f)
            log(f"  {label}: {R:>5}-year records, {reps} reps done in {time.time() - t:5.1f}s")
        for row in summarise(res, truth, Ts):
            row.update({"experiment": label, "record_years": R,
                        "mean_k": float(np.mean([r["k"] for r in res if r is not None]))})
            rows.append(row)
    return rows, complete
