#!/usr/bin/env python3
"""Reproduce every result, table and figure of the FINALITY study.

    python run_study.py              # study settings: 1,000 records per cell, B = 199, 2M-year truth (~5 min, 1 core)
    python run_study.py --quick      # smoke run, about a minute
    python run_study.py --reps 5000  # tighter Monte Carlo error for a final paper
    python run_study.py --budget 120 # stop between coverage cells after ~2 min; rerun to resume (cached)

Outputs go to results/: fig1-fig5 (PNG), results.json, results.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from finality import attacker, contagion, coverage, evt, lda, plots  # noqa: E402

SYSTEMIC = 20e9


def fmt_usd(v: float) -> str:
    if not np.isfinite(v):
        return "unbounded"
    return f"${v / 1e12:.2f}T" if v >= 1e12 else (f"${v / 1e9:.2f}B" if v >= 1e9 else f"${v / 1e6:.1f}M")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=1000, help="records per coverage cell")
    ap.add_argument("--boot", type=int, default=199, help="bootstrap resamples per interval")
    ap.add_argument("--truth-years", type=int, default=2_000_000, help="years in the long 'truth' simulation")
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--budget", type=float, default=0, help="seconds for new coverage cells this run (0 = no limit)")
    a = ap.parse_args()
    if a.quick:
        a.reps, a.boot, a.truth_years = 40, 99, 400_000
    os.makedirs(a.out, exist_ok=True)
    log = lambda *x: print(*x, flush=True)
    t_start = time.time()
    s_lda, s_truth, s_evt, s_a, s_b = np.random.SeedSequence(a.seed).spawn(5)
    R = {"settings": vars(a)}

    # 1. attacker target selection ------------------------------------------------------------
    bank, bridge = attacker.VENUES[0], attacker.VENUES[2]
    dec = attacker.log_gap_decomposition(bank, bridge)
    R["attacker"] = {"payoff_usd": {v.name: v.payoff() for v in attacker.VENUES},
                     "bank_break_even_recovery": bank.break_even_recovery(),
                     "log_gap_bank_to_bridge": dec,
                     "finality_share_of_gap": dec["finality (1 - P_recovery)"] / dec["total"]}
    plots.fig_attacker(attacker.VENUES, dec, attacker.recovery_curve(bank), f"{a.out}/fig1_attacker.png")
    log("1 attacker model done")

    # 2. systemic loss model --------------------------------------------------------------------
    p = lda.LDAParams()
    sim = lda.simulate(200_000, p, np.random.default_rng(s_lda))
    R["lda"] = lda.risk_metrics(sim.annual, SYSTEMIC, 10)
    R["lda"]["p_year_at_least"] = {fmt_usd(T): float((sim.annual >= T).mean()) for T in (5e9, 10e9, 20e9, 50e9, 100e9)}
    plots.fig_lda(sim.annual, SYSTEMIC, f"{a.out}/fig2_lda.png")
    t = time.time()
    truth_annual = np.sort(lda.simulate_annual(a.truth_years, p, np.random.default_rng(s_truth)))
    truth = {T: lda.annual_return_level(truth_annual, T) for T in (100, 1000)}
    p_sys = float((truth_annual >= SYSTEMIC).mean())
    R["truth"] = {"years": a.truth_years, "annual_T100": truth[100], "annual_T1000": truth[1000],
                  "systemic_return_period": 1 / p_sys if p_sys else float("inf")}
    log(f"2 loss model done; truth from {a.truth_years:,} years in {time.time() - t:.1f}s")

    # 3. EVT: diagnostics on a 1,000-year record, intervals on a 100-year record -----------------
    rng = np.random.default_rng(s_evt)
    long_rec = lda.simulate(1000, p, rng)
    m_long = evt.fit_pot(long_rec.incidents, 1000, 0.98, annual=long_rec.annual)
    me_u, me_e = evt.mean_excess(long_rec.incidents, 1 - np.logspace(np.log10(0.5), np.log10(10 / long_rec.incidents.size), 70))
    hill = evt.hill(long_rec.incidents, kmax=int(0.3 * long_rec.incidents.size))
    short = lda.simulate(100, p, rng)
    m100 = evt.fit_pot(short.incidents, 100, 0.95, annual=short.annual)
    t_min = 1.0 / -np.expm1(-m100.rate) * 1.02
    Ts = np.logspace(np.log10(max(t_min, 2.0)), 4, 30)
    draws = evt.rl_bootstrap(m100, Ts, B=max(a.boot, 399), rng=rng)
    bands = {"delta": np.array([evt.rl_ci_delta(m100, T) for T in Ts]).T,
             "profile": np.array([evt.rl_ci_profile(m100, T) for T in Ts]).T,
             "bootstrap": evt.percentile_ci(draws)}
    truth_T = np.logspace(0.05, 4, 90)
    truth_x = np.quantile(truth_annual, 1 - 1 / truth_T)
    top = max(truth_x[-1], np.nanmax(np.where(np.isfinite(bands["profile"][1]), bands["profile"][1], 0)))
    demo = {"Ts": Ts, "truth_T": truth_T, "truth_x": truth_x, "record_annual": short.annual,
            "point": evt.return_level(m100, Ts), "bands": bands, "ylim": (8e7, min(top * 2.5, 5e12))}
    plots.fig_evt({"me_u": me_u, "me_e": me_e, "hill": hill, "model": m_long, "alpha_true": p.tail_alpha,
                   "years": 1000}, demo, f"{a.out}/fig3_evt.png")
    ci_1000 = {"delta": evt.rl_ci_delta(m100, 1000), "profile": evt.rl_ci_profile(m100, 1000),
               "bootstrap": tuple(evt.percentile_ci(evt.rl_bootstrap(m100, [1000], B=max(a.boot, 399), rng=rng))[:, 0])}
    R["evt"] = {
        "planted_xi": p.xi_true,
        "record_1000y": {"u": m_long.u, "k": m_long.k, "xi": m_long.fit.xi,
                         "xi_ci95": [m_long.fit.xi - 1.96 * (1 + m_long.fit.xi) / np.sqrt(m_long.k),
                                     m_long.fit.xi + 1.96 * (1 + m_long.fit.xi) / np.sqrt(m_long.k)],
                         "T1000": float(evt.return_level(m_long, 1000)),
                         "systemic_return_period": evt.return_period(m_long, SYSTEMIC)},
        "record_100y": {"u": m100.u, "k": m100.k, "xi": m100.fit.xi, "T1000": float(evt.return_level(m100, 1000)),
                        "ci_T1000": {k: [float(v[0]), float(v[1])] for k, v in ci_1000.items()},
                        "systemic_return_period": evt.return_period(m100, SYSTEMIC)},
    }
    log("3 EVT figure done")

    # 4. coverage study --------------------------------------------------------------------------
    t = time.time()
    cache, deadline = os.path.join(a.out, "cache"), (t + a.budget if a.budget > 0 else None)
    dA = coverage.GPDDesign()
    rows, okA = coverage.coverage_study(dA, [30, 100, 300], (100, 1000), {T: dA.truth(T) for T in (100, 1000)},
                                        a.reps, a.boot, s_a, "A", log, cache, deadline)
    rowsB, okB = coverage.coverage_study(coverage.LDADesign(params=p), [30, 100, 1000], (100, 1000), truth,
                                         a.reps, a.boot, s_b, "B", log, cache, deadline)
    rows += rowsB
    if not (okA and okB):
        log("coverage study incomplete: rerun the same command to resume from the cache")
        return
    R["coverage"] = rows
    plots.fig_coverage(rows, f"{a.out}/fig4_coverage.png")
    log(f"4 coverage study done in {time.time() - t:.0f}s")

    # 5. network contagion -----------------------------------------------------------------------
    net = contagion.build_network(7)
    big, vend = net.largest(), net.vendor_clients(0.3)
    levs = np.linspace(0.1, 2.0, 77)
    sweeps = {f"one bank hit ({net.names[big]}, the largest)": contagion.phase_sweep(net, [big], 0.2, levs),
              f"shared vendor hit ({vend.size} banks)": contagion.phase_sweep(net, vend, 0.2, levs)}
    rank = contagion.systemic_ranking(net, 0.5, 0.2)
    cas = {f"{net.names[big]} hit, exposure 0.5x": contagion.debtrank(net, 0.5, [big], 0.2),
           "vendor hit, exposure 0.5x": contagion.debtrank(net, 0.5, vend, 0.2),
           "vendor hit, exposure 1.2x": contagion.debtrank(net, 1.2, vend, 0.2)}
    plots.fig_contagion(net, levs, sweeps, rank, cas, f"{a.out}/fig5_contagion.png")
    order = np.argsort(-rank)
    R["contagion"] = {
        "seed": 7, "rho1": net.rho1, "critical_exposure": net.critical_exposure,
        "system_equity": float(net.eq.sum()),
        "top_systemic": [[net.names[i], float(rank[i]), float(net.ta[i])] for i in order[:5]],
        "largest_bank": net.names[big], "size_rank_of_most_systemic": int(np.argsort(-net.ta).tolist().index(order[0]) + 1),
        "scenarios": {k: {"direct": contagion.equity_loss(net, h[1]), "total": contagion.equity_loss(net, h[-1]),
                          "multiplier": contagion.multiplier(net, h), "defaults": int((h[-1] >= 0.999).sum()),
                          "rounds": int(len(h) - 2)} for k, h in cas.items()},
    }
    log("5 contagion done")

    R["runtime_seconds"] = time.time() - t_start
    with open(f"{a.out}/results.json", "w") as f:
        json.dump(R, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    write_markdown(R, f"{a.out}/results.md")
    log(f"all done in {R['runtime_seconds']:.0f}s -> {a.out}/")


def write_markdown(R: dict, path: str) -> None:
    L = ["# FINALITY study: results", "",
         f"Settings: {R['settings']['reps']} records per coverage cell, bootstrap B = {R['settings']['boot']}, "
         f"truth from {R['settings']['truth_years']:,} simulated years, seed {R['settings']['seed']}. "
         "All parameters are illustrative.", "", "## 1. Attacker target selection", "",
         "| Venue | Risk-adjusted payoff |", "|---|---:|"]
    L += [f"| {k} | {fmt_usd(v) if v > 0 else '-' + fmt_usd(-v)} |" for k, v in R["attacker"]["payoff_usd"].items()]
    d = R["attacker"]["log_gap_bank_to_bridge"]
    L += ["", f"Bank break-even P(recovery): {R['attacker']['bank_break_even_recovery']:.3f}. "
          f"Log-gap bank to bridge {d['total']:+.3f}, of which finality {d['finality (1 - P_recovery)']:+.3f} "
          f"({R['attacker']['finality_share_of_gap']:.0%}).", "", "## 2. Systemic loss model", ""]
    m = R["lda"]
    L += [f"Mean annual loss {fmt_usd(m['mean'])}; VaR 99.9% {fmt_usd(m['VaR999'])}; ES 99.9% {fmt_usd(m['ES999'])}. "
          f"Truth from the long run: 1-in-100 {fmt_usd(R['truth']['annual_T100'])}, 1-in-1,000 "
          f"{fmt_usd(R['truth']['annual_T1000'])}, a $20B year once per {R['truth']['systemic_return_period']:,.0f} years.",
          "", "## 3. EVT on single records", ""]
    e = R["evt"]
    r1, r0 = e["record_1000y"], e["record_100y"]
    L += [f"1,000-year record: xi = {r1['xi']:.3f} (95% CI {r1['xi_ci95'][0]:.2f} to {r1['xi_ci95'][1]:.2f}; planted "
          f"{e['planted_xi']:.3f}); 1-in-1,000 estimate {fmt_usd(r1['T1000'])}.", "",
          f"100-year record: xi = {r0['xi']:.3f}, 1-in-1,000 estimate {fmt_usd(r0['T1000'])} "
          f"(truth {fmt_usd(R['truth']['annual_T1000'])}).", "", "| Method | 95% interval for the 1-in-1,000 loss |", "|---|---|"]
    L += [f"| {k} | {fmt_usd(v[0])} to {fmt_usd(v[1])} |" for k, v in r0["ci_T1000"].items()]
    L += ["", "## 4. Coverage study (nominal 95%)", "",
          "| Exp. | Record | mean k | T | Method | Coverage (±1.96 SE) | Truth above interval | Median upper/lower | Unbounded |",
          "|---|---:|---:|---:|---|---:|---:|---:|---:|"]
    for r in R["coverage"]:
        L.append(f"| {r['experiment']} | {r['record_years']:,} y | {r['mean_k']:.0f} | {r['T']:,} | {r['method']} | "
                 f"{r['coverage']:.1%} ± {1.96 * r['mc_se']:.1%} | {r['miss_low']:.1%} | {r['median_width_ratio']:.1f}× | "
                 f"{r['share_unbounded']:.0%} |")
    c = R["contagion"]
    L += ["", "## 5. Network contagion (seed 7, identical to the web instrument)", "",
          f"rho(K W) = {c['rho1']:.4f}; critical exposure {c['critical_exposure']:.3f}× equity. "
          f"Most systemic: " + ", ".join(f"{n} ×{m:.2f}" for n, m, _ in c["top_systemic"][:3]) +
          f"; the most systemic bank is number {c['size_rank_of_most_systemic']} by size.", "",
          "| Scenario | Direct | After contagion | Multiplier | Defaults | Rounds |", "|---|---:|---:|---:|---:|---:|"]
    L += [f"| {k} | {fmt_usd(v['direct'])} | {fmt_usd(v['total'])} | ×{v['multiplier']:.2f} | {v['defaults']} | {v['rounds']} |"
          for k, v in c["scenarios"].items()]
    L += ["", f"Runtime {R['runtime_seconds']:.0f}s."]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
