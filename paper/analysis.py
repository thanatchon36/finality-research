#!/usr/bin/env python3
"""Statistical analysis for the paper. Reads results/results.json and the per-record interval cache,
then writes LaTeX tables (paper/tables/*.tex) and number macros (paper/tables/numbers.tex).

Every number quoted in the paper comes from this script; nothing is transcribed by hand.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np
from scipy.stats import binomtest, norm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from finality import attacker  # noqa: E402
from finality.contagion import build_network  # noqa: E402
from finality.coverage import GPDDesign  # noqa: E402
from finality.lda import LDAParams  # noqa: E402

OUT = os.path.join(HERE, "tables")
os.makedirs(OUT, exist_ok=True)
R = json.load(open(os.path.join(ROOT, "results", "results.json")))
TRUTH = {"A": {T: GPDDesign().truth(T) for T in (100, 1000)},
         "B": {100: R["truth"]["annual_T100"], 1000: R["truth"]["annual_T1000"]}}
METHODS = ("delta", "bootstrap", "profile")
MNAME = {"delta": "Delta", "bootstrap": "Bootstrap", "profile": "Profile LR"}
RWORD = {30: "Thirty", 100: "Hundred", 300: "ThreeHundred", 1000: "Thousand"}
TWORD = {100: "C", 1000: "K"}
ALPHA = 0.05
macros: dict[str, str] = {}


def mac(name: str, value: str) -> None:
    assert re.fullmatch(r"[A-Za-z]+", name), name
    macros[name] = value


def usd(v: float, digits: int = 2) -> str:
    if not np.isfinite(v):
        return r"$\infty$"
    if abs(v) >= 1e12:
        return rf"\${v / 1e12:.{digits}f}T"
    if abs(v) >= 1e9:
        return rf"\${v / 1e9:.{digits}f}B"
    return rf"\${v / 1e6:.1f}M"


def wilson(x: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = x / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# ------------------------------------------------------------------------- load per-record intervals
cells: dict[tuple[str, int], list] = {}
for f in sorted(glob.glob(os.path.join(ROOT, "results", "cache", "*.json"))):
    m = re.match(r"([AB])_(\d+)y_", os.path.basename(f))
    cells[(m.group(1), int(m.group(2)))] = [r for r in json.load(open(f)) if r is not None]
assert len(cells) == 6, f"expected 6 cached cells, found {len(cells)}"


def outcomes(recs, method, T, truth):
    iv = np.array([r[f"{method}|{T}"] for r in recs], float)
    lo, hi = iv[:, 0], iv[:, 1]
    ok = np.isfinite(lo)
    cover = (lo <= truth) & (truth <= hi)
    below = hi < truth                 # interval entirely below the truth: risk understated
    above = lo > truth
    with np.errstate(divide="ignore", invalid="ignore"):
        lw, lt = np.log10(lo), np.log10(truth)
        score = (np.log10(hi) - lw) + (2 / ALPHA) * np.clip(lw - lt, 0, None) + (2 / ALPHA) * np.clip(lt - np.log10(hi), 0, None)
    return dict(ok=ok, cover=cover, below=below, above=above, ratio=hi / lo, score=score)


stats = {}
for (exp, Ryr), recs in sorted(cells.items()):
    kbar = float(np.mean([r["k"] for r in recs]))
    for T in (100, 1000):
        truth = TRUTH[exp][T]
        pts = np.array([r[f"point|{T}"] for r in recs])
        o = {m: outcomes(recs, m, T, truth) for m in METHODS}
        for m in METHODS:
            x = o[m]
            n = int(x["ok"].sum())
            c = int(x["cover"][x["ok"]].sum())
            lo, hi = wilson(c, n)
            s = dict(n=n, cov=c / n, wlo=lo, whi=hi, below=float(x["below"][x["ok"]].mean()),
                     above=float(x["above"][x["ok"]].mean()), ratio=float(np.median(x["ratio"][x["ok"]])),
                     score=float(np.mean(x["score"][x["ok"]])), kbar=kbar,
                     bias=float(np.nanmedian(pts / truth)), failed=len(recs) - n)
            if m != "profile":                      # paired McNemar test: profile vs this method
                both = x["ok"] & o["profile"]["ok"]
                b = int((o["profile"]["cover"] & ~x["cover"] & both).sum())
                cc = int((~o["profile"]["cover"] & x["cover"] & both).sum())
                s["mcnemar_b"], s["mcnemar_c"] = b, cc
                s["mcnemar_p"] = float(binomtest(min(b, cc), b + cc, 0.5).pvalue) if b + cc else 1.0
            stats[(exp, Ryr, T, m)] = s

# ------------------------------------------------------------------------- macros: coverage study
for (exp, Ryr, T, m), s in stats.items():
    base = f"{exp}{RWORD[Ryr]}{MNAME[m].split()[0]}{TWORD[T]}"
    mac("cov" + base, f"{100 * s['cov']:.1f}")
    mac("below" + base, f"{100 * s['below']:.1f}")
    mac("ratio" + base, f"{s['ratio']:,.0f}" if s["ratio"] >= 100 else f"{s['ratio']:.1f}")
    mac("score" + base, f"{s['score']:.2f}")
    mac("medRatio" + f"{exp}{RWORD[Ryr]}{TWORD[T]}", f"{s['bias']:.2f}")
mac("Reps", f"{len(next(iter(cells.values()))):,}")
mac("BootB", str(R["settings"]["boot"]))
mac("TruthYears", f"{R['truth']['years']:,}")

# ------------------------------------------------------------------------- table: main coverage (T = 1,000)
def cov_cell(s):
    return rf"{100 * s['cov']:.1f} \tiny{{[{100 * s['wlo']:.1f}, {100 * s['whi']:.1f}]}}"


rows = []
for exp in ("A", "B"):
    for Ryr in sorted({k[1] for k in stats if k[0] == exp}):
        s = {m: stats[(exp, Ryr, 1000, m)] for m in METHODS}
        label = r"A: exact GPD" if (exp == "A" and Ryr == 30) else (r"B: loss pipeline" if (exp == "B" and Ryr == 30) else "")
        rows.append(f"{label} & {Ryr:,} & {s['delta']['kbar']:.0f} & {s['delta']['bias']:.2f} & " + " & ".join(cov_cell(s[m]) for m in METHODS) + r" \\")
    if exp == "A":
        rows.append(r"\addlinespace")
with open(os.path.join(OUT, "tab_coverage_main.tex"), "w") as f:
    f.write("\n".join(rows) + "\n")

# ------------------------------------------------------------------------- table: anatomy of misses (T = 1,000)
rows = []
for exp in ("A", "B"):
    for Ryr in sorted({k[1] for k in stats if k[0] == exp}):
        s = {m: stats[(exp, Ryr, 1000, m)] for m in METHODS}
        p = lambda x: r"$<$0.001" if x < 0.001 else f"{x:.3f}"
        rows.append(f"{exp} & {Ryr:,} & " + " & ".join(f"{100 * s[m]['below']:.1f}" for m in METHODS) + " & "
                    + " & ".join((f"{s[m]['ratio']:,.0f}" if s[m]['ratio'] >= 100 else f"{s[m]['ratio']:.1f}") for m in METHODS) + " & "
                    + " & ".join(f"{s[m]['score']:.2f}" for m in METHODS) + " & "
                    + f"{p(s['delta']['mcnemar_p'])} & {p(s['bootstrap']['mcnemar_p'])}" + r" \\")
    if exp == "A":
        rows.append(r"\addlinespace")
with open(os.path.join(OUT, "tab_misses.tex"), "w") as f:
    f.write("\n".join(rows) + "\n")

# ------------------------------------------------------------------------- appendix table: everything
rows = []
for exp in ("A", "B"):
    for Ryr in sorted({k[1] for k in stats if k[0] == exp}):
        for T in (100, 1000):
            for m in METHODS:
                s = stats[(exp, Ryr, T, m)]
                rows.append(f"{exp} & {Ryr:,} & {s['kbar']:.0f} & {T:,} & {MNAME[m]} & {100 * s['cov']:.1f} & "
                            f"[{100 * s['wlo']:.1f}, {100 * s['whi']:.1f}] & {100 * s['below']:.1f} & {100 * s['above']:.1f} & "
                            + (f"{s['ratio']:,.0f}" if s['ratio'] >= 100 else f"{s['ratio']:.1f}")
                            + f" & {s['score']:.2f} & {s['bias']:.2f}" + r" \\")
            rows.append(r"\addlinespace[2pt]")
with open(os.path.join(OUT, "tab_coverage_full.tex"), "w") as f:
    f.write("\n".join(rows[:-1]) + "\n")

# ------------------------------------------------------------------------- attacker
V = attacker.VENUES
short = ["Bank core (SWIFT-connected)", "Custodial crypto exchange", "DeFi cross-chain bridge", "Hardware-wallet phishing"]
rows = []
for v, nm in zip(V, short):
    rows.append(f"{nm} & {v.p_breach:.2f} & {v.value / 1e6:,.0f} & {v.p_recovery:.2f} & {v.haircut:.2f} & {v.p_attribution:.2f} & "
                f"{v.sanction / 1e6:,.0f} & {v.cost / 1e6:.1f} & {v.retained() / 1e6:.1f} & "
                + (r"$-$" if v.payoff() < 0 else "") + f"{abs(v.payoff()) / 1e6:.1f}" + r" \\")
with open(os.path.join(OUT, "tab_venues.tex"), "w") as f:
    f.write("\n".join(rows) + "\n")
dec = R["attacker"]["log_gap_bank_to_bridge"]
fac = {"breach probability": "Breach", "value at risk": "Value", "finality (1 - P_recovery)": "Finality", "laundering (1 - haircut)": "Laundering"}
for k, w in fac.items():
    mac("gap" + w, f"{dec[k]:+.3f}")
    mac("share" + w, f"{100 * dec[k] / dec['total']:.0f}")
mac("gapTotal", f"{dec['total']:+.3f}")
mac("retainedRatio", f"{np.exp(dec['total']):.1f}")
mac("breakEven", f"{R['attacker']['bank_break_even_recovery']:.3f}")
mac("breakEvenExchange", f"{V[1].break_even_recovery():.3f}")
mac("breakEvenBridge", f"{V[2].break_even_recovery():.3f}")
mac("bridgeKappa", f"{V[2].p_recovery:.2f}")
mac("exchangeKappa", f"{V[1].p_recovery:.2f}")
for v, w in zip(V, ("Bank", "Exchange", "Bridge", "Wallet")):
    mac("pay" + w, (r"$-$" if v.payoff() < 0 else "") + usd(abs(v.payoff()), 1))

# ------------------------------------------------------------------------- loss model
p = LDAParams()
L = R["lda"]
for k, w in (("mean", "Mean"), ("median", "Median"), ("VaR95", "VaRA"), ("VaR99", "VaRB"), ("VaR999", "VaRC"), ("ES99", "ESB"), ("ES999", "ESC")):
    mac("lda" + w, usd(L[k]))
mac("ldaPsys", f"{100 * L['p_systemic_year']:.3f}")
mac("ldaDecade", f"{100 * L['p_within_10y']:.2f}")
mac("truthC", usd(R["truth"]["annual_T100"]))
mac("truthK", usd(R["truth"]["annual_T1000"]))
mac("truthSys", f"{R['truth']['systemic_return_period']:,.0f}")
mac("xiTrue", f"{p.xi_true:.3f}")
mac("truthAK", usd(TRUTH["A"][1000]))
mac("truthAC", usd(TRUTH["A"][100]))

# ------------------------------------------------------------------------- single-record EVT
E = R["evt"]
r1, r0 = E["record_1000y"], E["record_100y"]
mac("longU", usd(r1["u"]))
mac("longK", str(r1["k"]))
mac("longXi", f"{r1['xi']:.2f}")
mac("longXiLo", f"{r1['xi_ci95'][0]:.2f}")
mac("longXiHi", f"{r1['xi_ci95'][1]:.2f}")
mac("longPsi", usd(r1["T1000"]))
mac("shortU", usd(r0["u"]))
mac("shortK", str(r0["k"]))
mac("shortXi", f"{r0['xi']:.2f}")
mac("shortPsi", usd(r0["T1000"]))
for m, w in (("delta", "Delta"), ("profile", "Profile"), ("bootstrap", "Boot")):
    lo, hi = r0["ci_T1000"][m]
    mac("short" + w + "Lo", usd(lo))
    mac("short" + w + "Hi", usd(hi))

# ------------------------------------------------------------------------- contagion
C = R["contagion"]
net = build_network(7)
mac("rhoOne", f"{C['rho1']:.4f}")
mac("critExp", f"{C['critical_exposure']:.3f}")
mac("sysEquity", usd(C["system_equity"], 0))
top = C["top_systemic"]
for i, w in enumerate(("One", "Two", "Three")):
    mac("top" + w + "Name", top[i][0])
    mac("top" + w + "Mult", f"{top[i][1]:.2f}")
    mac("top" + w + "Assets", usd(top[i][2], 0))
mac("mostSystemicSizeRank", str(C["size_rank_of_most_systemic"]))
sc = list(C["scenarios"].values())
for s, w in zip(sc, ("Big", "VendLow", "VendHigh")):
    mac("sc" + w + "Direct", usd(s["direct"]))
    mac("sc" + w + "Total", usd(s["total"]))
    mac("sc" + w + "Mult", f"{s['multiplier']:.2f}")
    mac("sc" + w + "Def", str(s["defaults"]))
    mac("sc" + w + "Rounds", str(s["rounds"]))
mac("scVendHighShare", f"{100 * sc[2]['total'] / C['system_equity']:.1f}")
p1 = net.names.index("P1")
mac("pOneAssets", usd(net.ta[p1], 0))
ratios = sorted(net.ta[i] / net.ta[p1] for i in range(5) if C["top_systemic"][1][1] > 0 and net.names[i] in ("C3", "C4", "C5"))
mac("pOneRatioLo", f"{ratios[0]:.0f}")
mac("pOneRatioHi", f"{ratios[-1]:.0f}")
rank = sorted(((n, mlt, ta) for n, mlt, ta in top), key=lambda t: -t[1])
size_rank = {net.names[i]: r + 1 for r, i in enumerate(np.argsort(-net.ta))}
from finality.contagion import systemic_ranking  # noqa: E402
rank_all = systemic_ranking(net, 0.5, 0.2)
mac("pOneSizeRank", str(size_rank["P1"]))
from finality.contagion import debtrank, equity_loss  # noqa: E402
for nm in ("C1", "P1"):
    i = net.names.index(nm)
    h = debtrank(net, 0.5, [i], 0.2)
    mac(("cOne" if nm == "C1" else "pOne") + "Total", usd(equity_loss(net, h[-1])))
    mac(("cOne" if nm == "C1" else "pOne") + "Direct", usd(equity_loss(net, h[1])))
mac("pOneBeatsCore", str(sum(1 for i in range(5) if rank_all[i] < rank_all[net.names.index("P1")])))
mac("nBanks", str(net.n))
rows = [f"{i + 1} & {n} & {'core' if n.startswith('C') else 'periphery'} & {ta / 1e9:,.0f} & {size_rank[n]} & {mlt:.2f}" + r" \\" for i, (n, mlt, ta) in enumerate(rank)]
with open(os.path.join(OUT, "tab_ranking.tex"), "w") as f:
    f.write("\n".join(rows) + "\n")
names = {0: r"Largest bank (C5) hit, $\ell = 0.5$", 1: r"Shared vendor (8 banks), $\ell = 0.5$", 2: r"Shared vendor (8 banks), $\ell = 1.2$"}
rows = [f"{names[i]} & {usd(s['direct'])} & {usd(s['total'])} & {s['multiplier']:.2f} & {s['defaults']} & {s['rounds']}" + r" \\" for i, s in enumerate(sc)]
with open(os.path.join(OUT, "tab_scenarios.tex"), "w") as f:
    f.write("\n".join(rows) + "\n")

# ------------------------------------------------------------------------- compound-sum correction on a long record
from finality import evt, lda  # noqa: E402
s_lda = np.random.SeedSequence(R["settings"]["seed"]).spawn(5)[0]
sim = lda.simulate(200_000, LDAParams(), np.random.default_rng(s_lda))
mf = evt.fit_pot(sim.incidents, 200_000, 0.995, annual=sim.annual)
mac("fullK", f"{mf.k:,}")
mac("fullU", usd(mf.u))
mac("fullXi", f"{mf.fit.xi:.3f}")
mac("fullXiSE", f"{(1 + mf.fit.xi) / np.sqrt(mf.k):.3f}")
mac("fullShift", usd(mf.shift))
for T, w in ((100, "C"), (1000, "K")):
    inc = float(evt.rl_incident(mf.u, mf.rate, mf.fit.xi, mf.fit.sigma, T))
    tv = TRUTH["B"][T]
    mac("fullInc" + w, usd(inc))
    mac("fullAgg" + w, usd(inc + mf.shift))
    mac("fullIncErr" + w, f"{100 * (inc / tv - 1):+.1f}")
    mac("fullAggErr" + w, f"{100 * ((inc + mf.shift) / tv - 1):+.1f}")
mac("minN", f"{min(v['n'] for v in stats.values()):,}")
mac("unboundedMax", f"{max(float(np.mean(np.isinf(np.array([r[f'profile|{T}'] for r in recs])[:, 1]))) for (e, Ryr), recs in cells.items() for T in (100, 1000)):.0%}")
print("full-record check:", {k: macros[k] for k in macros if k.startswith("full")}, "minN", macros["minN"], "unbounded", macros["unboundedMax"])

# ------------------------------------------------------------------------- write macros
with open(os.path.join(OUT, "numbers.tex"), "w") as f:
    f.write("% generated by paper/analysis.py -- do not edit\n")
    for k, v in sorted(macros.items()):
        f.write(f"\\newcommand{{\\{k}}}{{{re.sub(r'(?<!\\\\)%', r'\\%', v)}}}\n")
print(f"wrote {len(macros)} macros and 7 tables to {OUT}")
sys.exit(0)
for key in [("B", 30, 1000), ("B", 100, 1000), ("A", 30, 1000), ("B", 1000, 1000)]:
    print(key, {m: (round(100 * stats[key + (m,)]['cov'], 1), round(100 * stats[key + (m,)]['below'], 1),
                    round(stats[key + (m,)]['score'], 2), stats[key + (m,)].get('mcnemar_p')) for m in METHODS})
