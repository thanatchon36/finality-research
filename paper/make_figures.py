#!/usr/bin/env python3
"""Vector figures for the paper, sized to the 6.8 in text block of the arXiv layout and set in a
Times-compatible face (STIX) to match the body text. Reproduces run_study.py's seeds exactly, and
runs an automated overlap check (figlint.py) on every figure; exits non-zero if anything overlaps."""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
from figlint import lint  # noqa: E402
from finality import attacker, contagion, evt, lda  # noqa: E402

OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
R = json.load(open(os.path.join(ROOT, "results", "results.json")))
SEED, TRUTH_YEARS, BOOT = R["settings"]["seed"], R["settings"]["truth_years"], R["settings"]["boot"]

INK, MUTED, FAINT, GRID = "#122030", "#5a6b7e", "#8a99ab", "#e4eaf1"
HOT, AMBER, COOL, ACCENT, VIOLET = "#d1402a", "#e09a0c", "#0b9480", "#2f5fd6", "#5b3df0"
METHOD = {"delta": ("Delta method", HOT, "o", "-"), "bootstrap": ("Parametric bootstrap", AMBER, "D", "--"),
          "profile": ("Profile likelihood", VIOLET, "s", "-")}
W = 6.8   # text width of the arXiv layout, in inches

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 4.0, "axes.labelsize": 8, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
    "legend.fontsize": 6.9, "legend.frameon": True, "legend.framealpha": 0.92, "legend.edgecolor": "none",
    "legend.facecolor": "white", "legend.borderpad": 0.35, "legend.labelspacing": 0.3,
    "legend.handlelength": 1.7, "legend.handletextpad": 0.5, "legend.borderaxespad": 0.35,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": FAINT, "axes.linewidth": 0.7,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.3, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def usd(v, _=None):
    a = abs(v)
    if a >= 1e12:
        return f"${v / 1e12:g}T"
    if a >= 1e9:
        return f"${v / 1e9:g}B"
    if a >= 1e6:
        return f"${v / 1e6:g}M"
    return f"${v:g}"


MONEY = FuncFormatter(usd)
ISSUES: list[str] = []


def save(fig, name):
    ISSUES.extend(lint(fig, name))
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=170)
    plt.close(fig)


# ------------------------------------------------------------------------------------------ data
p = lda.LDAParams()
s_lda, s_truth, s_evt, _, _ = np.random.SeedSequence(SEED).spawn(5)
sim = lda.simulate(200_000, p, np.random.default_rng(s_lda))
truth_annual = np.sort(lda.simulate_annual(TRUTH_YEARS, p, np.random.default_rng(s_truth)))
rng = np.random.default_rng(s_evt)
long_rec = lda.simulate(1000, p, rng)
m_long = evt.fit_pot(long_rec.incidents, 1000, 0.98, annual=long_rec.annual)
me_u, me_e = evt.mean_excess(long_rec.incidents, 1 - np.logspace(np.log10(0.5), np.log10(10 / long_rec.incidents.size), 70))
hill = evt.hill(long_rec.incidents, kmax=int(0.3 * long_rec.incidents.size))
short = lda.simulate(100, p, rng)
m100 = evt.fit_pot(short.incidents, 100, 0.95, annual=short.annual)
t_min = 1.0 / -np.expm1(-m100.rate) * 1.02
Ts = np.logspace(np.log10(max(t_min, 2.0)), 4, 30)
draws = evt.rl_bootstrap(m100, Ts, B=max(BOOT, 399), rng=rng)
assert abs(m100.fit.xi - R["evt"]["record_100y"]["xi"]) < 1e-12, "seed replay drifted from results.json"
assert abs(m_long.fit.xi - R["evt"]["record_1000y"]["xi"]) < 1e-12

# ------------------------------------------------------------------------------------------ attacker
V = attacker.VENUES
fig, ax = plt.subplots(1, 3, figsize=(W, 2.3), gridspec_kw={"width_ratios": [1.05, 1, 1.1], "wspace": 0.55})
names = ["Bank core", "Exchange", "Bridge", "HW wallet"]
pay = np.array([v.payoff() for v in V]) / 1e6
ax[0].barh(names, pay, color=[HOT if x > 0 else COOL for x in pay], height=0.62)
ax[0].axvline(0, color=INK, lw=0.7)
ax[0].invert_yaxis()
ax[0].set_xlim(pay.min() - 38, pay.max() + 36)
for i, x in enumerate(pay):
    ax[0].text(max(x, 0) + 4, i, f"{x:+.1f}".replace("-", "\u2212"), va="center", ha="left", fontsize=7)
ax[0].set_xlabel(r"expected payoff $\Pi_v$ (\$M)")
ax[0].set_title("(a) Attacker payoff by venue")
bank = V[0]
g, cur = attacker.recovery_curve(bank)
be = bank.break_even_recovery()
ax[1].plot(g, cur / 1e6, color=INK)
ax[1].fill_between(g, 0, cur / 1e6, where=cur > 0, color=HOT, alpha=0.13, lw=0)
ax[1].axhline(0, color=FAINT, lw=0.7, ls="--")
ax[1].axvline(be, color=HOT, lw=0.9, ls=":")
ax[1].text(be - 0.04, 31, rf"$\kappa^*$ = {be:.3f}", color=HOT, ha="right", fontsize=7.2)
ax[1].plot([bank.p_recovery], [bank.payoff() / 1e6], "o", color=COOL, ms=4.5)
ax[1].annotate("bank today", xy=(bank.p_recovery, bank.payoff() / 1e6), xytext=(0.06, -14), fontsize=7, color=COOL,
               va="center", arrowprops=dict(arrowstyle="-|>", color=COOL, lw=0.7, shrinkA=2, shrinkB=4, mutation_scale=6))
ax[1].set_xlabel(r"bank recovery probability $\kappa$")
ax[1].set_ylabel(r"$\Pi_{\mathrm{bank}}$ (\$M)")
ax[1].set_title("(b) The finality lever")
dec = R["attacker"]["log_gap_bank_to_bridge"]
lab = ["Breach prob.", "Value at risk", "Finality", "Laundering"]
keys = ["breach probability", "value at risk", "finality (1 - P_recovery)", "laundering (1 - haircut)"]
vals = np.array([dec[k] for k in keys])
ax[2].barh(lab, vals, color=[VIOLET if l == "Finality" else FAINT for l in lab], height=0.62)
ax[2].axvline(0, color=INK, lw=0.7)
ax[2].invert_yaxis()
for i, v in enumerate(vals):
    ax[2].text(max(v, 0) + 0.09, i, f"{v / dec['total']:.0%}".replace("-", "\u2212"), va="center", ha="left", fontsize=7)
ax[2].set_xlim(vals.min() - 0.55, vals.max() + 0.6)
ax[2].set_xlabel(r"contribution to $\ln(\mathcal{R}_{\mathrm{bridge}}/\mathcal{R}_{\mathrm{bank}})$")
ax[2].set_title("(c) Decomposing the gap")
save(fig, "fig2_attacker")

# ------------------------------------------------------------------------------------------ loss model
fig, ax = plt.subplots(1, 2, figsize=(W, 2.35), gridspec_kw={"wspace": 0.28})
cut = np.quantile(sim.annual, 0.995)
ax[0].hist(sim.annual[sim.annual < cut] / 1e9, bins=80, color=ACCENT, alpha=0.85)
for q, c in ((0.95, AMBER), (0.99, HOT)):
    v = np.quantile(sim.annual, q) / 1e9
    ax[0].axvline(v, ymax=0.66, color=c, ls="--", lw=0.9, label=f"VaR {q:.0%}: \\${v:.2f}B")
ax[0].set_xlabel("annual loss (\\$B)")
ax[0].set_ylabel("simulated years")
ax[0].set_title("(a) Annual losses below the 99.5th percentile")
ax[0].legend(loc="upper right")
s = np.sort(sim.annual)
surv = 1.0 - np.arange(1, s.size + 1) / s.size
keep = (s > 1e7) & (surv > 0)
ax[1].loglog(s[keep][::20], surv[keep][::20], color=VIOLET, lw=1.2, label="simulated, $2\\times10^5$ years")
x0 = np.logspace(np.log10(3e9), np.log10(s.max()), 20)
ref = surv[np.searchsorted(s, 3e9)] * (x0 / 3e9) ** (-p.tail_alpha)
ax[1].loglog(x0, ref, color=INK, ls=":", lw=1, label=rf"reference slope $-\alpha = -{p.tail_alpha}$")
ylo = 2e-6
s20 = surv[np.searchsorted(s, 20e9)]
ax[1].vlines(20e9, ylo, s20, colors=HOT, linestyles="--", lw=0.9, label="\\$20B: once per %s years" % f"{R['truth']['systemic_return_period']:,.0f}")
ax[1].plot([20e9], [s20], "o", color=HOT, ms=3)
ax[1].set_ylim(ylo, 1.6)
ax[1].xaxis.set_major_formatter(MONEY)
ax[1].set_xlabel("annual loss $x$")
ax[1].set_ylabel(r"$P(S > x)$")
ax[1].set_title("(b) Exceedance curve")
ax[1].legend(loc="lower left")
save(fig, "fig3_loss")

# ------------------------------------------------------------------------------------------ single-record EVT
fig = plt.figure(figsize=(W, 5.05))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.35], hspace=0.5, wspace=0.27)
a0, a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
a0.plot(me_u, me_e, "o-", color=ACCENT, ms=2, lw=0.9)
a0.axvline(m_long.u, color=VIOLET, ls=":", lw=0.9)
a0.text(0.31, 0.88, rf"$\hat\xi$ = {m_long.fit.xi:.2f} at $u$ (98th pct.)", transform=a0.transAxes, color=VIOLET, fontsize=7.2)
a0.xaxis.set_major_formatter(MONEY)
a0.yaxis.set_major_formatter(MONEY)
a0.set_xlabel("threshold $u$")
a0.set_ylabel("mean excess $e(u)$")
a0.set_title("(a) Mean excess, 1,000-year record")
k = np.arange(1, hill.size + 1)
sel = k >= 10
a1.fill_between(k[sel], hill[sel] * (1 - 1.96 / np.sqrt(k[sel])), hill[sel] * (1 + 1.96 / np.sqrt(k[sel])), color=ACCENT, alpha=0.14, lw=0)
a1.plot(k[sel], hill[sel], color=ACCENT, lw=1, label=r"Hill $\hat\alpha_k$")
a1.axhline(p.tail_alpha, color=COOL, ls="--", lw=1, label=rf"planted $\alpha$ = {p.tail_alpha}")
a1.axvline(m_long.k, ymax=0.6, color=VIOLET, ls=":", lw=0.9, label=f"$k$ at $u$ ({m_long.k})")
a1.set_xscale("log")
a1.set_ylim(0, 3.6)
a1.set_xlabel("$k$ largest incidents")
a1.set_ylabel(r"tail index $\hat\alpha$")
a1.set_title("(b) Hill plot, same record")
a1.legend(loc="upper right", ncol=1)
truth_T = np.logspace(0.05, 4, 90)
a2.plot(truth_T, np.quantile(truth_annual, 1 - 1 / truth_T), color=FAINT, lw=2.4, label=f"truth ({TRUTH_YEARS / 1e6:g} million simulated years)")
rec = np.sort(short.annual)[::-1]
a2.plot(101 / np.arange(1, rec.size + 1), rec, "o", color=ACCENT, ms=2.4, label="the 100-year record")
bands = {"delta": np.array([evt.rl_ci_delta(m100, T) for T in Ts]).T, "bootstrap": evt.percentile_ci(draws),
         "profile": np.array([evt.rl_ci_profile(m100, T) for T in Ts]).T}
lo, hi = bands["profile"]
a2.fill_between(Ts, lo, np.where(np.isfinite(hi), hi, 1e16), color=VIOLET, alpha=0.15, lw=0, label="profile likelihood 95%")
for m in ("delta", "bootstrap"):
    name, col, _, ls = METHOD[m]
    a2.plot(Ts, bands[m][0], ls, color=col, lw=1)
    a2.plot(Ts, bands[m][1], ls, color=col, lw=1, label=f"{name.lower()} 95%")
a2.plot(Ts, evt.return_level(m100, Ts), color=VIOLET, lw=1.6, label=r"point estimate $\hat\psi_T + \bar s$")
a2.axvline(100, color=AMBER, ls=":", lw=0.9)
a2.text(110, 1.25e8, "end of record", color=AMBER, fontsize=7)
a2.set_xscale("log")
a2.set_yscale("log")
a2.set_xlim(1, 1e4)
a2.set_ylim(8e7, 5e12)
a2.yaxis.set_major_formatter(MONEY)
a2.set_xlabel("return period $T$ (years)")
a2.set_ylabel("annual loss")
a2.set_title("(c) Return levels extrapolated from one 100-year record")
a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
save(fig, "fig4_evt")

# ------------------------------------------------------------------------------------------ coverage
rows = R["coverage"]
fig, ax = plt.subplots(2, 2, figsize=(W, 5.0), sharex="col", gridspec_kw={"height_ratios": [1.3, 1], "hspace": 0.3, "wspace": 0.25})
for c, (code, title) in enumerate((("A", "Experiment A: exact GPD"), ("B", "Experiment B: loss pipeline"))):
    sub = [r for r in rows if r["experiment"] == code]
    Rs = sorted({r["record_years"] for r in sub})
    x = np.arange(len(Rs))
    for j, m in enumerate(("delta", "bootstrap", "profile")):
        name, col, mk, _ = METHOD[m]
        off = (j - 1) * 0.17
        for T in (100, 1000):
            pts = [next(r for r in sub if r["record_years"] == Rr and r["T"] == T and r["method"] == m) for Rr in Rs]
            cov = np.array([q["coverage"] for q in pts]) * 100
            se = np.array([q["mc_se"] for q in pts]) * 100
            big = T == 1000
            ax[0, c].errorbar(x + off + (0 if big else 0.06), cov, yerr=1.96 * se, fmt=mk, ms=4 if big else 3, color=col,
                              mfc=col if big else "white", mew=0.8, capsize=1.5, lw=0.8, label=f"{name}, $T$ = {T:,}" if c == 0 else None)
            if big:
                ax[1, c].plot(x + off, [q["median_width_ratio"] for q in pts], mk + "-", color=col, ms=3.5, lw=0.9)
    ax[0, c].axhline(95, color=INK, ls="--", lw=0.8, label="nominal 95% level" if c == 0 else None)
    ax[0, c].set_ylim(50, 101.5)
    ax[0, c].set_title(f"({'ab'[c]}) {title}")
    ax[0, c].set_ylabel("coverage (%)")
    ks = [np.mean([r["mean_k"] for r in sub if r["record_years"] == Rr]) for Rr in Rs]
    ax[1, c].set_xticks(x, [f"{Rr:,} yrs\n" + rf"($\bar k$ = {kk:.0f})" for Rr, kk in zip(Rs, ks)])
    ax[1, c].set_yscale("log")
    ax[1, c].set_ylabel("median upper / lower")
    ax[1, c].set_xlabel("record length")
    ax[1, c].set_title(f"({'cd'[c]}) Interval width at $T$ = 1,000")
ax[0, 0].legend(loc="lower right", ncol=1, fontsize=6.6)
save(fig, "fig5_coverage")

# ------------------------------------------------------------------------------------------ contagion
net = contagion.build_network(7)
big, vend = net.largest(), net.vendor_clients(0.3)
levs = np.linspace(0.1, 2.0, 77)
crit = net.critical_exposure
rank = contagion.systemic_ranking(net, 0.5, 0.2)
fig, ax = plt.subplots(1, 3, figsize=(W, 2.3), gridspec_kw={"wspace": 0.36})
ax[0].axvspan(crit, levs.max(), color=HOT, alpha=0.07, lw=0)
ax[0].plot(levs, contagion.phase_sweep(net, [big], 0.2, levs), color=INK, label="largest bank")
ax[0].plot(levs, contagion.phase_sweep(net, vend, 0.2, levs), color=VIOLET, label="vendor")
ax[0].axvline(crit, color=HOT, ls="--", lw=0.9)
ax[0].set_xticks([0.5, crit, 1.5, 2.0], ["0.5", r"$\ell^*$", "1.5", "2.0"])
ax[0].set_yscale("log")
ax[0].set_xlabel(r"exposure scale $\ell$")
ax[0].set_ylabel(r"multiplier $\mathcal{M}$")
ax[0].set_title("(a) Phase transition")
ax[0].legend(loc="lower right", handlelength=1.3)
col = np.where(net.core, ACCENT, AMBER)
ax[1].scatter(net.ta, rank, s=8 + 36 * np.sqrt(net.ta / net.ta.max()), c=col, edgecolor="white", lw=0.4, zorder=3)
for i in np.argsort(-rank)[:3]:
    ax[1].annotate(net.names[i], (net.ta[i], rank[i]), xytext=(4.5, 3.5), textcoords="offset points", fontsize=7, weight="bold")
ax[1].set_xscale("log")
ax[1].xaxis.set_major_formatter(MONEY)
ax[1].set_xlabel("total assets")
ax[1].set_ylabel(r"multiplier $\mathcal{M}$")
ax[1].set_title("(b) Importance is not size")
ax[1].set_ylim(0.85, rank.max() * 1.2)
tot = net.eq.sum()
for (lab_, h), c in zip({f"{net.names[big]}, $\\ell$ = 0.5": contagion.debtrank(net, 0.5, [big], 0.2),
                         "vendor, $\\ell$ = 0.5": contagion.debtrank(net, 0.5, vend, 0.2),
                         "vendor, $\\ell$ = 1.2": contagion.debtrank(net, 1.2, vend, 0.2)}.items(), (INK, VIOLET, HOT)):
    ax[2].plot(np.arange(h.shape[0]), h @ net.eq / tot * 100, color=c, label=lab_)
ax[2].set_xlabel("propagation round")
ax[2].set_ylabel("system equity lost (%)")
ax[2].set_title("(c) Cascades")
ax[2].legend(loc="upper left")
save(fig, "fig6_contagion")

print("figures written to", OUT)
print(f"overlap check: {len(ISSUES)} issue(s)")
for i in ISSUES:
    print("  -", i)
if ISSUES:
    sys.exit(1)
