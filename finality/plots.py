"""Publication figures. Palette matches the FINALITY web instrument."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

PAL = dict(ink="#122030", muted="#5a6b7e", faint="#8a99ab", grid="#e4eaf1", hot="#e5472d",
           amber="#f3a712", cool="#0ea894", accent="#3366e6", violet="#6d4dff")
METHOD = {"delta": ("Delta method", PAL["hot"], "o"),
          "profile": ("Profile likelihood", PAL["violet"], "s"),
          "bootstrap": ("Parametric bootstrap", PAL["amber"], "D")}


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 9.5,
        "font.family": "DejaVu Sans", "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.axisbelow": True,
        "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": PAL["faint"], "axes.labelcolor": PAL["ink"], "text.color": PAL["ink"],
        "xtick.color": PAL["muted"], "ytick.color": PAL["muted"], "axes.grid": True,
        "grid.color": PAL["grid"], "grid.linewidth": 0.8, "legend.frameon": False, "legend.fontsize": 8.5,
    })


def usd(v, _=None) -> str:
    a = abs(v)
    if a >= 1e12:
        return f"${v / 1e12:g}T"
    if a >= 1e9:
        return f"${v / 1e9:g}B"
    if a >= 1e6:
        return f"${v / 1e6:g}M"
    return f"${v:g}"


MONEY = FuncFormatter(usd)


def fig_attacker(venues, decomposition: dict, recovery: tuple, path: str) -> None:
    style()
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.15, 1, 1.1]})
    names = [v.name for v in venues]
    pay = np.array([v.payoff() for v in venues]) / 1e6
    ax[0].barh(names, pay, color=[PAL["hot"] if p > 0 else PAL["cool"] for p in pay])
    ax[0].axvline(0, color=PAL["ink"], lw=0.8)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("attacker's risk-adjusted expected payoff ($M)")
    ax[0].set_xlim(pay.min() - 28, pay.max() + 22)
    ax[0].set_title("(a) Most money at stake, worst payoff")
    for i, p in enumerate(pay):
        ax[0].text(p + (3 if p >= 0 else -3), i, f"{p:+.1f}", va="center", ha="left" if p >= 0 else "right", fontsize=8.5)

    grid, curve = recovery
    bank = venues[0]
    ax[1].plot(grid, curve / 1e6, color=PAL["ink"], lw=2)
    ax[1].axhline(0, color=PAL["faint"], lw=0.8, ls="--")
    ax[1].fill_between(grid, 0, curve / 1e6, where=curve > 0, color=PAL["hot"], alpha=0.12)
    be = bank.break_even_recovery()
    ax[1].axvline(be, color=PAL["hot"], lw=1.2, ls=":")
    ax[1].annotate(f"break-even\nP(recovery) = {be:.2f}", (be, 0), xytext=(be - 0.38, 30), fontsize=8.5,
                   color=PAL["hot"], arrowprops=dict(arrowstyle="-", color=PAL["hot"], lw=0.8))
    ax[1].plot([bank.p_recovery], [bank.payoff() / 1e6], "o", color=PAL["cool"], ms=7)
    ax[1].set_xlabel("bank's P(recovery / clawback)")
    ax[1].set_ylabel("attacker payoff ($M)")
    ax[1].set_title("(b) The finality lever")

    parts = {k: v for k, v in decomposition.items() if k != "total"}
    labels, vals = list(parts), np.array(list(parts.values()))
    ax[2].barh(labels, vals, color=[PAL["violet"] if "finality" in l else PAL["faint"] for l in labels])
    ax[2].axvline(0, color=PAL["ink"], lw=0.8)
    ax[2].invert_yaxis()
    tot = decomposition["total"]
    for i, v in enumerate(vals):
        ax[2].text(v + (0.03 if v >= 0 else -0.03), i, f"{v:+.2f} ({v / tot:.0%})", va="center",
                   ha="left" if v >= 0 else "right", fontsize=8.5)
    ax[2].set_xlabel(f"contribution to ln(retained value, bridge / bank) = {tot:+.2f}")
    ax[2].set_title(f"(c) Bank vs bridge: finality is {decomposition['finality (1 - P_recovery)'] / tot:.0%} of the gap")
    ax[2].set_xlim(vals.min() - 1.1, vals.max() + 1.0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_lda(annual: np.ndarray, systemic: float, path: str) -> None:
    style()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    cut = np.quantile(annual, 0.995)
    ax[0].hist(annual[annual < cut] / 1e9, bins=90, color=PAL["accent"], alpha=0.85)
    for q, c in ((0.95, PAL["amber"]), (0.99, PAL["hot"])):
        v = np.quantile(annual, q) / 1e9
        ax[0].axvline(v, color=c, ls="--", lw=1.2, label=f"VaR {q:.0%} = ${v:.2f}B")
    ax[0].set_xlabel("annual banking-core cyber loss ($B)")
    ax[0].set_ylabel("simulated years")
    ax[0].set_title("(a) Ordinary years, below the 99.5th percentile")
    ax[0].legend()
    s = np.sort(annual)
    surv = 1.0 - np.arange(1, s.size + 1) / s.size
    keep = (s > 1e7) & (surv > 0)
    step = max(1, keep.sum() // 4000)
    ax[1].loglog(s[keep][::step], surv[keep][::step], color=PAL["violet"], lw=1.8)
    ax[1].axvline(systemic, color=PAL["hot"], ls="--", lw=1.2, label=f"systemic threshold {usd(systemic)}")
    ax[1].xaxis.set_major_formatter(MONEY)
    ax[1].set_xlabel("annual loss x")
    ax[1].set_ylabel("P(annual loss > x)")
    ax[1].set_title("(b) Exceedance curve: a power-law tail")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_evt(diag: dict, demo: dict, path: str) -> None:
    """diag: mean-excess and Hill inputs from a long record; demo: return-level bands for a short record."""
    style()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1, 1, 1.45]})
    u_me, e_me = diag["me_u"], diag["me_e"]
    ax[0].plot(u_me, e_me, "o-", color=PAL["accent"], ms=3, lw=1.2)
    m = diag["model"]
    if m.fit.xi < 1:
        xs = np.linspace(m.u, u_me.max(), 50)
        ax[0].plot(xs, (m.fit.sigma + m.fit.xi * (xs - m.u)) / (1 - m.fit.xi), color=PAL["violet"], lw=2,
                   label=rf"GPD fit above $u$ ($\hat\xi$ = {m.fit.xi:.2f})")
    else:
        ax[0].text(0.03, 0.9, rf"$\hat\xi$ = {m.fit.xi:.2f} $\geq$ 1: the fitted mean excess is infinite",
                   transform=ax[0].transAxes, fontsize=8.5, color=PAL["violet"])
    ax[0].axvline(m.u, color=PAL["violet"], ls=":", lw=1.2)
    ax[0].xaxis.set_major_formatter(MONEY)
    ax[0].yaxis.set_major_formatter(MONEY)
    ax[0].set_xlabel("threshold u")
    ax[0].set_ylabel("mean excess over u")
    ax[0].set_title(f"(a) Mean excess, {diag['years']:,}-year record")
    if ax[0].get_legend_handles_labels()[0]:
        ax[0].legend(loc="upper left")

    a = diag["hill"]
    k = np.arange(1, a.size + 1)
    sel = k >= 10
    ax[1].fill_between(k[sel], a[sel] * (1 - 1.96 / np.sqrt(k[sel])), a[sel] * (1 + 1.96 / np.sqrt(k[sel])),
                       color=PAL["accent"], alpha=0.14, lw=0)
    ax[1].plot(k[sel], a[sel], color=PAL["accent"], lw=1.4, label="Hill estimate")
    ax[1].axhline(diag["alpha_true"], color=PAL["cool"], ls="--", lw=1.4, label=rf"planted $\alpha$ = {diag['alpha_true']:.1f}")
    ax[1].axvline(m.k, color=PAL["violet"], ls=":", lw=1.2, label=rf"$k$ at threshold $u$ ({m.k})")
    ax[1].set_xscale("log")
    ax[1].set_ylim(0, 4)
    ax[1].set_xlabel("k largest incidents (log)")
    ax[1].set_ylabel(r"tail index $\hat\alpha$")
    ax[1].set_title("(b) Hill plot settles near the planted index")
    ax[1].legend(loc="upper right")

    Ts = demo["Ts"]
    ax[2].plot(demo["truth_T"], demo["truth_x"], color=PAL["faint"], lw=2.4, label="truth (long simulation)")
    rec = np.sort(demo["record_annual"])[::-1]
    Rn = rec.size
    ax[2].plot((Rn + 1) / np.arange(1, Rn + 1), rec, "o", color=PAL["accent"], ms=3.2, label=f"your {Rn}-year record")
    for meth, ls in (("delta", "-"), ("bootstrap", "--")):
        lab, col, _ = METHOD[meth]
        lo, hi = demo["bands"][meth]
        ax[2].plot(Ts, lo, ls, color=col, lw=1.3)
        ax[2].plot(Ts, hi, ls, color=col, lw=1.3, label=f"{lab} 95%")
    lo, hi = demo["bands"]["profile"]
    ax[2].fill_between(Ts, lo, np.where(np.isfinite(hi), hi, 1e16), color=PAL["violet"], alpha=0.16, lw=0,
                       label="Profile likelihood 95%")
    ax[2].plot(Ts, demo["point"], color=PAL["violet"], lw=2.2, label="GPD point estimate")
    ax[2].axvline(Rn, color=PAL["amber"], ls=":", lw=1.2)
    ax[2].text(Rn * 1.08, demo["ylim"][0] * 1.4, "end of record", color=PAL["amber"], fontsize=8.5)
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xlim(1, 1e4)
    ax[2].set_ylim(*demo["ylim"])
    ax[2].yaxis.set_major_formatter(MONEY)
    ax[2].set_xlabel("return period T, years (log)")
    ax[2].set_ylabel("annual loss")
    ax[2].set_title(f"(c) Extrapolating a {Rn}-year record to 1-in-10,000 years")
    ax[2].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_coverage(rows: list[dict], path: str, nominal: float = 0.95) -> None:
    style()
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8), sharex="col", gridspec_kw={"height_ratios": [1.35, 1]})
    exps = [("A", "Experiment A: exceedances exactly GPD"), ("B", "Experiment B: cyber loss pipeline")]
    for c, (code, title) in enumerate(exps):
        sub = [r for r in rows if r["experiment"] == code]
        Rs = sorted({r["record_years"] for r in sub})
        Ts = sorted({r["T"] for r in sub})
        x = np.arange(len(Rs))
        for j, meth in enumerate(("delta", "bootstrap", "profile")):
            lab, col, mk = METHOD[meth]
            off = (j - 1) * 0.16
            for T in Ts:
                pts = [next((r for r in sub if r["record_years"] == R and r["T"] == T and r["method"] == meth), None) for R in Rs]
                if any(p is None for p in pts):
                    continue
                cov = np.array([p["coverage"] for p in pts])
                se = np.array([p["mc_se"] for p in pts])
                big = T == max(Ts)
                ax[0, c].errorbar(x + off + (0 if big else 0.05), cov * 100, yerr=1.96 * se * 100, fmt=mk, ms=7 if big else 5,
                                  color=col, mfc=col if big else "white", capsize=2, lw=1,
                                  label=f"{lab}, T = {T:,}" if c == 0 else None)
                if big:
                    w = np.array([p["median_width_ratio"] for p in pts])
                    ax[1, c].plot(x + off, w, mk + "-", color=col, ms=6, lw=1.2)
        ax[0, c].axhline(nominal * 100, color=PAL["ink"], ls="--", lw=1)
        ax[0, c].text(-0.42, nominal * 100 + 0.9, "nominal 95%", fontsize=8.5, ha="left")
        ax[0, c].set_ylim(40, 102)
        ax[0, c].set_title(f"({'a' if c == 0 else 'b'}) {title}")
        ax[0, c].set_ylabel("coverage of the truth (%)")
        ks = [np.mean([r["mean_k"] for r in sub if r["record_years"] == R]) for R in Rs]
        ax[1, c].set_xticks(x, [f"{R:,} yrs\n(k ≈ {k:.0f})" for R, k in zip(Rs, ks)])
        ax[1, c].set_yscale("log")
        ax[1, c].set_ylabel("median width, upper / lower")
        ax[1, c].set_title(f"({'c' if c == 0 else 'd'}) Price of honesty: interval width at T = {max(Ts):,}")
        ax[1, c].set_xlabel("record length (exceedances used)")
    ax[0, 0].legend(loc="lower right", fontsize=7.8, ncol=1)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_contagion(net, levs, sweeps: dict, ranking: np.ndarray, cascades: dict, path: str) -> None:
    style()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    crit = net.critical_exposure
    ax[0].axvspan(crit, levs.max(), color=PAL["hot"], alpha=0.06, lw=0)
    for (lab, ys), col in zip(sweeps.items(), (PAL["ink"], PAL["violet"])):
        ax[0].plot(levs, ys, lw=2, color=col, label=lab)
    ax[0].axvline(crit, color=PAL["hot"], ls="--", lw=1.2)
    ax[0].text(crit * 1.03, 1.15, rf"$\rho(\Lambda)$ = 1 at {crit:.2f}$\times$", color=PAL["hot"], fontsize=9)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("exposure / equity")
    ax[0].set_ylabel("contagion multiplier (total / direct loss)")
    ax[0].set_title(r"(a) Phase transition at $\rho(\Lambda) = 1$")
    ax[0].legend(loc="upper left")

    col = np.where(net.core, PAL["accent"], PAL["amber"])
    ax[1].scatter(net.ta, ranking, s=18 + 60 * np.sqrt(net.ta / net.ta.max()), c=col, edgecolor="white", lw=0.6, zorder=3)
    for i in np.argsort(-ranking)[:3]:
        ax[1].annotate(net.names[i], (net.ta[i], ranking[i]), xytext=(5, 4), textcoords="offset points", fontsize=9, weight="bold")
    ax[1].set_xscale("log")
    ax[1].xaxis.set_major_formatter(MONEY)
    ax[1].set_xlabel("total assets (log)")
    ax[1].set_ylabel("multiplier when this bank alone is hit")
    ax[1].set_title("(b) Systemic importance is not size")
    ax[1].scatter([], [], c=PAL["accent"], label="core")
    ax[1].scatter([], [], c=PAL["amber"], label="periphery")
    ax[1].legend(loc="upper left")

    tot = net.eq.sum()
    for (lab, hist), c in zip(cascades.items(), (PAL["ink"], PAL["violet"], PAL["hot"])):
        loss = hist @ net.eq / tot * 100
        ax[2].plot(np.arange(loss.size), loss, "-", color=c, lw=2, label=lab)
    ax[2].set_xlabel("round (0 = before the attack, 1 = direct hit)")
    ax[2].set_ylabel("share of all bank equity lost (%)")
    ax[2].set_title("(c) Cascades, round by round")
    ax[2].legend(loc="center right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
