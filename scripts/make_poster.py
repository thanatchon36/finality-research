"""LinkedIn poster, ``docs/assets/linkedin_poster.png`` (1440 x 1800), in the format of the thainowcast poster.

The chart is real output of the study: the first 50 successfully fitted records of coverage cell B (full loss
pipeline, 30-year records, seed 20261108), read from ``results/cache/``, with the delta-method and
profile-likelihood 95% intervals for the 1-in-1,000-year annual loss against the true value from 2,000,000
simulated years. Every number on the poster is read from ``results/results.json`` or the cache.
Fira Sans (SIL OFL) is downloaded once into ``.cache/fonts/``.

Usage: ``python scripts/make_poster.py``.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "linkedin_poster.png"
FONTS = ROOT / ".cache" / "fonts"
W, H = 1440, 1800
BG, PANEL, RULE = "#03060d", "#0a1222", "#1c2a45"
AMBER, PALE, WHITE, MUTED, DIM = "#f5a623", "#ffe8a3", "#eef2f8", "#9fb0d0", "#6b7a99"
HOT, COOL = "#ff5a36", "#3d5175"
N_SHOW = 50


def fonts() -> None:
    FONTS.mkdir(parents=True, exist_ok=True)
    for w in ("Regular", "Medium", "SemiBold", "Bold"):
        f = FONTS / f"FiraSans-{w}.ttf"
        if not f.exists():
            urllib.request.urlretrieve(
                f"https://raw.githubusercontent.com/google/fonts/main/ofl/firasans/FiraSans-{w}.ttf", f)
        font_manager.fontManager.addfont(str(f))
    plt.rcParams["font.family"] = "Fira Sans"


def usd(v: float) -> str:
    return f"${v / 1e12:.0f}T" if v >= 1e12 else (f"${v / 1e9:.1f}B" if v >= 1e9 else f"${v / 1e6:.0f}M")


def main() -> int:
    fonts()
    R = json.loads((ROOT / "results" / "results.json").read_text())
    truth = R["truth"]["annual_T1000"]
    cell = {(c["experiment"], c["record_years"], c["T"], c["method"]): c for c in R["coverage"]}
    delta, prof = cell[("B", 30, 1000, "delta")], cell[("B", 30, 1000, "profile")]
    if delta["miss_high"] != 0.0:
        raise SystemExit("the poster says every delta miss is too low; the results disagree")
    miss = 1 - delta["coverage"]
    recs = [r for r in json.loads(next((ROOT / "results" / "cache").glob("B_30y_*.json")).read_text()) if r][:N_SHOW]
    recs.sort(key=lambda r: r["point|1000"])
    n_records = sum(len(json.loads(p.read_text())) for p in (ROOT / "results" / "cache").glob("*.json"))

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)

    def text(x, y, s, size, color=WHITE, weight="regular", **kw):
        return fig.text(x / W, 1 - y / H, s, fontsize=size, color=color, fontweight=weight, va="baseline", **kw)

    # header and hero
    text(94, 92, "FINALITY", 19, AMBER, "semibold")
    text(368, 92, "open-source research on cyber tail risk in banking", 18, MUTED)
    text(78, 362, f"{miss:.0%}", 178, AMBER, "semibold")
    text(104, 438, "of the time", 44, PALE)
    for i, line in enumerate(("textbook 95% intervals built on", "30 years of data miss the", "1-in-1,000-year cyber loss.")):
        text(104, 500 + 49 * i, line, 23, WHITE)
    text(104, 668, "And every single miss", 25, AMBER, "medium")
    text(104, 719, "is too low.", 25, AMBER, "medium")

    # chart: real intervals from the coverage study
    x0, x1, y0, y1 = 760, 1380, 175, 1250
    ax = fig.add_axes([x0 / W, 1 - y1 / H, (x1 - x0) / W, (y1 - y0) / H], facecolor=BG)
    lo_x, hi_x = 3e8, 2e14
    ax.set_xscale("log")
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(-1, N_SHOW)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=DIM, length=0, labelsize=13, pad=8)
    ax.set_yticks([])
    ax.set_xticks([1e9, 1e10, 1e11, 1e12, 1e13, 1e14], ["$1B", "$10B", "$100B", "$1T", "$10T", "$100T"])
    for t in ax.get_xticklabels():
        t.set_fontfamily("Fira Sans")
    for g in (1e9, 1e10, 1e11, 1e12, 1e13, 1e14):
        ax.axvline(g, color=RULE, lw=0.8, zorder=0)
    ax.axvline(truth, color=PALE, lw=1.6, ls=(0, (4, 3)), zorder=1)
    for i, r in enumerate(recs):
        y = N_SHOW - 1 - i
        dlo, dhi = r["delta|1000"]
        plo, phi = r["profile|1000"]
        missed = dhi < truth
        ax.plot([dlo, dhi], [y + 0.17] * 2, color=HOT if missed else COOL, lw=3.4, solid_capstyle="round", zorder=3)
        clipped = not np.isfinite(phi) or phi > hi_x
        ax.plot([plo, min(phi, hi_x * 0.92)], [y - 0.2] * 2, color=PALE, lw=1.4, alpha=0.85,
                solid_capstyle="round", zorder=2)
        if clipped:
            ax.plot(hi_x * 0.92, y - 0.2, marker=">", ms=4, color=PALE, alpha=0.85, zorder=2)
        ax.plot(r["point|1000"], y + 0.17, "o", ms=3.6, color=WHITE if not missed else PALE, zorder=4)
    text(x0, 128, f"{N_SHOW} simulated 30-year loss records, each with its 95% intervals", 14.5, MUTED)
    text(x0, 152, "for the 1-in-1,000-year loss", 14.5, MUTED)
    tx = (np.log10(truth) - np.log10(lo_x)) / (np.log10(hi_x) - np.log10(lo_x)) * (x1 - x0) + x0
    text(tx + 8, 196, f"true value  {usd(truth)}", 15, PALE, "medium")

    # callouts
    def callout(y, head, color, lines, swatch_lw):
        text(712, y, head, 21, color, "semibold", ha="right")
        fig.add_artist(plt.Line2D([724 / W, 748 / W], [1 - (y - 7) / H] * 2, color=color, lw=swatch_lw,
                                  solid_capstyle="round"))
        for j, ln in enumerate(lines):
            text(712, y + 36 + 31 * j, ln, 16.5, WHITE if color != DIM else MUTED, ha="right")

    callout(812, "Textbook interval", HOT, [f"delta method: covers the truth", f"only {delta['coverage']:.0%} of the time"], 4)
    callout(972, "Profile likelihood", PALE, [f"covers it {prof['coverage']:.0%} of the time,", "honest about what 30 years can say"], 2)
    callout(1132, "The estimate itself", MUTED, [f"median {delta['median_point_over_truth']:.2f}× the true loss:",
                                                 "short records hide the tail"], 0.01)

    # stats row
    fig.add_artist(plt.Line2D([94 / W, 1346 / W], [1 - 1326 / H] * 2, color=RULE, lw=1.2))
    dec = R["attacker"]
    stats = [(f"{n_records:,}", ["simulated records, each", "scored against the truth"]),
             (f"{dec['finality_share_of_gap']:.0%}", ["of the attacker's payoff", "gap is finality"]),
             (f"{R['contagion']['critical_exposure']:.2f}×", ["equity: where interbank", "contagion turns explosive"]),
             (f"{R['truth']['years'] / 1e6:.0f}M", ["simulated years", "behind the truth"])]
    for k, (num, lab) in enumerate(stats):
        x = 94 + 324 * k
        text(x, 1432, num, 46, AMBER, "semibold")
        for j, ln in enumerate(lab):
            text(x, 1482 + 30 * j, ln, 15.5, MUTED)

    # links box
    fig.add_artist(FancyBboxPatch((86 / W, 1 - 1730 / H), 1268 / W, 184 / H, boxstyle="round,pad=0,rounding_size=0.012",
                                  facecolor=PANEL, edgecolor=RULE, lw=1.4, transform=fig.transFigure))
    text(130, 1604, "Code + paper", 14.5, MUTED)
    text(130, 1640, "github.com/thanatchon36/finality-research", 19, WHITE)
    text(764, 1604, "Dataset", 14.5, MUTED)
    text(764, 1640, "hf.co/datasets/thanatchon36/finality-research", 19, WHITE)
    text(130, 1690, "All parameters are illustrative: the method is the contribution, the numbers are not a forecast.",
         13, MUTED)

    text(94, 1770, "Thanatchon Chongmankhong  ·  views are my own  ·  DOI 10.5281/zenodo.24050729", 14, MUTED)
    text(1346, 1770, f"first {N_SHOW} records of coverage cell B (30-year records) · seed 20261108", 11.5, DIM, ha="right")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=BG, dpi=100)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
