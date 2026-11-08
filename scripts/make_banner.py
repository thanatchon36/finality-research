"""README banner, ``docs/assets/banner.png``, drawn from the paper's own interbank network.

The left panel is the core-periphery network of ``finality.contagion.build_network`` (the seed used in the
paper and the web instrument) after a cyber loss at the largest core bank, propagated by linear DebtRank at a
exposure scale of 1.05, above the critical 0.87. Node area is total assets, colour is final distress (amber = wiped
out), edges are interbank exposures. Same layout and palette as the thainowcast banner.

Usage: ``python scripts/make_banner.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finality.contagion import build_network, debtrank  # noqa: E402

BG, AMBER, PALE, MUTED, EDGE = "#03060d", "#f5a623", "#ffe8a3", "#9fb0d0", "#2b3c5c"
OUT = ROOT / "docs" / "assets" / "banner.png"


def layout(net, rng: np.random.Generator) -> np.ndarray:
    """Core banks on an inner ring, periphery on a jittered outer ring, ordered by their main core lender."""
    n_core = int(net.core.sum())
    pos = np.zeros((net.n, 2))
    for i in range(n_core):
        a = 2 * np.pi * i / n_core + 0.3
        pos[i] = 0.32 * np.cos(a), 0.32 * np.sin(a)
    periph = np.arange(n_core, net.n)
    anchor = np.argmax(net.W[periph][:, :n_core] + net.W[:n_core, periph].T, axis=1)
    order = periph[np.lexsort((net.ta[periph], anchor))]
    for k, i in enumerate(order):
        a = 2 * np.pi * k / len(order) + rng.normal(0, 0.04)
        r = 0.86 + rng.normal(0, 0.05)
        pos[i] = r * np.cos(a), r * np.sin(a)
    return pos


def main() -> int:
    net = build_network(seed=7)
    seed_bank = int(np.argmax(net.ta))
    hist = debtrank(net, lev=1.05, seeds=[seed_bank], shock=0.6)
    h = hist[-1]
    pos = layout(net, np.random.default_rng(20261108))
    cmap = LinearSegmentedColormap.from_list("heat", ["#1a2a4a", "#b0560f", AMBER, PALE])

    fig = plt.figure(figsize=(16.8, 6.24), dpi=100, facecolor=BG)
    ax = fig.add_axes([0.03, 0.04, 0.30, 0.92], facecolor=BG)
    ax.set_aspect("equal")
    ax.axis("off")
    w = net.W * net.ta[:, None]
    w = w / w.max()
    for i, j in zip(*np.nonzero(net.W)):
        heat = 0.5 * (h[i] + h[j])
        ax.plot(*pos[[i, j]].T, color=AMBER if heat > 0.5 else EDGE,
                alpha=float(0.08 + 0.55 * np.sqrt(w[i, j])) * (0.9 if heat > 0.5 else 0.6),
                lw=0.5 + 2.2 * np.sqrt(w[i, j]), solid_capstyle="round", zorder=1)
    size = 40 + 2600 * np.sqrt(net.ta / net.ta.max())
    ax.scatter(*pos.T, s=size * 3.2, c=h, cmap=cmap, vmin=0, vmax=1, alpha=0.12, lw=0, zorder=2)  # glow
    ax.scatter(*pos.T, s=size, c=h, cmap=cmap, vmin=0, vmax=1, edgecolors=BG, linewidths=1.2, zorder=3)
    ax.scatter(*pos[seed_bank], s=size[seed_bank] * 1.9, facecolors="none", edgecolors=PALE, lw=1.6, zorder=4)
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)

    fig.text(0.352, 0.60, "finality", color=AMBER, fontsize=64, fontweight="bold", va="baseline")
    fig.text(0.352, 0.42, "False confidence in the tail: extreme-value uncertainty,\n"
             "finality and contagion in banking-sector cyber risk",
             color="#e8eff7", fontsize=17.5, linespacing=1.55, va="baseline")
    fig.text(0.352, 0.24, "attacker economics · planted Pareto tail · profile-likelihood intervals · "
             "coverage audit · DebtRank", color=MUTED, fontsize=12.5, va="baseline")
    fig.text(0.985, 0.035, f"Interbank network of the paper (seed 7, {net.n} banks): a 60% loss at the largest "
             "core bank, linear DebtRank at exposure 1.05 (critical 0.87) · node area = assets, colour = distress",
             color="#6b7a99", fontsize=8.5, ha="right", va="baseline")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=BG, dpi=100)
    print(f"wrote {OUT.relative_to(ROOT)}; distressed banks (h > 0.5): {(h > 0.5).sum()} of {net.n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
