"""Automated layout check for matplotlib figures.

For a drawn figure it reports
  1. any two visible text objects whose glyph boxes overlap (titles, labels, tick labels,
     annotations and legend entries, across all panels);
  2. any in-axes text or legend box that touches plotted data: line segments (densified in
     display space and widened by half the line width), markers, scatter points, error bars
     and opaque bars. Light fills (alpha < 0.3), used for confidence bands and shaded regions,
     are treated as background.
Distances are measured at 200 dpi (one pixel is 0.36 pt), and a clearance of CLEAR pixels
(about 1.1 pt) is required between any text and any other text or stroke, so near-touches fail too.
"""
from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.legend import Legend
from matplotlib.text import Text

DPI = 200
CLEAR = 3.0   # required clearance in pixels at DPI (about 1.1 pt), beyond stroke half-widths


def _inter(a, b) -> float:
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w * h if (w > 0 and h > 0) else 0.0


def _split_nan(pts: np.ndarray) -> list[np.ndarray]:
    ok = np.all(np.isfinite(pts), axis=1)
    segs, cur = [], []
    for p, good in zip(pts, ok):
        if good:
            cur.append(p)
        elif cur:
            segs.append(np.array(cur))
            cur = []
    if cur:
        segs.append(np.array(cur))
    return segs


def _densify(pts: np.ndarray, step: float = 1.0) -> np.ndarray:
    if len(pts) < 2:
        return pts
    out = [pts[:1]]
    for p, q in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.ceil(np.hypot(*(q - p)) / step)))
        t = np.linspace(0, 1, n + 1)[1:, None]
        out.append(p + t * (q - p))
    return np.vstack(out)


def _geometry(ax, renderer, dpi):
    """Display-space sample points of the data in an axes, with a visual radius per point."""
    k = dpi / 72.0
    shapes, bars = [], []
    for ln in ax.get_lines():
        if not ln.get_visible():
            continue
        xy = np.asarray(ln.get_xydata(), float)
        if xy.size == 0:
            continue
        pts = ln.get_transform().transform(xy)
        if not str(ln.get_label()).startswith("_"):
            name = ln.get_label()
        else:
            from matplotlib.colors import to_hex
            name = f"unlabelled {ln.get_linestyle()} line {to_hex(ln.get_color())}"
        if ln.get_linestyle() not in ("None", "none", "", " ") and ln.get_linewidth() > 0:
            for seg in _split_nan(pts):
                d = _densify(seg)
                shapes.append((d, np.full(len(d), ln.get_linewidth() * k / 2), name))
        if ln.get_marker() not in (None, "None", "none", "", " "):
            good = pts[np.all(np.isfinite(pts), axis=1)]
            shapes.append((good, np.full(len(good), ln.get_markersize() * k / 2), name + " (markers)"))
    for c in ax.collections:
        if not c.get_visible():
            continue
        if isinstance(c, PathCollection):
            offs = c.get_offset_transform().transform(np.asarray(c.get_offsets(), float))
            sizes = np.asarray(c.get_sizes(), float)
            if len(offs) == 0:
                continue
            rad = np.sqrt(np.broadcast_to(sizes if sizes.size else [36.0], (len(offs),))) / 2 * k
            shapes.append((offs, rad, "scatter"))
        elif isinstance(c, LineCollection):
            lw = (c.get_linewidths()[0] if len(c.get_linewidths()) else 1.0) * k / 2
            for seg in c.get_segments():
                if len(seg) >= 2:
                    d = _densify(c.get_transform().transform(np.asarray(seg, float)))
                    shapes.append((d, np.full(len(d), lw), "error bar"))
    for p in ax.patches:
        if not p.get_visible() or not p.get_fill():
            continue
        alpha = p.get_facecolor()[3] if p.get_alpha() is None else p.get_alpha()
        if alpha >= 0.3:
            bars.append((p.get_window_extent(renderer), "bar"))
    axbb = ax.get_window_extent(renderer)
    clipped = []
    for pts, rad, name in shapes:
        if len(pts) == 0:
            continue
        m = (pts[:, 0] >= axbb.x0 - 1) & (pts[:, 0] <= axbb.x1 + 1) & (pts[:, 1] >= axbb.y0 - 1) & (pts[:, 1] <= axbb.y1 + 1)
        if m.any():
            clipped.append((pts[m], rad[m], name))
    return clipped, bars


def _tick_labels(fig):
    """All tick-label objects, and the subset matplotlib actually draws (tick inside the view limits)."""
    every, drawn = set(), set()
    for ax in fig.axes:
        for axis, lim, scale in ((ax.xaxis, ax.get_xlim(), ax.get_xscale()), (ax.yaxis, ax.get_ylim(), ax.get_yscale())):
            lo, hi = min(lim), max(lim)
            tol = 1e-9 * (hi - lo)
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                for lab in (tick.label1, tick.label2):
                    every.add(lab)
                    loc = tick.get_loc()
                    if loc is not None and lo - tol <= loc <= hi + tol and lab.get_visible() and tick.get_visible():
                        drawn.add(lab)
    return every, drawn


def lint(fig, name: str) -> list[str]:
    fig.set_dpi(DPI)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    issues = []
    every, drawn = _tick_labels(fig)
    texts = [t for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip() and (t not in every or t in drawn)]
    # Text.get_window_extent measures the glyphs only (for annotations it excludes the arrow)
    boxes = [(t, Text.get_window_extent(t, r)) for t in texts]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _inter(boxes[i][1].padded(CLEAR / 2), boxes[j][1].padded(CLEAR / 2)) > 0:
                issues.append(f"{name}: text '{boxes[i][0].get_text()}' overlaps text '{boxes[j][0].get_text()}'")
    legends = [lg for lg in fig.findobj(Legend) if lg.get_visible()]
    legend_texts = {t for lg in legends for t in lg.get_texts()}
    for ax in fig.axes:
        shapes, bars = _geometry(ax, r, fig.dpi)
        skip = {ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label,
                *ax.get_xticklabels(), *ax.get_yticklabels(), *legend_texts}
        targets = [(f"text '{t.get_text()}'", bb) for t, bb in boxes
                   if t.axes is ax and t not in skip and _inter(bb, ax.get_window_extent(r)) > 0]
        targets += [("legend", lg.get_window_extent(r)) for lg in legends if lg.axes is ax]
        for desc, bb in targets:
            for pts, rad, what in shapes:
                dx = np.maximum.reduce([bb.x0 - pts[:, 0], np.zeros(len(pts)), pts[:, 0] - bb.x1])
                dy = np.maximum.reduce([bb.y0 - pts[:, 1], np.zeros(len(pts)), pts[:, 1] - bb.y1])
                hit = np.hypot(dx, dy) < rad + CLEAR
                if hit.any():
                    issues.append(f"{name}: {desc} touches {what}")
            for pbb, what in bars:
                if _inter(bb.expanded(1, 1).padded(CLEAR), pbb) > 0:
                    issues.append(f"{name}: {desc} overlaps a {what}")
    return sorted(set(issues))
