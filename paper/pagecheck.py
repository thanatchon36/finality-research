#!/usr/bin/env python3
"""Page-fill audit: for every page, how far down the text block content reaches and the largest blank gap."""
import glob, os, subprocess, sys
import numpy as np
from PIL import Image
DPI, TOP, BOTTOM, H = 60, 0.95, 0.90, 11.0
pdf = sys.argv[1] if len(sys.argv) > 1 else "paper.pdf"
tmp = "/tmp/pagecheck"; os.makedirs(tmp, exist_ok=True)
for f in glob.glob(tmp + "/*.png"): os.remove(f)
subprocess.run(["pdftoppm", "-r", str(DPI), "-gray", "-png", pdf, tmp + "/p"], check=True)
y0, y1 = int(TOP * DPI), int((H - BOTTOM) * DPI)
files = sorted(glob.glob(tmp + "/p-*.png")); worst = []
for i, f in enumerate(files, 1):
    ink = (np.asarray(Image.open(f).convert("L")) < 200)[y0:y1].any(axis=1)
    rows = np.flatnonzero(ink)
    fill = (rows[-1] + 1) / (y1 - y0) if rows.size else 0.0
    blank, run, best = ~ink[rows[0]:rows[-1] + 1], 0, 0
    for b in blank:
        run = run + 1 if b else 0; best = max(best, run)
    gap = best / (y1 - y0)
    flag = "  <- check" if (fill < 0.94 and i < len(files)) or gap > 0.06 else ""
    worst.append((fill, gap))
    print(f"page {i:2d}: content reaches {fill:5.1%} of the text block; largest blank gap {gap:5.1%}{flag}")
print(f"pages {len(files)}; mean fill (excluding last) {np.mean([w[0] for w in worst[:-1]]):.1%}")
