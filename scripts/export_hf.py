"""Build the Hugging Face dataset, ``hf/``, from the study's seeds and its cached results.

Everything is regenerated from master seed 20261108 exactly as ``run_study.py`` draws it, and each table is
checked against ``results/results.json`` before it is written, so the dataset and the paper cannot drift apart.
The dataset card is ``scripts/hf_README.md``; this script copies it, adds the CC BY 4.0 licence and writes
``SHA256SUMS``. Upload with ``make hf-upload`` (needs ``hf auth login``).

Usage: ``python scripts/export_hf.py``.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finality import attacker, contagion, coverage, evt, lda  # noqa: E402

OUT = ROOT / "hf"
RES = json.loads((ROOT / "results" / "results.json").read_text())
SYSTEMIC = 20e9


def close(a: float, b: float, rel: float = 1e-9) -> None:
    if not math.isclose(a, b, rel_tol=rel, abs_tol=1e-12):
        raise SystemExit(f"reproduction check failed: {a!r} != {b!r}")


def write_csv(path: str, header: list[str], rows) -> int:
    p = OUT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(["" if (isinstance(x, float) and math.isnan(x)) else x for x in r])
            n += 1
    print(f"  {path}: {n:,} rows")
    return n


def write_json(path: str, obj) -> None:
    p = OUT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2) + "\n")
    print(f"  {path}")


def export_attacker() -> None:
    venues = attacker.VENUES
    write_csv("attacker/venues.csv",
              ["venue", "p_breach", "value_usd", "p_recovery", "haircut", "p_attribution", "sanction_usd",
               "cost_usd", "retained_value_usd", "payoff_usd", "break_even_recovery"],
              ([v.name, v.p_breach, v.value, v.p_recovery, v.haircut, v.p_attribution, v.sanction, v.cost,
                v.retained(), v.payoff(), v.break_even_recovery()] for v in venues))
    for v in venues:
        close(v.payoff(), RES["attacker"]["payoff_usd"][v.name])
    dec = attacker.log_gap_decomposition(venues[0], venues[2])
    for k, val in dec.items():
        close(val, RES["attacker"]["log_gap_bank_to_bridge"][k])
    write_csv("attacker/log_gap_bank_to_bridge.csv", ["factor", "log_gap", "share_of_total"],
              ([k, v, v / dec["total"]] for k, v in dec.items()))
    rows = []
    for v in venues:
        grid, pay = attacker.recovery_curve(v)
        rows += [[v.name, float(r), float(x)] for r, x in zip(grid, pay)]
    write_csv("attacker/recovery_curves.csv", ["venue", "p_recovery", "payoff_usd"], rows)


def export_loss(s_lda, s_truth, s_evt) -> dict:
    p = lda.LDAParams()
    write_json("loss/parameters.json", {**asdict(p), "xi_true": p.xi_true, "systemic_threshold_usd": SYSTEMIC,
                                        "currency": "USD", "note": "illustrative calibration"})
    sim = lda.simulate(200_000, p, np.random.default_rng(s_lda))
    close(float(sim.annual.mean()), RES["lda"]["mean"])
    write_csv("loss/annual_losses_200k.csv", ["year", "n_incidents", "annual_loss_usd"],
              zip(range(1, sim.annual.size + 1), sim.counts.tolist(), sim.annual.tolist()))
    write_json("loss/risk_metrics.json", RES["lda"])

    print("  simulating the 2,000,000-year truth ...")
    truth_annual = np.sort(lda.simulate_annual(RES["truth"]["years"], p, np.random.default_rng(s_truth)))
    truth = {T: lda.annual_return_level(truth_annual, T) for T in (100, 1000)}
    close(truth[100], RES["truth"]["annual_T100"])
    close(truth[1000], RES["truth"]["annual_T1000"])
    Ts = np.unique(np.round(np.logspace(np.log10(1.25), 5, 200), 6))
    write_csv("loss/truth_return_levels.csv", ["return_period_years", "annual_return_level_usd"],
              ([float(T), float(lda.annual_return_level(truth_annual, T))] for T in Ts))

    rng = np.random.default_rng(s_evt)
    long_rec = lda.simulate(1000, p, rng)
    short = lda.simulate(100, p, rng)
    m_long = evt.fit_pot(long_rec.incidents, 1000, 0.98, annual=long_rec.annual)
    m100 = evt.fit_pot(short.incidents, 100, 0.95, annual=short.annual)
    close(m_long.fit.xi, RES["evt"]["record_1000y"]["xi"])
    close(m100.fit.xi, RES["evt"]["record_100y"]["xi"])
    for name, rec in (("record_1000y", long_rec), ("record_100y", short)):
        years = np.repeat(np.arange(1, rec.annual.size + 1), rec.counts)
        write_csv(f"loss/{name}_incidents.csv", ["year", "loss_usd"], zip(years.tolist(), rec.incidents.tolist()))
    write_json("loss/single_record_fits.json", RES["evt"])
    return truth


def export_coverage(truth_b: dict) -> None:
    truth = {"A": {T: coverage.GPDDesign().truth(T) for T in (100, 1000)}, "B": truth_b}
    rows, check = [], {}
    for path in sorted((ROOT / "results" / "cache").glob("*.json")):
        exp, years = path.stem.split("_")[:2]
        years = int(years.rstrip("y"))
        for i, r in enumerate(json.loads(path.read_text()), 1):
            for T in (100, 1000):
                tv = truth[exp][T]
                for m in coverage.METHODS:
                    if r is None:
                        rows.append([exp, years, i, "", T, m, "", "", "", tv, "", "", "", "fit failed"])
                        continue
                    lo, hi = r[f"{m}|{T}"]
                    ok = math.isfinite(lo)
                    cov = ok and lo <= tv <= hi
                    check.setdefault((exp, years, T, m), []).append(cov) if ok else None
                    rows.append([exp, years, i, r["k"], T, m, r[f"point|{T}"], lo, hi, tv,
                                 int(cov) if ok else "", int(ok and hi < tv) if ok else "",
                                 int(ok and lo > tv) if ok else "", "" if ok else "no interval"])
    for c in RES["coverage"]:
        got = check[(c["experiment"], c["record_years"], c["T"], c["method"])]
        close(float(np.mean(got)), c["coverage"])
    write_csv("coverage/intervals.csv",
              ["experiment", "record_years", "record", "exceedances", "return_period", "method", "point_usd",
               "lower_usd", "upper_usd", "truth_usd", "covered", "truth_above", "truth_below", "note"], rows)
    keys = ["experiment", "record_years", "mean_k", "T", "method", "n", "coverage", "mc_se", "miss_low",
            "miss_high", "median_width_ratio", "share_unbounded", "median_point_over_truth", "failed_records"]
    write_csv("coverage/summary.csv", keys, ([c[k] for k in keys] for c in RES["coverage"]))
    write_json("coverage/designs.json", {
        "A": {"description": "exceedances drawn exactly from a GPD", **asdict(coverage.GPDDesign()),
              "record_years": [30, 100, 300], "truth_usd": truth["A"]},
        "B": {"description": "full loss pipeline (loss/parameters.json), threshold at a quantile of incidents",
              "u_quantile": coverage.LDADesign().u_quantile, "record_years": [30, 100, 1000], "truth_usd": truth["B"]},
        "reps_per_cell": RES["settings"]["reps"], "bootstrap_B": RES["settings"]["boot"], "nominal_level": 0.95,
        "seed": RES["settings"]["seed"]})


def export_contagion() -> None:
    net = contagion.build_network(7)
    close(net.critical_exposure, RES["contagion"]["critical_exposure"])
    big, vend = net.largest(), net.vendor_clients(0.3)
    rank = contagion.systemic_ranking(net, 0.5, 0.2)
    write_csv("contagion/banks.csv",
              ["bank", "tier", "total_assets_usd", "equity_usd", "interbank_intensity", "vendor_client",
               "systemic_multiplier"],
              ([net.names[i], "core" if net.core[i] else "periphery", float(net.ta[i]), float(net.eq[i]),
                float(net.k[i]), int(i in set(vend.tolist())), float(rank[i])] for i in range(net.n)))
    write_csv("contagion/exposures.csv", ["creditor", "debtor", "weight"],
              ([net.names[i], net.names[j], float(net.W[i, j])] for i, j in zip(*np.nonzero(net.W))))
    levs = np.linspace(0.1, 2.0, 77)
    big_s = contagion.phase_sweep(net, [big], 0.2, levs)
    vend_s = contagion.phase_sweep(net, vend, 0.2, levs)
    write_csv("contagion/phase_sweep.csv", ["exposure_scale", "multiplier_largest_bank", "multiplier_vendor"],
              zip(levs.tolist(), big_s.tolist(), vend_s.tolist()))
    cas = {f"{net.names[big]} hit, exposure 0.5x": contagion.debtrank(net, 0.5, [big], 0.2),
           "vendor hit, exposure 0.5x": contagion.debtrank(net, 0.5, vend, 0.2),
           "vendor hit, exposure 1.2x": contagion.debtrank(net, 1.2, vend, 0.2)}
    for k, h in cas.items():
        close(contagion.equity_loss(net, h[-1]), RES["contagion"]["scenarios"][k]["total"])
    write_csv("contagion/cascades.csv", ["scenario", "round", "bank", "distress"],
              ([k, r, net.names[i], float(h[r, i])] for k, h in cas.items() for r in range(h.shape[0])
               for i in range(net.n)))
    write_json("contagion/summary.json", RES["contagion"])


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    s_lda, s_truth, s_evt, _, _ = np.random.SeedSequence(RES["settings"]["seed"]).spawn(5)
    print("attacker"); export_attacker()
    print("loss"); truth = export_loss(s_lda, s_truth, s_evt)
    print("coverage"); export_coverage(truth)
    print("contagion"); export_contagion()
    print("raw")
    shutil.copytree(ROOT / "results" / "cache", OUT / "raw" / "cache")
    for f in ("results.json", "results.md"):
        shutil.copyfile(ROOT / "results" / f, OUT / "raw" / f)
    shutil.copyfile(ROOT / "scripts" / "hf_README.md", OUT / "README.md")
    (OUT / "LICENSE").write_text(
        "Creative Commons Attribution 4.0 International (CC BY 4.0)\n"
        "https://creativecommons.org/licenses/by/4.0/legalcode\n\n"
        "Copyright (c) 2026 Thanatchon Chongmankhong. All data in this dataset are simulated from illustrative\n"
        "parameters by the finality package (https://github.com/thanatchon36/finality-research, MIT licence).\n")
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (OUT / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(OUT)}\n" for p in files))
    size = sum(p.stat().st_size for p in files)
    print(f"wrote {OUT.relative_to(ROOT)}/ ({len(files)} files, {size / 1e6:.1f} MB); all reproduction checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
