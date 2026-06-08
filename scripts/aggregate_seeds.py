"""Aggregate multi-seed recover-check runs into mean +/- std error bars.

Reads results/<base>_seed{0..K}/recover_summary.csv, computes the mean and
standard deviation of the late-window improvement ratios
e_x(late, formulation) / e_x(late, none) across seeds, and prints a table plus
a bar chart with error bars. Establishes whether the A-vs-AOT margin is
statistically meaningful.

Usage:
    python scripts/aggregate_seeds.py \
        --base results/test_moment_obs_ABC_driven_BGK --seeds 0,1,2,3,4
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from plot_style import COLORS, LABELS, apply_style, smooth_log, figsize as figsz, tidy_log_yaxis  # noqa: E402,F401
apply_style()

import numpy as np

METRICS = ["e_rho", "e_u", "e_T", "e_f"]
FORMS = ["none", "aot", "A", "B", "C"]


def _read(csv_path: Path) -> dict:
    out = {}
    if not csv_path.exists():
        return out
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            out[row["formulation"]] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out", default="v1/figure/e_seed_errorbars.png")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    runs = [_read(Path(f"{args.base}_seed{s}") / "recover_summary.csv") for s in seeds]
    runs = [r for r in runs if r]
    n = len(runs)
    print(f"aggregating {n} seeds")

    # ratio_x[form][metric] = list over seeds of late(form)/late(none)
    stats = {f: {} for f in FORMS}
    for f in FORMS:
        for m in METRICS:
            vals = []
            for r in runs:
                try:
                    late = float(r[f][f"{m}_late"]); base = float(r["none"][f"{m}_late"])
                    if base > 0:
                        vals.append(late / base)
                except (KeyError, ValueError):
                    pass
            if vals:
                stats[f][m] = (float(np.mean(vals)), float(np.std(vals)))

    # Table
    print(f"\nImprovement ratio vs none (mean +/- std over {n} seeds):")
    hdr = "formulation | " + " | ".join(f"{m}" for m in METRICS)
    print(hdr); print("-" * len(hdr))
    for f in FORMS:
        cells = []
        for m in METRICS:
            if m in stats[f]:
                mu, sd = stats[f][m]
                cells.append(f"{mu:.3f}+/-{sd:.3f}")
            else:
                cells.append("--")
        print(f"{f:>11} | " + " | ".join(cells))

    # Bar chart with error bars for the moment channels e_u, e_T.
    MLAB = {"e_rho": r"$e_\rho$", "e_u": r"$e_u$", "e_T": r"$e_T$", "e_f": r"$e_f$"}
    fig, axes = plt.subplots(1, 2, figsize=figsz(1.0, 0.50))
    for ax, m in zip(axes, ["e_u", "e_T"]):
        fs = [f for f in ["aot", "A", "B", "C"] if m in stats[f]]
        mus = [stats[f][m][0] for f in fs]; sds = [stats[f][m][1] for f in fs]
        bars = ax.bar(fs, mus, yerr=sds, capsize=6,
                      color=[COLORS[f] for f in fs], alpha=0.85,
                      error_kw=dict(elinewidth=2, ecolor="k"))
        top = max(mu + sd for mu, sd in zip(mus, sds))
        ax.set_ylim(0, top * 1.35)   # zoom so the (significant) gaps are visible
        for f, mu, sd in zip(fs, mus, sds):
            ax.annotate(f"{mu:.3f}\n$\\pm${sd:.3f}", (f, mu + sd), ha="center",
                        va="bottom", fontsize=8)
        ax.set_ylabel(f"{MLAB.get(m, m)} improvement ratio (vs. none)")
        ax.set_title(MLAB.get(m, m))
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches=None); plt.close(fig)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
