"""Nudging-strength sensitivity (validates C's parameter-free 1:2 rate).

Reads the fitted decay rates from homogeneous rate-validation runs at several
nudging strengths and plots, per formulation, the velocity/temperature rates
and their ratio against the strength. Formulation C should show rates that
scale linearly with the strength while the ratio stays pinned at 2 (the
Fisher-Rao value, independent of the strength); A/B/AOT scale similarly but at
a tunable, non-canonical ratio.

Usage:
    python scripts/plot_gamma_sensitivity.py \
        --base results/exp1_homogeneous_rate --strengths 0.5,1,2,4
The strength=1 run is the baseline `results/exp1_homogeneous_rate`; others are
`..._g<strength>` (e.g. _g0p5, _g2, _g4).
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



def _rates(csv_path: Path, formulation: str):
    if not csv_path.exists():
        return (float("nan"),) * 3
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["formulation"] == formulation:
                return (float(row["rate_u"]), float(row["rate_T"]),
                        float(row["ratio_T_over_u"]))
    return (float("nan"),) * 3


def _dir(base: str, g: float) -> Path:
    if abs(g - 1.0) < 1e-9:
        return Path(base)
    tag = ("g" + str(g)).replace(".", "p")
    return Path(f"{base}_{tag}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--strengths", default="0.5,1,2,4")
    ap.add_argument("--out", default="v1/figure/e_gamma_sensitivity.png")
    args = ap.parse_args()
    gs = [float(x) for x in args.strengths.split(",")]

    fig, (axr, axq) = plt.subplots(1, 2, figsize=figsz(1.0, 0.42))
    for fm in ("aot", "A", "C"):
        ru = []; rt = []; rq = []
        for g in gs:
            u, t, q = _rates(_dir(args.base, g) / "rate_summary.csv", fm)
            ru.append(u); rt.append(t); rq.append(q)
        axr.plot(gs, ru, "o-", color=COLORS[fm], label=f"{fm} $e_u$")
        axr.plot(gs, rt, "s--", color=COLORS[fm], alpha=0.6, label=f"{fm} $e_T$")
        axq.plot(gs, rq, "o-", color=COLORS[fm], label=fm)
    axr.set_xlabel("nudging strength"); axr.set_ylabel("fitted decay rate")
    axr.set_title("Rates vs. strength"); axr.grid(True, alpha=0.3); axr.legend(fontsize=8)
    axq.axhline(2.0, color="tab:green", ls=":", alpha=0.7, label="C target $1{:}2$")
    axq.axhline(1.0, color="gray", ls=":", alpha=0.5, label="ratio $1{:}1$")
    axq.set_xlabel("nudging strength"); axq.set_ylabel(r"ratio rate$_T$/rate$_u$")
    axq.set_title("Ratio is strength-invariant")
    axq.grid(True, alpha=0.3); axq.legend(fontsize=8); axq.set_ylim(0, 2.6)
    fig.tight_layout(); Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches=None); plt.close(fig)
    print(f"wrote {args.out}")
    for fm in ("aot", "A", "C"):
        print(f"  {fm}: " + ", ".join(
            f"g={g}:ratio={_rates(_dir(args.base, g)/'rate_summary.csv', fm)[2]:.3g}" for g in gs))


if __name__ == "__main__":
    main()
