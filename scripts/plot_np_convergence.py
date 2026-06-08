"""Np-convergence of the moment-recovery floor (validates the mean-field limit).

Reads the late-window moment errors of Formulation A from a set of
driven-BGK recover-check runs at increasing particle number and plots them
against N_p on log-log axes; the shot-noise floor of the empirical moments
scales like N_p^{-1/2}, so the fitted slope near -1/2 is a numerical witness
of propagation of chaos (Prop. mean-field limit).

Usage:
    python scripts/plot_np_convergence.py \
        --base results/test_moment_obs_ABC_driven_BGK --nps 10000,30000,100000,300000,1000000
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


def _late(csv_path: Path, formulation: str, metric: str) -> float:
    if not csv_path.exists():
        return float("nan")
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["formulation"] == formulation:
                return float(row[f"{metric}_late"])
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="results/<name> base (suffix _np<NP> per run)")
    ap.add_argument("--nps", default="10000,30000,100000,300000,1000000")
    ap.add_argument("--out", default="v1/figure/e_np_convergence.png")
    args = ap.parse_args()
    nps = [int(x) for x in args.nps.split(",")]
    eu = [_late(Path(f"{args.base}_np{n}") / "recover_summary.csv", "A", "e_u") for n in nps]
    eT = [_late(Path(f"{args.base}_np{n}") / "recover_summary.csv", "A", "e_T") for n in nps]
    nps_a = np.array(nps, float)

    fig, ax = plt.subplots(figsize=figsz(0.62, 0.80))
    for series, lab, c in ((eu, r"$e_u$", "tab:blue"), (eT, r"$e_T$", "tab:green")):
        s = np.array(series, float)
        ok = np.isfinite(s) & (s > 0)
        ax.loglog(nps_a[ok], s[ok], "o-", color=c, label=lab)
        if ok.sum() >= 2:
            slope = np.polyfit(np.log(nps_a[ok]), np.log(s[ok]), 1)[0]
            ax.loglog(nps_a[ok], np.exp(np.polyval([slope, np.log(s[ok][0]) - slope*np.log(nps_a[ok][0])], np.log(nps_a[ok]))),
                      "--", color=c, alpha=0.5, label=f"{lab} slope={slope:.2f}")
    ref = eu[2] if np.isfinite(eu[2]) else 0.1
    ax.loglog(nps_a, ref * np.sqrt(nps_a[2] / nps_a), "k:", alpha=0.6, label=r"$N_p^{-1/2}$ ref")
    ax.set_xlabel(r"$N_p$"); ax.set_ylabel("late-window moment error")
    ax.set_title("Moment-recovery floor")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out); plt.close(fig)
    print(f"wrote {args.out}")
    print("Np, e_u_late(A), e_T_late(A):")
    for n, a, b in zip(nps, eu, eT):
        print(f"  {n:>8} {a:.4g} {b:.4g}")


if __name__ == "__main__":
    main()
