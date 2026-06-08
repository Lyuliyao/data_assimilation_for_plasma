"""Restyle the E1 rate-validation figure from saved curves (no sim re-run).

Reads results/exp1_homogeneous_rate/rate_curves.npz and rate_summary.csv and
re-renders the e_u/e_T decay panels for none/A/B/C (AOT removed), with the same
paper style as run_rate_validation.py. Writes v1/figure/e1_rate.png.
"""
from __future__ import annotations
import csv
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from plot_style import COLORS, LABELS, apply_style, smooth_log, figsize as figsz, tidy_log_yaxis  # noqa: E402

apply_style()

ROOT = pathlib.Path("/ocean/projects/mth210003p/lyuliyao/DA/plasma")
RESDIR = ROOT / "results" / "exp1_homogeneous_rate"
OUT = ROOT / "v1" / "figure" / "e1_rate.png"
FORMS = ["none", "A", "B", "C"]

z = np.load(RESDIR / "rate_curves.npz")
rates = {}
with open(RESDIR / "rate_summary.csv") as fp:
    for row in csv.DictReader(fp):
        rates[row["formulation"]] = (row["rate_u"], row["rate_T"])

panels = {"e_u": r"$e_u$  (bulk velocity)", "e_T": r"$e_T$  (temperature)"}
fig, axes = plt.subplots(1, 2, figsize=figsz(1.0, 0.40))
for name, ax in (("e_u", axes[0]), ("e_T", axes[1])):
    rate_key = "rate_u" if name == "e_u" else "rate_T"
    idx = 0 if name == "e_u" else 1
    for f in FORMS:
        t = z[f"{f}_t"]
        e = np.asarray(z[f"{f}_{name}"], dtype=float)
        if e.size == 0:
            continue
        r = rates.get(f, ("nan", "nan"))[idx]
        try:
            rv = float(r)
            lab = LABELS.get(f, f) + ("" if not np.isfinite(rv) else f" (rate ${rv:.2g}$)")
        except ValueError:
            lab = LABELS.get(f, f)
        e = np.maximum(e, 1e-16)
        ax.semilogy(t, e, color=COLORS[f], alpha=0.15, lw=0.7)
        ax.semilogy(t, smooth_log(e), color=COLORS[f], lw=1.8, label=lab)
    ax.set_title(panels[name])
    ax.set_xlabel("$t$")
    ax.grid(True, which="both", alpha=0.3)
    tidy_log_yaxis(ax)
    ax.legend()
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
plt.close(fig)
print(f"wrote {OUT}")
