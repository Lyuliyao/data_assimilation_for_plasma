"""Regenerate the recover-check paper figures from saved .npz (no sim re-run).

Rebuilds the moment error-curve panels and velocity marginals for the driven,
resonant, and obstruction experiments using the shared paper style
(scripts/plot_style.py): uniform per-method colors, serif/CM fonts, smoothing.
Writes straight into v1/figure/ with the names the manuscript references.
"""
from __future__ import annotations
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from plot_style import moment_error_curves, velocity_marginal

ROOT = pathlib.Path("/ocean/projects/mth210003p/lyuliyao/DA/plasma")
FIG = ROOT / "v1" / "figure"


def load(d: pathlib.Path, forms):
    runs, finals = {}, {}
    for f in forms:
        z = np.load(d / f"{f}.npz")
        runs[f] = {"t": z["t"]}
        for m in ("e_rho", "e_u", "e_T", "e_f", "e_phi"):
            k = f"assim_log_{m}"
            if k in z.files and np.asarray(z[k]).size:
                runs[f][m] = z[k]
        finals[f] = {"v": z["final_assim_v"], "w": z["final_assim_w"]}
    z0 = np.load(d / f"{forms[0]}.npz")
    finals["truth"] = {"v": z0["final_truth_v"], "w": z0["final_truth_w"]}
    return runs, finals


JOBS = [
    # (subdir, forms, error_fig, vkde_fig, vmin, vmax)
    ("test_moment_obs_ABC_driven_BGK", ["none", "A", "B", "C"],
     "e2_driven_err.png", "e2_driven_vkde.png", -8, 8),
    ("test_moment_obs_ABC_driven_BGK_truly_resonant", ["none", "A", "B", "C"],
     "e3_resonant_err.png", None, -12, 12),
    ("exp_obstruction_two_stream", ["none", "A", "B", "C", "naive_kl"],
     None, "e_obstruction_vkde.png", -8, 8),
]

for sub, forms, ef, vf, vmin, vmax in JOBS:
    d = ROOT / "results" / sub
    forms = [f for f in forms if (d / f"{f}.npz").exists()]
    if not forms:
        print(f"  SKIP {sub}: no npz"); continue
    runs, finals = load(d, forms)
    if ef:
        moment_error_curves(FIG / ef, runs, forms)
        print(f"  wrote {ef}  ({sub}, forms={forms})")
    if vf:
        velocity_marginal(FIG / vf, finals, forms, vmin, vmax)
        print(f"  wrote {vf}  ({sub})")
print("done.")
