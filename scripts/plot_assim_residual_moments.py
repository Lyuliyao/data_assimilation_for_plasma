"""Spatial residual heatmaps showing where each assim mode differs from
truth. At ε=10⁻³ the absolute phase-space looks identical for all modes,
but the rho/j/M residuals show the position mismatch at the right scale.

Reads:
    results/<truth>/truth.npz                          — has rho, j, M
    results/<case>/<mode>/assim_diagnostics.npz  — has e_rho per step

Computes assim rho, j, M at a few snapshot times by reading particles.h5
(truth and each mode), then plots:

    rows: 4 modes (none, phi, phi_dphi, phi_dphi_d2phi)
    cols: rho_assim - rho_truth, j_assim - j_truth, M_assim - M_truth
    each cell: heatmap (t, x), colour scale matched per column

Usage:
    python scripts/plot_assim_residual_moments.py --truth-dir cases/position_mismatch/results/truth
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from mfda.backend_reference import cic_deposit, cic_deposit_current
from mfda.kinetic_stress import cic_deposit_kinetic_stress

MODES = ("none", "phi", "phi_dphi", "phi_dphi_d2phi")
LABELS = {"none": "no nudge", "phi": "phi snapshot",
          "phi_dphi": "phi + d_t phi",
          "phi_dphi_d2phi": "phi + d_t phi + d_t^2 phi"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth-dir", required=True)
    ap.add_argument("--n-times", type=int, default=20,
                    help="evenly-spaced time samples (across whole run)")
    ap.add_argument("--Nx", type=int, default=128)
    args = ap.parse_args()

    truth_dir = Path(args.truth_dir)
    parent = truth_dir.parent
    name = truth_dir.name
    truth_h5 = truth_dir / "particles.h5"
    truth = np.load(truth_dir / "truth.npz")
    rho_truth_full = truth["rho"]
    j_truth_full = truth["j"]
    M_truth_full = truth["M"]

    with h5py.File(truth_h5, "r") as f:
        Nt = f["x"].shape[0]
        L = float(f.attrs.get("L", 4.0 * np.pi))
        dt = float(f.attrs.get("dt", 0.01))
    Nx = args.Nx
    samples = sorted({int(round(i * (Nt - 1) / (args.n_times - 1)))
                      for i in range(args.n_times)})
    times = np.array([s * dt for s in samples])

    # collect modes that have particles.h5
    mode_paths: dict[str, Path] = {}
    for mode in MODES:
        p = parent / mode / "particles.h5"
        if p.exists():
            mode_paths[mode] = p

    if not mode_paths:
        raise SystemExit("no assim particles found")

    # Pre-allocate per-mode residual arrays.
    n_t = len(samples)
    drho = {m: np.zeros((n_t, Nx)) for m in mode_paths}
    dj = {m: np.zeros((n_t, Nx)) for m in mode_paths}
    dM = {m: np.zeros((n_t, Nx)) for m in mode_paths}

    for i, n in enumerate(samples):
        rho_t = rho_truth_full[n]
        j_t = j_truth_full[n]
        M_t = M_truth_full[n]
        for mode, p in mode_paths.items():
            with h5py.File(p, "r") as f:
                x = f["x"][n]
                v = f["v"][n]
                wp = f["w"][:]
            rho_a = cic_deposit(x, wp, L, Nx)
            j_a = cic_deposit_current(x, v, wp, L, Nx)
            M_a = cic_deposit_kinetic_stress(x, v, wp, L, Nx)
            drho[mode][i] = rho_a - rho_t
            dj[mode][i] = j_a - j_t
            dM[mode][i] = M_a - M_t
        print(f"[res] sampled t={times[i]:.1f} ({i+1}/{n_t})")

    # Plot: 4 rows (modes) x 3 cols (drho, dj, dM); per-column shared color scale.
    fig, axes = plt.subplots(len(mode_paths), 3,
                             figsize=(13, 2.5 * len(mode_paths)),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    extent = (times[0], times[-1], 0.0, L)
    titles = (
        r"$\rho_{\rm assim} - \rho_{\rm truth}$",
        r"$j_{\rm assim} - j_{\rm truth}$",
        r"$M_{\rm assim} - M_{\rm truth}$",
    )

    # column-wise vmax (over all modes for fair visual comparison)
    vmax_drho = float(np.percentile(np.abs(np.array(list(drho.values()))), 99.5))
    vmax_dj = float(np.percentile(np.abs(np.array(list(dj.values()))), 99.5))
    vmax_dM = float(np.percentile(np.abs(np.array(list(dM.values()))), 99.5))
    vmaxes = (vmax_drho, vmax_dj, vmax_dM)

    for row, mode in enumerate(mode_paths):
        for col, (title, arr_dict, vmax) in enumerate([
            (titles[0], drho, vmax_drho),
            (titles[1], dj, vmax_dj),
            (titles[2], dM, vmax_dM),
        ]):
            ax = axes[row, col]
            im = ax.imshow(arr_dict[mode].T, origin="lower", aspect="auto",
                           extent=extent, cmap="RdBu_r",
                           vmin=-vmax, vmax=+vmax)
            if row == 0:
                ax.set_title(title, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{LABELS[mode]}\nx", fontsize=9)
            if row == len(mode_paths) - 1:
                ax.set_xlabel("t")
        # one colorbar at the end of each row
        fig.colorbar(im, ax=axes[row, :], shrink=0.8, pad=0.02)
    fig.suptitle(f"{name}: assim − truth residuals (lower amplitude = better)")
    fig.tight_layout(rect=(0, 0, 0.94, 1))
    out = truth_dir / "assim_residual_moments.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[res] wrote {out}")


if __name__ == "__main__":
    main()
