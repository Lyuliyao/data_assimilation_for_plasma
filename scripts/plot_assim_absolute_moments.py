"""Absolute (not residual) heatmaps of rho, j, M for truth + each assim mode.

5 rows (truth + 4 modes) × 3 cols (rho, j, M). Color scales matched per
column so the comparison is direct. Reads particles.h5 lazily.

Usage:
    python scripts/plot_assim_absolute_moments.py --truth-dir cases/position_mismatch/results/truth
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
    ap.add_argument("--show-pert", action="store_true",
                    help="subtract baselines (rho-1, M-sigma^2*L) so the "
                         "tiny perturbations are visible at this scale")
    args = ap.parse_args()

    truth_dir = Path(args.truth_dir)
    parent = truth_dir.parent
    name = truth_dir.name
    truth_h5 = truth_dir / "particles.h5"
    truth_npz = np.load(truth_dir / "truth.npz")
    rho_truth_full = np.asarray(truth_npz["rho"])
    j_truth_full = np.asarray(truth_npz["j"])
    M_truth_full = np.asarray(truth_npz["M"])

    with h5py.File(truth_h5, "r") as f:
        Nt = f["x"].shape[0]
        L = float(f.attrs.get("L", 4.0 * np.pi))
        dt = float(f.attrs.get("dt", 0.01))

    Nx = args.Nx
    samples = sorted({int(round(i * (Nt - 1) / (args.n_times - 1)))
                      for i in range(args.n_times)})
    times = np.array([s * dt for s in samples])

    # Collect modes that have particles.h5
    mode_paths: dict[str, Path] = {}
    for mode in MODES:
        p = parent / mode / "particles.h5"
        if p.exists():
            mode_paths[mode] = p
    if not mode_paths:
        raise SystemExit("no assim particles found")

    n_t = len(samples)
    n_mode = len(mode_paths) + 1  # +1 for truth row
    rho_panel = np.zeros((n_mode, n_t, Nx))
    j_panel = np.zeros((n_mode, n_t, Nx))
    M_panel = np.zeros((n_mode, n_t, Nx))

    # Row 0 = truth (read from truth.npz which already has the moments).
    for i, n in enumerate(samples):
        rho_panel[0, i] = rho_truth_full[n]
        j_panel[0, i] = j_truth_full[n]
        M_panel[0, i] = M_truth_full[n]

    # Rows 1..N: assim modes (re-deposit from particles.h5).
    for j_mode, (mode, p) in enumerate(mode_paths.items(), start=1):
        for i, n in enumerate(samples):
            with h5py.File(p, "r") as f:
                x = f["x"][n]
                v = f["v"][n]
                wp = f["w"][:]
            rho_panel[j_mode, i] = cic_deposit(x, wp, L, Nx)
            j_panel[j_mode, i] = cic_deposit_current(x, v, wp, L, Nx)
            M_panel[j_mode, i] = cic_deposit_kinetic_stress(x, v, wp, L, Nx)
        print(f"[abs] sampled mode={mode}")

    # Optionally subtract baselines so the tiny perturbations are visible.
    if args.show_pert:
        # rho baseline = 1 (neutral background), M baseline = mean of truth M
        rho_panel = rho_panel - 1.0
        # j baseline is 0 already
        M_baseline = float(np.mean(M_panel[0]))   # truth's M average
        M_panel = M_panel - M_baseline
        suffix = "  (baseline-subtracted)"
        cmap = "RdBu_r"
    else:
        suffix = ""
        cmap = "viridis"

    # Plot.
    fig, axes = plt.subplots(n_mode, 3, figsize=(13, 2.3 * n_mode),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    extent = (times[0], times[-1], 0.0, L)
    titles = (
        rf"$\rho(x, t)${suffix}",
        rf"$j(x, t)${suffix}",
        rf"$M(x, t)${suffix}",
    )

    # Per-column color scale derived from truth + all modes.
    vmax_rho = float(np.percentile(np.abs(rho_panel), 99.5))
    vmax_j = float(np.percentile(np.abs(j_panel), 99.5))
    vmax_M = float(np.percentile(np.abs(M_panel), 99.5))
    vmaxes = (vmax_rho, vmax_j, vmax_M)

    row_labels = ["truth"] + [LABELS[m] for m in mode_paths]
    for row in range(n_mode):
        for col, (panel, vmax, title) in enumerate(zip(
                (rho_panel, j_panel, M_panel),
                vmaxes,
                titles)):
            ax = axes[row, col]
            kwargs = dict(origin="lower", aspect="auto", extent=extent)
            if args.show_pert:
                kwargs.update(cmap=cmap, vmin=-vmax, vmax=+vmax)
            else:
                kwargs.update(cmap=cmap, vmin=0 if col == 0 else -vmax,
                              vmax=vmax)
            im = ax.imshow(panel[row].T, **kwargs)
            if row == 0:
                ax.set_title(title, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{row_labels[row]}\nx", fontsize=9)
            if row == n_mode - 1:
                ax.set_xlabel("t")
        fig.colorbar(im, ax=axes[row, :], shrink=0.8, pad=0.02)

    fig.suptitle(f"{name}: absolute moments rho, j, M  (truth + each assim mode){suffix}")
    fig.tight_layout(rect=(0, 0, 0.94, 1))
    out_name = "assim_absolute_moments_pert.png" if args.show_pert else "assim_absolute_moments.png"
    out = truth_dir / out_name
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[abs] wrote {out}")


if __name__ == "__main__":
    main()
