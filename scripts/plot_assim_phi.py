"""Compare phi(t,x) between truth and each assim mode.

For each sampled time, reads particles.h5, deposits rho, solves Poisson
to get phi. Truth phi is read directly from truth.npz.

Output:
  results/<truth-dir>/assim_phi_compare.png         absolute phi heatmaps
  results/<truth-dir>/assim_phi_pert_compare.png    same but mean-subtracted

Usage:
  python scripts/plot_assim_phi.py --truth-dir cases/position_mismatch/results/truth
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from mfda.backend_reference import cic_deposit
from mfda.poisson import potential_from_density

MODES = ("none", "phi", "phi_dphi", "phi_dphi_d2phi")
LABELS = {"none": "no nudge", "phi": "phi snapshot",
          "phi_dphi": "phi + d_t phi",
          "phi_dphi_d2phi": "phi + d_t phi + d_t^2 phi"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth-dir", required=True)
    ap.add_argument("--n-times", type=int, default=80,
                    help="evenly-spaced time samples (across whole run)")
    ap.add_argument("--Nx", type=int, default=128)
    args = ap.parse_args()

    truth_dir = Path(args.truth_dir)
    parent = truth_dir.parent
    name = truth_dir.name
    truth_h5 = truth_dir / "particles.h5"
    truth_npz = np.load(truth_dir / "truth.npz")
    phi_truth_full = np.asarray(truth_npz["phi"])

    with h5py.File(truth_h5, "r") as f:
        Nt = f["x"].shape[0]
        L = float(f.attrs.get("L", 4.0 * np.pi))
        dt = float(f.attrs.get("dt", 0.01))

    Nx = args.Nx
    samples = sorted({int(round(i * (Nt - 1) / (args.n_times - 1)))
                      for i in range(args.n_times)})
    times = np.array([s * dt for s in samples])

    mode_paths: dict[str, Path] = {}
    for mode in MODES:
        p = parent / mode / "particles.h5"
        if p.exists():
            mode_paths[mode] = p
    if not mode_paths:
        raise SystemExit("no assim particles found")

    n_t = len(samples)
    n_rows = 1 + len(mode_paths)
    phi_panel = np.zeros((n_rows, n_t, Nx))

    # Row 0: truth phi (read directly).
    for i, n in enumerate(samples):
        phi_panel[0, i] = phi_truth_full[n]

    # Rows 1..N: each mode — deposit rho, solve Poisson.
    for j_mode, (mode, p) in enumerate(mode_paths.items(), start=1):
        for i, n in enumerate(samples):
            with h5py.File(p, "r") as f:
                x = f["x"][n]
                wp = f["w"][:]
            rho_a = cic_deposit(x, wp, L, Nx)
            phi_panel[j_mode, i] = potential_from_density(rho_a, L)
        print(f"[phi] sampled mode={mode}")

    extent = (times[0], times[-1], 0.0, L)
    row_labels = ["truth"] + [LABELS[m] for m in mode_paths]
    vmax = float(np.percentile(np.abs(phi_panel), 99.5))

    # Absolute phi heatmaps.
    fig, axes = plt.subplots(n_rows, 1, figsize=(11, 1.8 * n_rows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for row in range(n_rows):
        im = axes[row].imshow(
            phi_panel[row].T, origin="lower", aspect="auto",
            extent=extent, cmap="RdBu_r", vmin=-vmax, vmax=+vmax,
        )
        axes[row].set_ylabel(f"{row_labels[row]}\nx", fontsize=9)
    axes[-1].set_xlabel("t")
    fig.suptitle(f"{name}: phi(x, t) — truth + each assim mode")
    fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    out = truth_dir / "assim_phi_compare.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[phi] wrote {out}")

    # Difference vs truth.
    diff_panel = np.zeros_like(phi_panel)
    for j_mode in range(1, n_rows):
        diff_panel[j_mode] = phi_panel[j_mode] - phi_panel[0]
    vmax_d = float(np.percentile(np.abs(diff_panel[1:]), 99.5))

    fig2, axes2 = plt.subplots(n_rows - 1, 1, figsize=(11, 1.8 * (n_rows - 1)),
                               sharex=True, sharey=True)
    axes2 = np.atleast_1d(axes2)
    for row in range(1, n_rows):
        im = axes2[row - 1].imshow(
            diff_panel[row].T, origin="lower", aspect="auto",
            extent=extent, cmap="RdBu_r", vmin=-vmax_d, vmax=+vmax_d,
        )
        axes2[row - 1].set_ylabel(f"{row_labels[row]}\nx", fontsize=9)
    axes2[-1].set_xlabel("t")
    fig2.suptitle(f"{name}: phi_assim - phi_truth (mode rows)")
    fig2.colorbar(im, ax=axes2, shrink=0.8, pad=0.02)
    out2 = truth_dir / "assim_phi_diff_compare.png"
    fig2.savefig(out2, dpi=140, bbox_inches="tight")
    plt.close(fig2)
    print(f"[phi] wrote {out2}")

    # ---- 1D line plots: phi(x) overlay (truth + each mode) at fixed times ----
    n_snap = 6
    snap_idx = np.linspace(0, n_t - 1, n_snap, dtype=int)
    x_grid = np.linspace(0.0, L, Nx, endpoint=False)
    colors = {"truth": "k", "no nudge": "C3", "phi snapshot": "C0",
              "phi + d_t phi": "C2", "phi + d_t phi + d_t^2 phi": "C1"}
    styles = {"truth": "-", "no nudge": "--", "phi snapshot": "-",
              "phi + d_t phi": "-", "phi + d_t phi + d_t^2 phi": ":"}

    fig3, axes3 = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=True)
    axes3 = axes3.ravel()
    for ax_idx, ti in enumerate(snap_idx):
        ax = axes3[ax_idx]
        for row, lab in enumerate(row_labels):
            ax.plot(x_grid, phi_panel[row, ti], color=colors[lab],
                    ls=styles[lab], lw=1.4, alpha=0.85, label=lab)
        ax.set_title(f"t = {times[ti]:.2f}", fontsize=10)
        ax.grid(alpha=0.3)
        if ax_idx % 3 == 0:
            ax.set_ylabel(r"$\phi(x)$")
        if ax_idx >= 3:
            ax.set_xlabel("x")
    axes3[0].legend(fontsize=8, loc="upper right")
    fig3.suptitle(f"{name}: phi(x) at sampled times — truth + each assim mode")
    fig3.tight_layout()
    out3 = truth_dir / "assim_phi_lines.png"
    fig3.savefig(out3, dpi=140, bbox_inches="tight")
    plt.close(fig3)
    print(f"[phi] wrote {out3}")

    # ---- 1D diff lines: (phi_assim - phi_truth)(x) ----
    fig4, axes4 = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=True)
    axes4 = axes4.ravel()
    for ax_idx, ti in enumerate(snap_idx):
        ax = axes4[ax_idx]
        for row in range(1, n_rows):
            lab = row_labels[row]
            ax.plot(x_grid, diff_panel[row, ti], color=colors[lab],
                    ls=styles[lab], lw=1.4, alpha=0.85, label=lab)
        ax.axhline(0, color="k", lw=0.6, alpha=0.5)
        ax.set_title(f"t = {times[ti]:.2f}", fontsize=10)
        ax.grid(alpha=0.3)
        if ax_idx % 3 == 0:
            ax.set_ylabel(r"$\phi_{\rm assim} - \phi_{\rm truth}$")
        if ax_idx >= 3:
            ax.set_xlabel("x")
    axes4[0].legend(fontsize=8, loc="upper right")
    fig4.suptitle(f"{name}: phi_assim - phi_truth at sampled times")
    fig4.tight_layout()
    out4 = truth_dir / "assim_phi_diff_lines.png"
    fig4.savefig(out4, dpi=140, bbox_inches="tight")
    plt.close(fig4)
    print(f"[phi] wrote {out4}")


if __name__ == "__main__":
    main()
