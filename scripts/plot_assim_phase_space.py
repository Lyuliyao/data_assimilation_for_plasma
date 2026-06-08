"""Phase-space + velocity-marginal figure comparing truth vs assim modes.

Reads:
    results/<truth-name>/particles.h5                     — truth particles
    results/<case>/<mode>/particles.h5        — assim particles per mode

Picks ~6 time indices, slices each H5 lazily (no full load of the 64 GB
files), and writes:
    results/<truth-name>/assim_phase_space_compare.png  — 6 rows x 4 cols heatmap
    results/<truth-name>/assim_vkde_compare.png         — 6-panel kde overlays

Usage:
    python scripts/plot_assim_phase_space.py --truth-dir cases/position_mismatch/results/truth
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

MODES = ("none", "phi", "phi_dphi", "phi_dphi_d2phi")
LABELS = {
    "none":           "no nudge",
    "phi":            "phi snapshot",
    "phi_dphi":       "phi + d_t phi",
    "phi_dphi_d2phi": "phi + d_t phi + d_t^2 phi",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth-dir", required=True)
    ap.add_argument("--n-snaps", type=int, default=6)
    ap.add_argument("--bandwidth", type=float, default=0.1)
    args = ap.parse_args()

    truth_dir = Path(args.truth_dir)
    parent = truth_dir.parent
    name = truth_dir.name
    truth_h5 = truth_dir / "particles.h5"
    if not truth_h5.exists():
        raise SystemExit(f"missing truth particles.h5 at {truth_h5}")

    with h5py.File(truth_h5, "r") as f:
        Nt = f["x"].shape[0]
        L = float(f.attrs.get("L", 4.0 * np.pi))
        dt = float(f.attrs.get("dt", 0.01))
    snap_idx = sorted({int(round(i * (Nt - 1) / (args.n_snaps - 1)))
                       for i in range(args.n_snaps)})
    times = [n * dt for n in snap_idx]
    print(f"[viz-h5] snap indices: {snap_idx}  times: {times}")

    # collect modes that have particles.h5
    mode_paths: dict[str, Path] = {}
    for mode in MODES:
        p = parent / mode / "particles.h5"
        if p.exists():
            mode_paths[mode] = p
    if not mode_paths:
        raise SystemExit("no assim particles.h5 found")

    cols = ["truth"] + list(mode_paths.keys())
    n_cols = len(cols)
    print(f"[viz-h5] columns: {cols}")

    # ---- velocity_kde overlay: 6 panels, all modes overlayed on each ----
    v_grid = np.linspace(-6.0, 6.0, 400)

    def kde(v, w):
        bins = len(v_grid)
        v_min, v_max = v_grid[0], v_grid[-1]
        dv = (v_max - v_min) / (bins - 1)
        edges = np.linspace(v_min - 0.5 * dv, v_max + 0.5 * dv, bins + 1)
        h, _ = np.histogram(v, bins=edges, weights=w, density=True)
        bw = args.bandwidth
        if bw > 0:
            n_ker = int(np.ceil(4 * bw / dv))
            kx = np.arange(-n_ker, n_ker + 1) * dv
            ker = np.exp(-0.5 * (kx / bw) ** 2)
            ker /= ker.sum()
            h = np.convolve(h, ker, mode="same")
        return h

    # cache truth+assim KDEs at each snapshot
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True, sharey=True)
    for i, n in enumerate(snap_idx):
        ax = axes.flat[i]
        # truth
        with h5py.File(truth_h5, "r") as f:
            v = f["v"][n]
            wp = f["w"][:]
        ax.plot(v_grid, kde(v, wp), color="k", lw=1.6, label="truth")
        # modes
        for mode, p in mode_paths.items():
            with h5py.File(p, "r") as f:
                v = f["v"][n]
                wp = f["w"][:]
            ax.plot(v_grid, kde(v, wp), lw=1.0, alpha=0.85, label=LABELS[mode])
        ax.set_title(f"t = {n*dt:.1f}")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")
    for ax in axes[-1, :]:
        ax.set_xlabel("v")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$f(v, t)$")
    fig.suptitle(f"{name}: velocity marginal — truth vs assim modes")
    fig.tight_layout()
    out_a = truth_dir / "assim_vkde_compare.png"
    fig.savefig(out_a, dpi=140)
    plt.close(fig)
    print(f"[viz-h5] wrote {out_a}")

    # ---- phase-space heatmap: 6 rows (snapshots) x N cols (truth + modes) ----
    Nx_ps, Nv_ps = 128, 128
    extent = (0.0, L, -6.0, 6.0)

    # color scale from truth at t=0
    with h5py.File(truth_h5, "r") as f:
        x0 = f["x"][snap_idx[0]]
        v0 = f["v"][snap_idx[0]]
        w0 = f["w"][:]
    H0, _, _ = np.histogram2d(
        x0, v0, bins=(Nx_ps, Nv_ps), range=((0.0, L), (-6.0, 6.0)), weights=w0,
    )
    vmax_img = float(np.percentile(H0.T, 99.5))

    n_rows = len(snap_idx)
    fig2, axes2 = plt.subplots(
        n_rows, n_cols, figsize=(2.7 * n_cols, 2.0 * n_rows),
        sharex=True, sharey=True,
    )
    axes2 = np.atleast_2d(axes2)

    for i, n in enumerate(snap_idx):
        # truth
        with h5py.File(truth_h5, "r") as f:
            x = f["x"][n]
            v = f["v"][n]
            wp = f["w"][:]
        H, _, _ = np.histogram2d(
            x, v, bins=(Nx_ps, Nv_ps), range=((0.0, L), (-6.0, 6.0)), weights=wp,
        )
        im = axes2[i, 0].imshow(H.T, origin="lower", extent=extent,
                                aspect="auto", cmap="viridis",
                                vmin=0.0, vmax=vmax_img)
        if i == 0:
            axes2[i, 0].set_title("truth")
        axes2[i, 0].set_ylabel(f"t={n*dt:.1f}\nv")
        # modes
        for j_col, (mode, p) in enumerate(mode_paths.items(), start=1):
            with h5py.File(p, "r") as f:
                x = f["x"][n]
                v = f["v"][n]
                wp = f["w"][:]
            H, _, _ = np.histogram2d(
                x, v, bins=(Nx_ps, Nv_ps), range=((0.0, L), (-6.0, 6.0)), weights=wp,
            )
            axes2[i, j_col].imshow(H.T, origin="lower", extent=extent,
                                   aspect="auto", cmap="viridis",
                                   vmin=0.0, vmax=vmax_img)
            if i == 0:
                axes2[i, j_col].set_title(LABELS[mode], fontsize=9)
    for ax in axes2[-1, :]:
        ax.set_xlabel("x")
    fig2.suptitle(f"{name}: phase-space f(x, v, t) — truth vs assim modes")
    fig2.tight_layout(rect=(0, 0, 0.94, 1))
    cbar_ax = fig2.add_axes((0.95, 0.15, 0.012, 0.7))
    fig2.colorbar(im, cax=cbar_ax)
    fig2.savefig(truth_dir / "assim_phase_space_compare.png", dpi=140)
    plt.close(fig2)
    print(f"[viz-h5] wrote {truth_dir / 'assim_phase_space_compare.png'}")


if __name__ == "__main__":
    main()
