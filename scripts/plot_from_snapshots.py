"""Render velocity_kde.png, phase_space.png, observation_moments.png from
saved particle snapshots produced by scripts/run_with_snapshots.py.

No simulation re-run — purely reads results/<name>/snapshots.npz and plots.

Usage:
    python scripts/plot_from_snapshots.py \
        --config configs/test0_identifiability_linear_combined5_best.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mfda.backend_reference import (
    cic_deposit,
    cic_deposit_current,
)
from mfda.config import load
from mfda.kinetic_stress import cic_deposit_kinetic_stress


def _weighted_kde(v, w, v_grid, bandwidth=0.1):
    bins = len(v_grid)
    v_min, v_max = v_grid[0], v_grid[-1]
    dv = (v_max - v_min) / (bins - 1)
    edges = np.linspace(v_min - 0.5 * dv, v_max + 0.5 * dv, bins + 1)
    h, _ = np.histogram(v, bins=edges, weights=w, density=True)
    if bandwidth and bandwidth > 0:
        n_ker = int(np.ceil(4 * bandwidth / dv))
        kx = np.arange(-n_ker, n_ker + 1) * dv
        kernel = np.exp(-0.5 * (kx / bandwidth) ** 2)
        kernel /= kernel.sum()
        h = np.convolve(h, kernel, mode="same")
    return h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--bandwidth", type=float, default=0.1)
    args = ap.parse_args()

    cfg = load(args.config)
    L = cfg.domain.L
    v_min, v_max = cfg.domain.v_min, cfg.domain.v_max
    Nx_grid = cfg.pic.Nx
    outdir = Path(cfg.outputs_dir) / cfg.name
    snap_path = outdir / "snapshots.npz"
    if not snap_path.exists():
        raise SystemExit(f"snapshots not found at {snap_path}; "
                         f"run scripts/run_with_snapshots.py --config {args.config} first")

    data = np.load(snap_path)
    snap_steps = list(data["snap_steps"])
    print(f"[plot] {len(snap_steps)} snapshots: steps {snap_steps}")

    snaps_truth = {}
    snaps_assim = {}
    for step in snap_steps:
        snaps_truth[step] = {k: data[f"truth_{step}_{k}"]
                              for k in ("x", "v", "w", "t")}
        snaps_assim[step] = {k: data[f"assim_{step}_{k}"]
                              for k in ("x", "v", "w", "t")}

    # ---- Figure 1: velocity-marginal evolution ----
    v_grid = np.linspace(v_min, v_max, 400)
    ncols = min(3, len(snap_steps))
    nrows = int(np.ceil(len(snap_steps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 2.8 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for i, step in enumerate(snap_steps):
        ax = axes.flat[i]
        t = float(snaps_truth[step]["t"])
        for label, snaps, color, lw in [
            ("truth", snaps_truth, "k", 1.6),
            ("assim", snaps_assim, "C3", 1.4),
        ]:
            s = snaps[step]
            h = _weighted_kde(s["v"], s["w"], v_grid, bandwidth=args.bandwidth)
            ax.plot(v_grid, h, color=color, label=label, lw=lw)
        ax.set_title(f"t = {t:.1f}")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")
    for j in range(len(snap_steps), axes.size):
        axes.flat[j].axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("v")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$f(v, t)$")
    fig.suptitle(f"{cfg.name}: velocity marginal")
    fig.tight_layout()
    fig.savefig(outdir / "velocity_kde.png", dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {outdir / 'velocity_kde.png'}")

    # ---- Figure 2: phase-space heatmaps (truth vs assim, 2 columns) ----
    Nx_ps, Nv_ps = 128, 128
    extent = (0.0, L, v_min, v_max)
    s0 = snaps_truth[snap_steps[0]]
    H0, _, _ = np.histogram2d(
        s0["x"], s0["v"], bins=(Nx_ps, Nv_ps),
        range=((0.0, L), (v_min, v_max)), weights=s0["w"],
    )
    vmax_img = float(np.percentile(H0.T, 99.5))
    fig2, axes2 = plt.subplots(
        len(snap_steps), 2, figsize=(7, 2.3 * len(snap_steps)),
        sharex=True, sharey=True,
    )
    axes2 = np.atleast_2d(axes2)
    for i, step in enumerate(snap_steps):
        t = float(snaps_truth[step]["t"])
        for j_col, snaps in enumerate([snaps_truth, snaps_assim]):
            s = snaps[step]
            H, _, _ = np.histogram2d(
                s["x"], s["v"], bins=(Nx_ps, Nv_ps),
                range=((0.0, L), (v_min, v_max)), weights=s["w"],
            )
            ax = axes2[i, j_col]
            im = ax.imshow(H.T, origin="lower", extent=extent, aspect="auto",
                           cmap="viridis", vmin=0.0, vmax=vmax_img)
            if i == 0:
                ax.set_title(["truth", "assim"][j_col])
            if j_col == 0:
                ax.set_ylabel(f"t={t:.1f}\nv")
    for ax in axes2[-1, :]:
        ax.set_xlabel("x")
    fig2.suptitle(f"{cfg.name}: phase-space f(x, v, t)")
    fig2.tight_layout(rect=(0, 0, 0.92, 1))
    cbar_ax = fig2.add_axes((0.94, 0.15, 0.015, 0.7))
    fig2.colorbar(im, cax=cbar_ax)
    fig2.savefig(outdir / "phase_space.png", dpi=150)
    plt.close(fig2)
    print(f"[plot] wrote {outdir / 'phase_space.png'}")

    # ---- Figure 3: observation-variable moments (rho, j, M) ----
    fig3, axes3 = plt.subplots(
        len(snap_steps), 3, figsize=(12, 2.3 * len(snap_steps)),
        sharex=True,
    )
    axes3 = np.atleast_2d(axes3)
    x_grid = np.linspace(0.0, L, Nx_grid, endpoint=False)
    moment_titles = (
        r"$\rho(x) = \int f\,dv$",
        r"$j(x) = \int v f\,dv$",
        r"$M(x) = \int v^2 f\,dv$",
    )
    for i, step in enumerate(snap_steps):
        t = float(snaps_truth[step]["t"])
        s_t = snaps_truth[step]
        s_a = snaps_assim[step]
        rho_t = cic_deposit(s_t["x"], s_t["w"], L, Nx_grid)
        rho_a = cic_deposit(s_a["x"], s_a["w"], L, Nx_grid)
        j_t = cic_deposit_current(s_t["x"], s_t["v"], s_t["w"], L, Nx_grid)
        j_a = cic_deposit_current(s_a["x"], s_a["v"], s_a["w"], L, Nx_grid)
        M_t = cic_deposit_kinetic_stress(s_t["x"], s_t["v"], s_t["w"], L, Nx_grid)
        M_a = cic_deposit_kinetic_stress(s_a["x"], s_a["v"], s_a["w"], L, Nx_grid)
        for j_col, (mt, ma, title) in enumerate(zip(
            (rho_t, j_t, M_t), (rho_a, j_a, M_a), moment_titles
        )):
            ax = axes3[i, j_col]
            ax.plot(x_grid, mt, color="k", lw=1.4, label="truth")
            ax.plot(x_grid, ma, color="C3", lw=1.2, label="assim")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(title)
                ax.legend(fontsize=8, loc="upper right")
            if j_col == 0:
                ax.set_ylabel(f"t={t:.1f}")
    for ax in axes3[-1, :]:
        ax.set_xlabel("x")
    fig3.suptitle(f"{cfg.name}: observation-variable matches (truth vs assim)")
    fig3.tight_layout()
    fig3.savefig(outdir / "observation_moments.png", dpi=150)
    plt.close(fig3)
    print(f"[plot] wrote {outdir / 'observation_moments.png'}")


if __name__ == "__main__":
    main()
