"""Visualize Test 0 phase-space and velocity-marginal evolution.

Re-runs the Test-0 assimilation lock-step (truth + velocity-nudge, then
truth + position-nudge) saving particle snapshots at a few evenly spaced
times, and produces:

    results/<name>/velocity_kde.png   — f(v,t) for truth vs both variants
    results/<name>/phase_space.png    — f(x,v,t) heatmaps, truth vs variants

Why re-run?  run_assimilation.py only persists final-time particles; the
visualization needs multiple snapshots, which it is cheaper to regenerate
deterministically (seed is fixed) than to thread a snapshot option through
the main loop.

Usage:
    python scripts/visualize_test0.py --config configs/test0_identifiability.yaml
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mfda.assimilation import _sample
from mfda.backend_reference import (
    cic_interpolate,
    field_solve,
    make_state,
    push_leapfrog_drift,
    push_leapfrog_half,
)
from mfda.config import load
from mfda.filtering import lowpass_filter
from mfda.nudging import apply_nudging
from mfda.observation import ObservationSpec, observe
from mfda.poisson import grad_1d, solve_poisson_1d


def _snapshot(state):
    return {
        "x": state.x.copy(),
        "v": state.v.copy(),
        "w": state.w.copy(),
        "phi": state.phi.copy(),
        "t": state.t,
    }


def run_with_snapshots(cfg, variant, snapshot_steps):
    """Lockstep truth+assim loop that saves particle snapshots at given step indices.

    Mirrors mfda.assimilation.run but keeps only per-snapshot state. Returns
    (snaps_truth, snaps_assim) as dicts keyed by step index.
    """
    rng = np.random.default_rng(cfg.seed)
    obs_rng = np.random.default_rng(cfg.observation.rng_seed)

    L, k = cfg.domain.L, cfg.domain.k
    Nx, Np, dt = cfg.pic.Nx, cfg.pic.Np, cfg.pic.dt

    x_t, v_t, w_t = _sample(cfg.truth_ic, Np, L, k, rng)
    x_a, v_a, w_a = _sample(cfg.assim_ic, Np, L, k, rng)
    truth = make_state(x_t, v_t, w_t, L, Nx, dt)
    assim = make_state(x_a, v_a, w_a, L, Nx, dt)

    obs_spec = ObservationSpec(
        kind=cfg.observation.kind,
        sigma=cfg.observation.sigma,
        every_m=cfg.observation.every_m,
        reconstruction=cfg.observation.reconstruction,
        every_q=cfg.observation.every_q,
        rng_seed=cfg.observation.rng_seed,
    )
    nudge = cfg.nudge

    snaps_truth: dict[int, dict] = {}
    snaps_assim: dict[int, dict] = {}

    if 0 in snapshot_steps:
        snaps_truth[0] = _snapshot(truth)
        snaps_assim[0] = _snapshot(assim)

    for n in range(cfg.pic.n_steps):
        push_leapfrog_half(truth, 0.5)
        push_leapfrog_half(assim, 0.5)
        push_leapfrog_drift(truth)
        push_leapfrog_drift(assim)
        field_solve(truth)
        field_solve(assim)

        if (n % obs_spec.every_q) == 0 and variant != "none":
            y = observe(truth.phi, obs_spec, rng=obs_rng)
            resid = assim.phi - y
            if nudge.lowpass_k_cut_frac < 1.0:
                resid = lowpass_filter(
                    resid, L,
                    k_cut_frac=nudge.lowpass_k_cut_frac,
                    sharpness=nudge.lowpass_sharpness,
                )
            psi = solve_poisson_1d(resid - resid.mean(), L)
            grad_psi_grid = grad_1d(psi, L)
            grad_psi_at_p = cic_interpolate(grad_psi_grid, assim.x, L)
            x_new, v_new = apply_nudging(
                variant, assim.x, assim.v, grad_psi_at_p,
                gamma=nudge.gamma, dt=dt, L=L,
            )
            assim.x, assim.v = x_new, v_new
            if variant == "position":
                field_solve(assim)

        push_leapfrog_half(truth, 0.5)
        push_leapfrog_half(assim, 0.5)
        truth.t += dt
        assim.t += dt

        step = n + 1
        if step in snapshot_steps:
            snaps_truth[step] = _snapshot(truth)
            snaps_assim[step] = _snapshot(assim)

    return snaps_truth, snaps_assim


def _weighted_hist(v, w, v_grid, bandwidth=None):
    """Weighted histogram of particle velocities, normalized to a pdf.

    If bandwidth is given, we smooth by convolving with a Gaussian kernel of
    that std (in units of v). This is effectively a weighted KDE on a grid.
    """
    bins = len(v_grid)
    v_min, v_max = v_grid[0], v_grid[-1]
    dv = (v_max - v_min) / (bins - 1)
    edges = np.linspace(v_min - 0.5 * dv, v_max + 0.5 * dv, bins + 1)
    h, _ = np.histogram(v, bins=edges, weights=w, density=True)
    if bandwidth is None or bandwidth <= 0:
        return h
    # Convolve with a Gaussian on the same grid.
    n_ker = int(np.ceil(4 * bandwidth / dv))
    kx = np.arange(-n_ker, n_ker + 1) * dv
    kernel = np.exp(-0.5 * (kx / bandwidth) ** 2)
    kernel /= kernel.sum()
    return np.convolve(h, kernel, mode="same")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/test0_identifiability.yaml")
    ap.add_argument("--n-snaps", type=int, default=6,
                    help="number of evenly spaced time snapshots (including t=0)")
    ap.add_argument("--bandwidth", type=float, default=0.1,
                    help="KDE bandwidth in velocity units (0 to disable smoothing)")
    args = ap.parse_args()

    cfg = load(args.config)
    Np = cfg.pic.Np
    n_steps = cfg.pic.n_steps
    dt = cfg.pic.dt
    L = cfg.domain.L
    v_min, v_max = cfg.domain.v_min, cfg.domain.v_max

    snap_steps = sorted({int(round(i * n_steps / (args.n_snaps - 1)))
                         for i in range(args.n_snaps)})

    print(f"[viz] snapshot steps: {snap_steps} (times: "
          f"{[s * dt for s in snap_steps]})")

    print("[viz] running velocity variant...")
    cfg_v = copy.deepcopy(cfg)
    cfg_v.nudge.variant = "velocity"
    snaps_truth, snaps_vel = run_with_snapshots(cfg_v, "velocity", set(snap_steps))

    print("[viz] running position variant...")
    cfg_p = copy.deepcopy(cfg)
    cfg_p.nudge.variant = "position"
    _, snaps_pos = run_with_snapshots(cfg_p, "position", set(snap_steps))

    print("[viz] running no-nudge baseline...")
    cfg_n = copy.deepcopy(cfg)
    cfg_n.nudge.variant = "none"
    _, snaps_none = run_with_snapshots(cfg_n, "none", set(snap_steps))

    outdir = Path(cfg.outputs_dir) / cfg.name
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: velocity-marginal evolution ----
    v_grid = np.linspace(v_min, v_max, 400)
    ncols = min(3, len(snap_steps))
    nrows = int(np.ceil(len(snap_steps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 2.8 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    for i, step in enumerate(snap_steps):
        ax = axes.flat[i]
        t = snaps_truth[step]["t"]
        for label, snaps, color, lw in [
            ("truth", snaps_truth, "k", 1.6),
            ("no-nudge", snaps_none, "C2", 1.3),
            ("velocity-nudge", snaps_vel, "C0", 1.3),
            ("position-nudge", snaps_pos, "C1", 1.3),
        ]:
            s = snaps[step]
            h = _weighted_hist(s["v"], s["w"], v_grid, bandwidth=args.bandwidth)
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
    fig.suptitle(
        f"{cfg.name}: velocity marginal  "
        f"(γ={cfg.nudge.gamma}, Np={Np}, bandwidth={args.bandwidth})"
    )
    fig.tight_layout()
    fig.savefig(outdir / "velocity_kde.png", dpi=150)
    plt.close(fig)
    print(f"[viz] wrote {outdir / 'velocity_kde.png'}")

    # ---- Figure 2: phase-space heatmaps ----
    Nx_ps, Nv_ps = 128, 128
    extent = (0.0, L, v_min, v_max)
    # Shared color scale from truth @ t=0 to keep visual comparison honest.
    s0 = snaps_truth[snap_steps[0]]
    H0, _, _ = np.histogram2d(
        s0["x"], s0["v"], bins=(Nx_ps, Nv_ps),
        range=((0.0, L), (v_min, v_max)), weights=s0["w"],
    )
    vmax_img = float(np.percentile(H0.T, 99.5))

    fig2, axes2 = plt.subplots(
        len(snap_steps), 4, figsize=(12, 2.3 * len(snap_steps)),
        sharex=True, sharey=True,
    )
    axes2 = np.atleast_2d(axes2)
    titles = ["truth", "no-nudge", "velocity-nudge", "position-nudge"]
    for i, step in enumerate(snap_steps):
        t = snaps_truth[step]["t"]
        for j, (label, snaps) in enumerate([
            ("truth", snaps_truth),
            ("none", snaps_none),
            ("velocity", snaps_vel),
            ("position", snaps_pos),
        ]):
            s = snaps[step]
            H, _, _ = np.histogram2d(
                s["x"], s["v"], bins=(Nx_ps, Nv_ps),
                range=((0.0, L), (v_min, v_max)), weights=s["w"],
            )
            ax = axes2[i, j]
            im = ax.imshow(
                H.T, origin="lower", extent=extent, aspect="auto",
                cmap="viridis", vmin=0.0, vmax=vmax_img,
            )
            if i == 0:
                ax.set_title(titles[j])
            if j == 0:
                ax.set_ylabel(f"t={t:.1f}\nv")
    for ax in axes2[-1, :]:
        ax.set_xlabel("x")
    fig2.suptitle(f"{cfg.name}: phase-space f(x, v, t)  (γ={cfg.nudge.gamma})")
    fig2.tight_layout(rect=(0, 0, 0.92, 1))
    cbar_ax = fig2.add_axes((0.94, 0.15, 0.015, 0.7))
    fig2.colorbar(im, cax=cbar_ax)
    fig2.savefig(outdir / "phase_space.png", dpi=150)
    plt.close(fig2)
    print(f"[viz] wrote {outdir / 'phase_space.png'}")


if __name__ == "__main__":
    main()
