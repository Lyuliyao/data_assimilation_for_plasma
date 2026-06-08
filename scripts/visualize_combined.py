"""Visualize phase-space and velocity-marginal evolution for the
combined-nudge campaign.

Re-runs the truth + three assim variants in lock-step with snapshot saves:

    truth                                — physical reference
    no-nudge                             — baseline (no channels enabled)
    snapshot position only               — best single-channel snapshot result
    combined (channels from the config)  — eqs. 17-18, the new method

and writes:
    results/<name>/velocity_kde.png
    results/<name>/phase_space.png

The "combined" column inherits its channel flags directly from the config
(position_snapshot + position_dtobs + velocity_dtobs by default).

Usage:
    python scripts/visualize_combined.py \
        --config configs/test0_identifiability_strong_combined.yaml
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mfda.assimilation import _sample
from mfda.backend_reference import (
    cic_deposit,
    cic_deposit_current,
    cic_interpolate,
    field_solve,
    make_state,
    push_leapfrog_drift,
    push_leapfrog_half,
)
from mfda.config import ChannelCfg, load
from mfda.filtering import lowpass_filter
from mfda.kinetic_stress import cic_deposit_kinetic_stress
from mfda.observation import ObservationSpec, observe
from mfda.observation_time import (
    time_derivative_observation,
    time_second_derivative_observation,
)
from mfda.poisson import (
    grad_1d,
    solve_chi,
    solve_poisson_1d,
    solve_poisson_from_d2,
    solve_poisson_from_div,
)


def _snapshot(state):
    return {
        "x": state.x.copy(),
        "v": state.v.copy(),
        "w": state.w.copy(),
        "phi": state.phi.copy(),
        "t": state.t,
    }


def run_with_snapshots(cfg, snapshot_steps):
    """Lock-step truth + assim loop with the channel-based dispatch.

    Mirrors mfda.assimilation.run but keeps only per-snapshot state. Returns
    (snaps_truth, snaps_assim). The assim run uses whichever channels the
    cfg has enabled (no override here — caller should set them).
    """
    rng = np.random.default_rng(cfg.seed)
    obs_rng = np.random.default_rng(cfg.observation.rng_seed)

    L, k = cfg.domain.L, cfg.domain.k
    Nx, Np, dt = cfg.pic.Nx, cfg.pic.Np, cfg.pic.dt
    n_steps = cfg.pic.n_steps

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
    tdcfg = cfg.observation.time_derivative
    t2cfg = cfg.observation.time_second_derivative
    ps_ch, vs_ch = nudge.position_snapshot, nudge.velocity_snapshot
    pd_ch, vd_ch = nudge.position_dtobs, nudge.velocity_dtobs
    p2d_ch, v2d_ch = nudge.position_d2tobs, nudge.velocity_d2tobs

    snaps_truth: dict[int, dict] = {}
    snaps_assim: dict[int, dict] = {}
    if 0 in snapshot_steps:
        snaps_truth[0] = _snapshot(truth)
        snaps_assim[0] = _snapshot(assim)

    y_prev: np.ndarray | None = None
    y_prev2: np.ndarray | None = None
    t_prev: float = 0.0
    t_prev2: float = 0.0

    for n in range(n_steps):
        push_leapfrog_half(truth, 0.5)
        push_leapfrog_half(assim, 0.5)
        push_leapfrog_drift(truth)
        push_leapfrog_drift(assim)
        field_solve(truth)
        field_solve(assim)

        need_psi0 = ps_ch.enabled or vs_ch.enabled
        need_psi1_channels = pd_ch.enabled or vd_ch.enabled
        need_psi2_channels = p2d_ch.enabled or v2d_ch.enabled
        any_observation = need_psi0 or need_psi1_channels or need_psi2_channels

        if (n % obs_spec.every_q) == 0 and any_observation:
            t_now_obs = (n + 1) * dt
            y = observe(truth.phi, obs_spec, rng=obs_rng)

            grad_psi0_grid = None
            if need_psi0:
                resid = assim.phi - y
                if nudge.lowpass_k_cut_frac < 1.0:
                    resid = lowpass_filter(
                        resid, L,
                        k_cut_frac=nudge.lowpass_k_cut_frac,
                        sharpness=nudge.lowpass_sharpness,
                    )
                psi0 = solve_poisson_1d(resid - resid.mean(), L)
                grad_psi0_grid = grad_1d(psi0, L)

            grad_psi1_grid = None
            lap_psi1_grid = None
            need_psi1 = (
                need_psi1_channels and tdcfg.enabled and y_prev is not None
            )
            if need_psi1:
                dt_obs = t_now_obs - t_prev
                z = time_derivative_observation(
                    y, y_prev, dt_obs, L,
                    lowpass_k_cut_frac=tdcfg.lowpass_k_cut_frac,
                    lowpass_sharpness=tdcfg.lowpass_sharpness,
                )
                j_a = cic_deposit_current(assim.x, assim.v, assim.w, L, Nx)
                dphi_dt = solve_poisson_from_div(j_a, L)
                psi1_resid = dphi_dt - z
                psi1_resid = psi1_resid - psi1_resid.mean()
                psi1 = solve_poisson_1d(psi1_resid, L)
                grad_psi1_grid = grad_1d(psi1, L)
                if pd_ch.enabled:
                    lap_psi1_grid = grad_1d(grad_psi1_grid, L)

            # psi2 (second-time-derivative adjoint) — needed for v2d/p2d.
            grad_psi2_grid: np.ndarray | None = None
            lap_psi2_grid: np.ndarray | None = None
            grad3_psi2_grid: np.ndarray | None = None
            grad_chi_grid: np.ndarray | None = None
            grad_E_h1psi2_grid: np.ndarray | None = None
            need_psi2 = (
                need_psi2_channels
                and t2cfg.enabled
                and y_prev is not None
                and y_prev2 is not None
            )
            if need_psi2:
                dt_obs2 = t_now_obs - t_prev
                w_obs = time_second_derivative_observation(
                    y, y_prev, y_prev2, dt_obs2, L,
                    lowpass_k_cut_frac=t2cfg.lowpass_k_cut_frac,
                    lowpass_sharpness=t2cfg.lowpass_sharpness,
                )
                M_a = cic_deposit_kinetic_stress(assim.x, assim.v, assim.w, L, Nx)
                rho_a = cic_deposit(assim.x, assim.w, L, Nx)
                d2phi_dt = solve_poisson_from_d2(M_a, rho_a, assim.E, L)
                psi2_resid = d2phi_dt - w_obs
                psi2_resid = psi2_resid - psi2_resid.mean()
                psi2 = solve_poisson_1d(psi2_resid, L)
                grad_psi2_grid = grad_1d(psi2, L)
                lap_psi2_grid = grad_1d(grad_psi2_grid, L)
                if p2d_ch.enabled:
                    grad3_psi2_grid = grad_1d(lap_psi2_grid, L)
                    chi = solve_chi(rho_a, grad_psi2_grid, L)
                    grad_chi_grid = grad_1d(chi, L)
                    grad_E_h1psi2_grid = grad_1d(assim.E * grad_psi2_grid, L)

            any_position = (
                ps_ch.enabled
                or (need_psi1 and pd_ch.enabled)
                or (need_psi2 and p2d_ch.enabled)
            )
            if any_position:
                dx_total = np.zeros_like(assim.x)
                if ps_ch.enabled:
                    g0p = cic_interpolate(grad_psi0_grid, assim.x, L)
                    dx_total = dx_total - ps_ch.gamma * g0p * dt
                if need_psi1 and pd_ch.enabled:
                    h1p = cic_interpolate(lap_psi1_grid, assim.x, L)
                    dx_total = dx_total - pd_ch.gamma * pd_ch.alpha * assim.v * h1p * dt
                if need_psi2 and p2d_ch.enabled:
                    h3_p = cic_interpolate(grad3_psi2_grid, assim.x, L)
                    grad_E_h1psi2_p = cic_interpolate(grad_E_h1psi2_grid, assim.x, L)
                    grad_chi_p = cic_interpolate(grad_chi_grid, assim.x, L)
                    p2d_term = (
                        assim.v * assim.v * h3_p
                        + grad_E_h1psi2_p
                        + grad_chi_p
                    )
                    dx_total = dx_total - p2d_ch.gamma * p2d_ch.alpha * p2d_term * dt
                assim.x = np.mod(assim.x + dx_total, L)
                field_solve(assim)

            any_velocity = (
                vs_ch.enabled
                or (need_psi1 and vd_ch.enabled)
                or (need_psi2 and v2d_ch.enabled)
            )
            if any_velocity:
                dv_total = np.zeros_like(assim.v)
                if vs_ch.enabled:
                    g0p = cic_interpolate(grad_psi0_grid, assim.x, L)
                    dv_total = dv_total + vs_ch.gamma * g0p
                if need_psi1 and vd_ch.enabled:
                    g1p = cic_interpolate(grad_psi1_grid, assim.x, L)
                    dv_total = dv_total + vd_ch.gamma * vd_ch.alpha * g1p
                if need_psi2 and v2d_ch.enabled:
                    h2p = cic_interpolate(lap_psi2_grid, assim.x, L)
                    dv_total = (
                        dv_total
                        + 2.0 * v2d_ch.gamma * v2d_ch.alpha * assim.v * h2p
                    )
                assim.v = assim.v - dv_total * dt

            # Persist y for next step. y_prev2 takes the previous y_prev.
            if tdcfg.enabled or t2cfg.enabled:
                if y_prev is not None:
                    y_prev2 = y_prev
                    t_prev2 = t_prev
                y_prev = y.copy()
                t_prev = t_now_obs

        push_leapfrog_half(truth, 0.5)
        push_leapfrog_half(assim, 0.5)
        truth.t += dt
        assim.t += dt

        step = n + 1
        if step in snapshot_steps:
            snaps_truth[step] = _snapshot(truth)
            snaps_assim[step] = _snapshot(assim)

    return snaps_truth, snaps_assim


def _disable_all_channels(nudge) -> None:
    nudge.position_snapshot = ChannelCfg()
    nudge.velocity_snapshot = ChannelCfg()
    nudge.position_d2tobs = ChannelCfg()
    nudge.velocity_d2tobs = ChannelCfg()
    nudge.position_dtobs = ChannelCfg()
    nudge.velocity_dtobs = ChannelCfg()


def _weighted_hist(v, w, v_grid, bandwidth=0.1):
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
    ap.add_argument("--n-snaps", type=int, default=6)
    ap.add_argument("--bandwidth", type=float, default=0.1)
    args = ap.parse_args()

    cfg = load(args.config)
    Np = cfg.pic.Np
    n_steps = cfg.pic.n_steps
    dt = cfg.pic.dt
    L = cfg.domain.L
    v_min, v_max = cfg.domain.v_min, cfg.domain.v_max

    snap_steps = sorted({int(round(i * n_steps / (args.n_snaps - 1)))
                         for i in range(args.n_snaps)})
    print(f"[viz-combined] snapshot steps: {snap_steps} "
          f"(times: {[s * dt for s in snap_steps]})")

    # 1) Combined run — channels straight from the config.
    print("[viz-combined] running combined (channels from config)...")
    snaps_truth, snaps_combined = run_with_snapshots(cfg, set(snap_steps))

    # 2) Snapshot-position-only — disable everything except position_snapshot.
    print("[viz-combined] running snapshot-position-only...")
    cfg_p = copy.deepcopy(cfg)
    _disable_all_channels(cfg_p.nudge)
    cfg_p.nudge.position_snapshot = ChannelCfg(enabled=True, gamma=1.0)
    _, snaps_pos = run_with_snapshots(cfg_p, set(snap_steps))

    # 3) No-nudge baseline.
    print("[viz-combined] running no-nudge baseline...")
    cfg_n = copy.deepcopy(cfg)
    _disable_all_channels(cfg_n.nudge)
    _, snaps_none = run_with_snapshots(cfg_n, set(snap_steps))

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
            ("no-nudge", snaps_none, "C2", 1.2),
            ("snapshot pos", snaps_pos, "C1", 1.2),
            ("combined", snaps_combined, "C3", 1.5),
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
        f"{cfg.name}: velocity marginal  (Np={Np}, bandwidth={args.bandwidth})"
    )
    fig.tight_layout()
    fig.savefig(outdir / "velocity_kde.png", dpi=150)
    plt.close(fig)
    print(f"[viz-combined] wrote {outdir / 'velocity_kde.png'}")

    # ---- Figure 2: phase-space heatmaps (4 columns) ----
    Nx_ps, Nv_ps = 128, 128
    extent = (0.0, L, v_min, v_max)
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
    titles = ["truth", "no-nudge", "snapshot pos", "combined"]
    for i, step in enumerate(snap_steps):
        t = snaps_truth[step]["t"]
        for j, snaps in enumerate([snaps_truth, snaps_none, snaps_pos, snaps_combined]):
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
    fig2.suptitle(f"{cfg.name}: phase-space f(x, v, t)")
    fig2.tight_layout(rect=(0, 0, 0.92, 1))
    cbar_ax = fig2.add_axes((0.94, 0.15, 0.015, 0.7))
    fig2.colorbar(im, cax=cbar_ax)
    fig2.savefig(outdir / "phase_space.png", dpi=150)
    plt.close(fig2)
    print(f"[viz-combined] wrote {outdir / 'phase_space.png'}")

    # ---- Figure 3: observation-variable matches per moment ----
    # rho(x) is the target of the snapshot phi residual,
    # j(x)   is the target of the dtobs   d/dt phi residual,
    # M(x)   is the target of the d2tobs  d^2/dt^2 phi residual.
    # Plot truth (black) vs combined (red) at each snapshot time.
    Nx_grid = cfg.pic.Nx
    moments = ("rho", "j", "M")
    moment_titles = (
        r"$\rho(x) = \int f\,dv$",
        r"$j(x) = \int v f\,dv$",
        r"$M(x) = \int v^2 f\,dv$",
    )
    fig3, axes3 = plt.subplots(
        len(snap_steps), 3, figsize=(12, 2.3 * len(snap_steps)),
        sharex=True,
    )
    axes3 = np.atleast_2d(axes3)
    x_grid = np.linspace(0.0, L, Nx_grid, endpoint=False)
    for i, step in enumerate(snap_steps):
        t = snaps_truth[step]["t"]
        s_t = snaps_truth[step]
        s_c = snaps_combined[step]
        rho_t = cic_deposit(s_t["x"], s_t["w"], L, Nx_grid)
        rho_c = cic_deposit(s_c["x"], s_c["w"], L, Nx_grid)
        j_t = cic_deposit_current(s_t["x"], s_t["v"], s_t["w"], L, Nx_grid)
        j_c = cic_deposit_current(s_c["x"], s_c["v"], s_c["w"], L, Nx_grid)
        M_t = cic_deposit_kinetic_stress(s_t["x"], s_t["v"], s_t["w"], L, Nx_grid)
        M_c = cic_deposit_kinetic_stress(s_c["x"], s_c["v"], s_c["w"], L, Nx_grid)
        for j_col, (mt, mc, title) in enumerate(zip(
            (rho_t, j_t, M_t), (rho_c, j_c, M_c), moment_titles
        )):
            ax = axes3[i, j_col]
            ax.plot(x_grid, mt, color="k", lw=1.4, label="truth")
            ax.plot(x_grid, mc, color="C3", lw=1.2, label="combined")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(title)
                ax.legend(fontsize=8, loc="upper right")
            if j_col == 0:
                ax.set_ylabel(f"t={t:.1f}")
    for ax in axes3[-1, :]:
        ax.set_xlabel("x")
    fig3.suptitle(
        f"{cfg.name}: observation-variable matches  "
        f"(per-moment, truth vs combined)"
    )
    fig3.tight_layout()
    fig3.savefig(outdir / "observation_moments.png", dpi=150)
    plt.close(fig3)
    print(f"[viz-combined] wrote {outdir / 'observation_moments.png'}")


if __name__ == "__main__":
    main()
