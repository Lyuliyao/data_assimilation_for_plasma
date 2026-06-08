"""Assim-only PIC loop reading the smoothed observation from disk.

Skips the parallel truth simulation entirely — y(t,x), z(t,x), w(t,x)
are read from observation_smoothed.npz produced by
scripts/save_smoothed_observation.py.

IC is the user-specified phase-shifted Maxwellian:

    f^model_0(x, v) = M(v) * (1 + epsilon cos(k x + theta))

with epsilon and theta from the CLI. Default theta = pi/3.

Modes:
  none            no nudging (baseline)
  phi             position_snapshot only — uses y only
  phi_dphi        position_snapshot + position_dtobs + velocity_dtobs
                  uses y and z
  phi_dphi_d2phi  full 5-channel combined: ps + pd + vd + p2d + v2d
                  uses y, z, and w

Saves particles at every step in particles.h5 alongside per-step
diagnostics in assim_diagnostics.npz.

Usage:
    python scripts/run_assim_from_obs.py \
        --truth-dir cases/position_mismatch/results/truth \
        --config cases/position_mismatch/config.yaml \
        --mode phi_dphi_d2phi --gamma 1.0 --alpha 1.0 --beta 0.01
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from mfda.backend_reference import (
    cic_deposit, cic_deposit_current, cic_interpolate, field_solve,
    make_state, push_leapfrog_drift, push_leapfrog_half,
)
from mfda.assimilation import _sample
from mfda.config import load
from mfda.filtering import lowpass_filter
from mfda.kinetic_stress import cic_deposit_kinetic_stress
from mfda.poisson import (
    grad_1d, solve_chi, solve_poisson_1d, solve_poisson_from_d2,
    solve_poisson_from_div,
)


MODES = {
    "none":           dict(),
    "phi":            dict(ps=True),
    "phi_dphi":       dict(ps=True, pd=True, vd=True),
    "phi_dphi_d2phi": dict(ps=True, pd=True, vd=True, p2d=True, v2d=True),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth-dir", required=True,
                    help="dir containing observation_smoothed.npz and truth.npz")
    ap.add_argument("--config", required=True,
                    help="YAML providing PIC params and domain")
    ap.add_argument("--mode", required=True, choices=list(MODES.keys()))
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="dtobs coefficient (used by pd, vd channels)")
    ap.add_argument("--beta", type=float, default=0.01,
                    help="d2tobs coefficient (used by p2d, v2d channels)")
    ap.add_argument("--epsilon", type=float, default=None,
                    help="OVERRIDE for assim_ic.alpha (only applied when set)")
    ap.add_argument("--theta", type=float, default=None,
                    help="OVERRIDE for assim_ic.theta0 (only applied when set, "
                         "and only meaningful for kind=ic_phase_error)")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--no-particles-h5", action="store_true")
    ap.add_argument("--lowpass-k-cut", type=float, default=0.40)
    args = ap.parse_args()

    cfg = load(args.config)
    truth_dir = Path(args.truth_dir)
    obs = np.load(truth_dir / "observation_smoothed.npz")
    truth = np.load(truth_dir / "truth.npz")
    y = np.asarray(obs["phi"])              # smoothed observation y(t, x)
    z = np.asarray(obs["z"])                # d_t y, smoothed
    w = np.asarray(obs["w"])                # d_t^2 y, smoothed
    rho_truth = np.asarray(truth["rho"])
    L = cfg.domain.L
    k = cfg.domain.k
    Nx = cfg.pic.Nx
    Np = cfg.pic.Np
    dt = cfg.pic.dt
    n_steps = cfg.pic.n_steps
    flags = MODES[args.mode]

    # Sample assim particles from cfg.assim_ic (CLI args override alpha/theta0
    # if provided, for backwards-compat with the position_mismatch test).
    if args.epsilon is not None:
        cfg.assim_ic.alpha = args.epsilon
    if args.theta is not None:
        cfg.assim_ic.theta0 = args.theta
    rng = np.random.default_rng(cfg.seed)
    x_a, v_a, w_a = _sample(cfg.assim_ic, Np, L, k, rng)
    state = make_state(x_a, v_a, w_a, L, Nx, dt)
    print(f"[assim] assim_ic kind={cfg.assim_ic.kind} "
          f"alpha={cfg.assim_ic.alpha} u0={cfg.assim_ic.u0} "
          f"sigma={cfg.assim_ic.sigma} theta0={cfg.assim_ic.theta0}")

    # Output dir.
    out_dir = Path(args.output_dir) if args.output_dir else (
        truth_dir.parent / args.mode
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[assim] mode={args.mode}  out_dir={out_dir}")

    # H5 setup.
    h5f = None
    if not args.no_particles_h5:
        h5_path = out_dir / "particles.h5"
        h5f = h5py.File(h5_path, "w")
        chunks = (1, Np)
        h5f.create_dataset("x", (n_steps + 1, Np), dtype=np.float64, chunks=chunks)
        h5f.create_dataset("v", (n_steps + 1, Np), dtype=np.float64, chunks=chunks)
        h5f.create_dataset("w", data=state.w)
        h5f.create_dataset("t", (n_steps + 1,), dtype=np.float64)
        h5f.attrs["mode"] = args.mode
        h5f.attrs["gamma"] = args.gamma
        h5f.attrs["alpha"] = args.alpha
        h5f.attrs["beta"] = args.beta
        h5f.attrs["epsilon"] = float(cfg.assim_ic.alpha)
        h5f.attrs["theta"] = float(cfg.assim_ic.theta0)
        h5f.attrs["u0"] = float(cfg.assim_ic.u0)
        h5f.attrs["sigma"] = float(cfg.assim_ic.sigma)
        h5f.attrs["assim_ic_kind"] = str(cfg.assim_ic.kind)
        h5f["x"][0] = state.x
        h5f["v"][0] = state.v
        h5f["t"][0] = 0.0

    # Per-step diagnostics.
    diag_t = np.zeros(n_steps + 1)
    diag_e_phi = np.zeros(n_steps + 1)
    diag_e_rho = np.zeros(n_steps + 1)
    diag_M_total = np.zeros(n_steps + 1)
    diag_t[0] = 0.0
    diag_e_phi[0] = float(np.sqrt(L / Nx) * np.linalg.norm(state.phi - y[0]))
    rho_0 = cic_deposit(state.x, state.w, L, Nx)
    diag_e_rho[0] = float(np.sqrt(L / Nx) * np.linalg.norm(rho_0 - rho_truth[0]))
    diag_M_total[0] = float(np.sum(state.w * state.v ** 2))

    for n in range(n_steps):
        # 1) leapfrog: half-kick, drift, field-solve
        push_leapfrog_half(state, 0.5)
        push_leapfrog_drift(state)
        field_solve(state)

        # 2) nudge block (if any channel enabled)
        if any(flags.values()):
            n_obs = n + 1
            y_n = y[n_obs]

            grad_psi0 = None
            need_psi0 = flags.get("ps", False) or flags.get("vs", False)
            if need_psi0:
                resid = state.phi - y_n
                if args.lowpass_k_cut < 1.0:
                    resid = lowpass_filter(resid, L, k_cut_frac=args.lowpass_k_cut)
                psi0 = solve_poisson_1d(resid - resid.mean(), L)
                grad_psi0 = grad_1d(psi0, L)

            grad_psi1 = lap_psi1 = None
            need_psi1 = flags.get("pd", False) or flags.get("vd", False)
            if need_psi1:
                z_n = z[n_obs]
                j_a = cic_deposit_current(state.x, state.v, state.w, L, Nx)
                dphi_dt = solve_poisson_from_div(j_a, L)
                psi1_resid = dphi_dt - z_n
                psi1_resid -= psi1_resid.mean()
                psi1 = solve_poisson_1d(psi1_resid, L)
                grad_psi1 = grad_1d(psi1, L)
                if flags.get("pd", False):
                    lap_psi1 = grad_1d(grad_psi1, L)

            grad_psi2 = lap_psi2 = grad3_psi2 = grad_chi = grad_E_h1psi2 = None
            need_psi2 = flags.get("p2d", False) or flags.get("v2d", False)
            if need_psi2:
                w_n = w[n_obs]
                M_a = cic_deposit_kinetic_stress(state.x, state.v, state.w, L, Nx)
                rho_a_local = cic_deposit(state.x, state.w, L, Nx)
                d2phi_dt = solve_poisson_from_d2(M_a, rho_a_local, state.E, L)
                psi2_resid = d2phi_dt - w_n
                psi2_resid -= psi2_resid.mean()
                psi2 = solve_poisson_1d(psi2_resid, L)
                grad_psi2 = grad_1d(psi2, L)
                lap_psi2 = grad_1d(grad_psi2, L)
                if flags.get("p2d", False):
                    grad3_psi2 = grad_1d(lap_psi2, L)
                    chi = solve_chi(rho_a_local, grad_psi2, L)
                    grad_chi = grad_1d(chi, L)
                    grad_E_h1psi2 = grad_1d(state.E * grad_psi2, L)

            # Position channel (combine snapshot, bilinear-dtobs, bilinear-quadratic-d2tobs).
            any_position = flags.get("ps", False) or \
                (need_psi1 and flags.get("pd", False)) or \
                (need_psi2 and flags.get("p2d", False))
            if any_position:
                dx_total = np.zeros_like(state.x)
                if flags.get("ps", False):
                    g0p = cic_interpolate(grad_psi0, state.x, L)
                    dx_total -= args.gamma * g0p * dt
                if need_psi1 and flags.get("pd", False):
                    h1p = cic_interpolate(lap_psi1, state.x, L)
                    dx_total -= args.gamma * args.alpha * state.v * h1p * dt
                if need_psi2 and flags.get("p2d", False):
                    h3p = cic_interpolate(grad3_psi2, state.x, L)
                    grad_Eh1psi2_p = cic_interpolate(grad_E_h1psi2, state.x, L)
                    grad_chip = cic_interpolate(grad_chi, state.x, L)
                    p2d_term = state.v ** 2 * h3p + grad_Eh1psi2_p + grad_chip
                    dx_total -= args.gamma * args.beta * p2d_term * dt
                state.x = np.mod(state.x + dx_total, L)
                field_solve(state)

            # Velocity channel.
            any_velocity = flags.get("vs", False) or \
                (need_psi1 and flags.get("vd", False)) or \
                (need_psi2 and flags.get("v2d", False))
            if any_velocity:
                dv_total = np.zeros_like(state.v)
                if flags.get("vs", False):
                    g0p = cic_interpolate(grad_psi0, state.x, L)
                    dv_total += args.gamma * g0p
                if need_psi1 and flags.get("vd", False):
                    g1p = cic_interpolate(grad_psi1, state.x, L)
                    dv_total += args.gamma * args.alpha * g1p
                if need_psi2 and flags.get("v2d", False):
                    h2p = cic_interpolate(lap_psi2, state.x, L)
                    dv_total += 2.0 * args.gamma * args.beta * state.v * h2p
                state.v -= dv_total * dt

        # 3) second half-kick
        push_leapfrog_half(state, 0.5)
        state.t += dt

        # Snapshot to H5.
        if h5f is not None:
            h5f["x"][n + 1] = state.x
            h5f["v"][n + 1] = state.v
            h5f["t"][n + 1] = state.t

        # Diagnostics.
        diag_t[n + 1] = state.t
        diag_e_phi[n + 1] = float(np.sqrt(L / Nx) * np.linalg.norm(state.phi - y[n + 1]))
        rho_a = cic_deposit(state.x, state.w, L, Nx)
        diag_e_rho[n + 1] = float(np.sqrt(L / Nx) * np.linalg.norm(rho_a - rho_truth[n + 1]))
        diag_M_total[n + 1] = float(np.sum(state.w * state.v ** 2))

    if h5f is not None:
        h5f.close()
        print(f"[assim] wrote {out_dir / 'particles.h5'}")

    np.savez_compressed(
        out_dir / "assim_diagnostics.npz",
        t=diag_t, e_phi=diag_e_phi, e_rho=diag_e_rho, M_total=diag_M_total,
        mode=np.array(args.mode), gamma=args.gamma, alpha=args.alpha,
        beta=args.beta,
        epsilon=float(cfg.assim_ic.alpha),
        theta=float(cfg.assim_ic.theta0),
        u0=float(cfg.assim_ic.u0),
        sigma=float(cfg.assim_ic.sigma),
    )
    manifest = {
        "kind": "assim_from_obs",
        "config": cfg.model_dump(),
        "truth_dir": str(truth_dir),
        "mode": args.mode,
        "gamma": args.gamma, "alpha": args.alpha, "beta": args.beta,
    }
    (out_dir / "assim_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[assim] wrote {out_dir / 'assim_diagnostics.npz'}")
    print(f"[assim] final  e_phi = {diag_e_phi[-1]:.6e}")
    print(f"[assim] final  e_rho = {diag_e_rho[-1]:.6e}")


if __name__ == "__main__":
    main()
