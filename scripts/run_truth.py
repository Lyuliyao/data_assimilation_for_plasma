"""Run the truth simulation and save phi_truth(t_n, x) plus a final snapshot.

By default ALSO writes particles.h5 with x[t,p], v[t,p], w[p] at every
PIC step so any downstream analysis (different smoothing, different
moments, visualisation) can read from disk instead of re-running.
At Np=1e6, n_steps=4000 the file is ~32 GB before compression, ~10-15
GB with gzip-4. Pass --no-particles-h5 to disable.

Usage:
    python scripts/run_truth.py --config configs/test1_linear_landau.yaml

Output: results/<name>/truth.npz, results/<name>/particles.h5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from mfda.assimilation import _sample  # noqa: F401 (reuse sampler dispatch)
from mfda.backend_reference import (
    cic_deposit,
    cic_deposit_current,
    field_solve,
    make_state,
    push_leapfrog_drift,
    push_leapfrog_half,
)
from mfda.config import load
from mfda.kinetic_stress import cic_deposit_kinetic_stress


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", default=None,
                    help="override output file (default: results/<name>/truth.npz)")
    ap.add_argument("--np", dest="Np", type=int, default=None,
                    help="override cfg.pic.Np (for noise vs. shot-count studies)")
    ap.add_argument("--name-suffix", default="",
                    help="appended to cfg.name when computing the default output path")
    ap.add_argument("--no-particles-h5", action="store_true",
                    help="skip writing particles.h5 (saves ~10-30 GB and a few "
                         "minutes per run)")
    args = ap.parse_args()

    cfg = load(args.config)
    if args.Np is not None:
        cfg.pic.Np = args.Np
    if args.name_suffix:
        cfg.name = f"{cfg.name}{args.name_suffix}"
    rng = np.random.default_rng(cfg.seed)

    L = cfg.domain.L
    k = cfg.domain.k
    Nx = cfg.pic.Nx
    Np = cfg.pic.Np
    dt = cfg.pic.dt
    n_steps = cfg.pic.n_steps

    x, v, w = _sample(cfg.truth_ic, Np, L, k, rng)
    state = make_state(x, v, w, L, Nx, dt)

    phi_hist = np.zeros((n_steps + 1, Nx))
    rho_hist = np.zeros((n_steps + 1, Nx))
    E_hist = np.zeros((n_steps + 1, Nx))
    j_hist = np.zeros((n_steps + 1, Nx))    # 1st velocity moment, for ∂_t ρ via continuity
    M_hist = np.zeros((n_steps + 1, Nx))    # 2nd velocity moment, for ∂_t² ρ via continuity
    t_hist = np.zeros(n_steps + 1)

    phi_hist[0] = state.phi
    E_hist[0] = state.E
    rho_hist[0] = cic_deposit(state.x, state.w, L, Nx)
    j_hist[0] = cic_deposit_current(state.x, state.v, state.w, L, Nx)
    M_hist[0] = cic_deposit_kinetic_stress(state.x, state.v, state.w, L, Nx)
    t_hist[0] = 0.0

    # Open particles.h5 in resumable per-step write mode if requested.
    out_path = Path(args.output) if args.output else Path(cfg.outputs_dir) / cfg.name / "truth.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path = out_path.parent / "particles.h5"
    h5f = None
    if not args.no_particles_h5:
        h5f = h5py.File(h5_path, "w")
        # Chunk per timestep along axis 0 so a single time-slice load is one
        # contiguous read. No compression — gzip-4 per step would add ~1 hour
        # to a Np=1e6 run; the user has the space.
        chunks = (1, Np)
        h5f.create_dataset("x", (n_steps + 1, Np), dtype=np.float64,
                           chunks=chunks)
        h5f.create_dataset("v", (n_steps + 1, Np), dtype=np.float64,
                           chunks=chunks)
        # w is constant in the collisionless backend; one entry suffices.
        h5f.create_dataset("w", data=state.w)
        h5f.create_dataset("t", (n_steps + 1,), dtype=np.float64)
        h5f.attrs["L"] = L
        h5f.attrs["Nx"] = Nx
        h5f.attrs["Np"] = Np
        h5f.attrs["dt"] = dt
        h5f.attrs["n_steps"] = n_steps
        h5f.attrs["seed"] = cfg.seed
        h5f["x"][0] = state.x
        h5f["v"][0] = state.v
        h5f["t"][0] = 0.0

    for n in range(n_steps):
        push_leapfrog_half(state, 0.5)
        push_leapfrog_drift(state)
        field_solve(state)
        push_leapfrog_half(state, 0.5)
        state.t += dt
        phi_hist[n + 1] = state.phi
        E_hist[n + 1] = state.E
        rho_hist[n + 1] = cic_deposit(state.x, state.w, L, Nx)
        j_hist[n + 1] = cic_deposit_current(state.x, state.v, state.w, L, Nx)
        M_hist[n + 1] = cic_deposit_kinetic_stress(state.x, state.v, state.w, L, Nx)
        t_hist[n + 1] = state.t
        if h5f is not None:
            h5f["x"][n + 1] = state.x
            h5f["v"][n + 1] = state.v
            h5f["t"][n + 1] = state.t

    if h5f is not None:
        h5f.close()
        print(f"wrote {h5_path}")

    np.savez_compressed(
        out_path,
        t=t_hist, phi=phi_hist, rho=rho_hist, E=E_hist,
        j=j_hist, M=M_hist,
        x_final=state.x, v_final=state.v, w_final=state.w,
    )
    manifest = {
        "kind": "truth",
        "config": cfg.model_dump(),
        "particles_h5": (str(h5_path) if h5f is not None else None),
    }
    (out_path.parent / "truth_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
