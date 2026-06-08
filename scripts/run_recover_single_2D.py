"""2D2V recover-check: run ONE formulation and save its npz.

Designed for sbatch array fan-out: each array task picks a formulation
(via --formulation or via $SLURM_ARRAY_TASK_ID) and writes
`results/<cfg.name>/<formulation>.npz`. The aggregation (plots + SUMMARY)
is a separate script -- see aggregate_recover_check_2D.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mfda.assimilation_moments_2d import run_moments_2d
from mfda.config_2d import load_moment_2d


FORMULATIONS = ["none", "aot", "A", "B", "C"]


def resolve_out_dir(cfg, cfg_path: Path) -> Path:
    project_root = Path(__file__).resolve().parent.parent
    out_root = Path(cfg.outputs_dir)
    if not out_root.is_absolute():
        out_root = (project_root / out_root).resolve()
    return out_root / cfg.name


def run_one(cfg_path: Path, formulation: str) -> Path:
    """Run a single formulation, save `<formulation>.npz`, return its path."""
    if formulation not in FORMULATIONS:
        raise ValueError(f"--formulation must be one of {FORMULATIONS}, got {formulation!r}")
    cfg = load_moment_2d(cfg_path)
    cfg.moment_nudge.formulation = formulation  # type: ignore[assignment]

    out_dir = resolve_out_dir(cfg, cfg_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[recover-single-2D] outputs: {out_dir}", flush=True)
    print(f"[recover-single-2D] running formulation={formulation} ...", flush=True)

    from mfda.backend_reference_2d import cic_deposit_2d
    from mfda.poisson_2d import potential_from_density_2d
    Lx, Ly = cfg.Lx, cfg.Ly
    Nx, Ny = cfg.pic.Nx, cfg.pic.Ny

    out = run_moments_2d(cfg)
    snap_truth = out.truth_log["snapshots"]
    snap_assim = out.assim_log["snapshots"]
    snap_steps = sorted(snap_assim.keys())
    snap_t = np.asarray(snap_steps, dtype=float) * cfg.pic.dt
    rho_truth_snaps = np.stack([
        cic_deposit_2d(s["x"], s["y"], s["w"], Lx, Ly, Nx, Ny)
        for s in [snap_truth[k] for k in snap_steps]
    ])
    rho_assim_snaps = np.stack([
        cic_deposit_2d(s["x"], s["y"], s["w"], Lx, Ly, Nx, Ny)
        for s in [snap_assim[k] for k in snap_steps]
    ])
    phi_truth_snaps = np.stack([
        potential_from_density_2d(r, Lx, Ly) for r in rho_truth_snaps
    ])
    phi_assim_snaps = np.stack([
        potential_from_density_2d(r, Lx, Ly) for r in rho_assim_snaps
    ])
    save = dict(
        t=out.t,
        assim_log_e_phi=out.assim_log["e_phi"],
        assim_log_e_rho=out.assim_log["e_rho"],
        assim_log_e_u=out.assim_log["e_u"],
        assim_log_e_T=out.assim_log["e_T"],
        assim_log_energy=out.assim_log["energy"],
        truth_log_energy=out.truth_log["energy"],
        truth_phi_hat=out.truth_log["phi_hat"],
        truth_rho_hat=out.truth_log["rho_hat"],
        truth_jlong_hat=out.truth_log["jlong_hat"],
        truth_T_hat=out.truth_log["T_hat"],
        assim_phi_hat=out.assim_log["phi_hat"],
        assim_rho_hat=out.assim_log["rho_hat"],
        assim_jlong_hat=out.assim_log["jlong_hat"],
        assim_T_hat=out.assim_log["T_hat"],
        driver_mode=out.truth_log["driver_mode"],
        r0_norm=out.assim_log["r0_norm"],
        r1_norm=out.assim_log["r1_norm"],
        r2_norm=out.assim_log["r2_norm"],
        t_nudge=out.assim_log["t_nudge"],
        bx_rms=out.assim_log["bx_rms"],
        by_rms=out.assim_log["by_rms"],
        bvx_rms=out.assim_log["bvx_rms"],
        bvy_rms=out.assim_log["bvy_rms"],
        snap_t=snap_t,
        rho_truth_snaps=rho_truth_snaps,
        rho_assim_snaps=rho_assim_snaps,
        phi_truth_snaps=phi_truth_snaps,
        phi_assim_snaps=phi_assim_snaps,
    )
    out_path = out_dir / f"{formulation}.npz"
    np.savez(out_path, **save)
    print(f"[recover-single-2D]   wrote {out_path}", flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to YAML")
    ap.add_argument(
        "--formulation",
        choices=FORMULATIONS,
        help="which formulation to run; if absent, taken from $SLURM_ARRAY_TASK_ID "
             "(0->none, 1->aot, 2->A, 3->B, 4->C)",
    )
    args = ap.parse_args()
    formulation = args.formulation
    if formulation is None:
        import os
        tid = os.environ.get("SLURM_ARRAY_TASK_ID")
        if tid is None:
            ap.error("--formulation is required unless $SLURM_ARRAY_TASK_ID is set")
        try:
            formulation = FORMULATIONS[int(tid)]
        except (ValueError, IndexError):
            ap.error(f"SLURM_ARRAY_TASK_ID={tid!r} does not index 0..4")
    run_one(Path(args.config), formulation)


if __name__ == "__main__":
    main()
