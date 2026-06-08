"""Sweep position_d2tobs.alpha (= beta) only, all other channels fixed.

The bilinear-quadratic position_d2tobs term contains the third spatial
derivative of psi_2 and a U^2 prefactor. At beta = 1.0 the assim run NaN'd
on every config (linear5, strong5, asymm5). This sweep walks beta down the
log axis to find the largest beta that keeps the run finite.

Other channels (ps, pd, vd, v2d) are kept at whatever the config has — only
position_d2tobs.alpha is varied per row.

Usage:
    python scripts/sweep_p2d_beta.py \
        --config configs/test0_identifiability_linear_combined5.yaml \
        --betas 1e-5 1e-4 1e-3 1e-2 1e-1 \
        --seeds 0

Output (default):
    results/<cfg.name>/p2d_beta_sweep.csv
"""
from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import numpy as np

from mfda.assimilation import run
from mfda.config import load


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--betas", nargs="+", type=float,
                    default=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    ap.add_argument("--nps", nargs="+", type=int, default=None,
                    help="Override pic.Np per row. If omitted, uses cfg.pic.Np.")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = load(args.config)
    outdir = Path(cfg.outputs_dir) / cfg.name
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = (Path(args.output) if args.output
                else outdir / "p2d_beta_sweep.csv")
    nps = args.nps if args.nps is not None else [cfg.pic.Np]

    fieldnames = [
        "Np", "p2d_beta", "seed",
        "e_phi_final", "e_rho_final", "e_f_final", "energy_final",
        "mode1_final", "mode1_truth_final", "e_j_final", "stable",
    ]

    with out_path.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()

        for seed in args.seeds:
            for Np in nps:
                for beta in args.betas:
                    cfg_r = copy.deepcopy(cfg)
                    cfg_r.seed = seed
                    cfg_r.pic.Np = Np
                    cfg_r.nudge.position_d2tobs.alpha = beta
                    print(f"[p2d_beta] Np={Np} beta={beta} seed={seed}")
                    out = run(cfg_r)
                    modes_a = np.asarray(out.assim_log["modes"])
                    modes_t = np.asarray(out.truth_log["modes"])

                    def _m1(m: np.ndarray) -> float:
                        return float(m[-1, 1]) if m.ndim == 2 and m.shape[1] > 1 else float("nan")

                    e_phi = float(out.assim_log["e_phi"][-1])
                    stable = int(np.isfinite(e_phi))
                    e_j_final = float("nan")
                    extras = out.assim_log.get("extras", {})
                    if "e_j" in extras and len(extras["e_j"]) > 0:
                        e_j_final = float(np.asarray(extras["e_j"])[-1])
                    wr.writerow({
                        "Np": Np,
                        "p2d_beta": beta,
                        "seed": seed,
                        "e_phi_final": e_phi,
                        "e_rho_final": out.assim_log["e_rho"][-1],
                        "e_f_final": out.assim_log["e_f"][-1],
                        "energy_final": out.assim_log["energy"][-1],
                        "mode1_final": _m1(modes_a),
                        "mode1_truth_final": _m1(modes_t),
                        "e_j_final": e_j_final,
                        "stable": stable,
                    })
                    f.flush()

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
