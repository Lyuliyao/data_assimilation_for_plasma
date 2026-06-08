"""Sweep the nudging strength gamma for a single (config, variant) pair.

Runs the assim loop once per gamma and writes a single CSV row per gamma
with the final-time diagnostics. Written as a thin wrapper around
mfda.assimilation.run so it stays compatible with any backend.

Usage:
    python scripts/sweep_gamma.py \
        --config configs/test0_identifiability_stable.yaml \
        --variant velocity \
        --gammas 0.0 0.25 0.5 1.0 2.0 4.0

Output (defaults):
    results/<cfg.name>/gamma_sweep_<variant>.csv
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
    ap.add_argument("--variant", choices=["velocity", "position", "none"],
                    required=True)
    ap.add_argument("--gammas", nargs="+", type=float,
                    default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0],
                    help="PIC seeds to sweep (one row per seed x gamma)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = load(args.config)
    outdir = Path(cfg.outputs_dir) / cfg.name
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = (Path(args.output) if args.output
                else outdir / f"gamma_sweep_{args.variant}.csv")

    fieldnames = ["variant", "gamma", "seed", "e_phi_final", "e_rho_final",
                  "e_f_final", "energy_final", "mode1_final", "mode1_truth_final"]

    with out_path.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()

        for seed in args.seeds:
            for gamma in args.gammas:
                cfg_r = copy.deepcopy(cfg)
                cfg_r.seed = seed
                # gamma = 0 is equivalent to variant = none (no coupling), so
                # skip the nudge path entirely to save particle-gather cost.
                effective_variant = "none" if gamma == 0.0 else args.variant
                cfg_r.nudge.set_legacy_variant(effective_variant, gamma)
                print(f"[sweep] variant={args.variant} gamma={gamma} seed={seed}")
                out = run(cfg_r)
                modes_a = np.asarray(out.assim_log["modes"])
                modes_t = np.asarray(out.truth_log["modes"])
                def _m1(m):
                    return float(m[-1, 1]) if m.ndim == 2 and m.shape[1] > 1 else float("nan")
                wr.writerow({
                    "variant": args.variant,
                    "gamma": gamma,
                    "seed": seed,
                    "e_phi_final": out.assim_log["e_phi"][-1],
                    "e_rho_final": out.assim_log["e_rho"][-1],
                    "e_f_final": out.assim_log["e_f"][-1],
                    "energy_final": out.assim_log["energy"][-1],
                    "mode1_final": _m1(modes_a),
                    "mode1_truth_final": _m1(modes_t),
                })
                f.flush()

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
