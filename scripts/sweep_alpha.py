"""Sweep the time-derivative observation weight alpha for one (config, variant).

Mirrors scripts/sweep_gamma.py but holds gamma fixed and varies
observation.time_derivative.alpha. Each call writes one CSV row per
(seed, alpha) with the final-time diagnostics.

Usage:
    python scripts/sweep_alpha.py \
        --config configs/test0_identifiability_linear_dtobs.yaml \
        --variant velocity --gamma 1.0 \
        --alphas 0.0 0.1 0.3 1.0 3.0 10.0 \
        --seeds 0

Output (default):
    results/<cfg.name>/alpha_sweep_<variant>.csv
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
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.0, 0.1, 0.3, 1.0, 3.0, 10.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0],
                    help="PIC seeds to sweep (one row per seed x alpha)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = load(args.config)
    outdir = Path(cfg.outputs_dir) / cfg.name
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = (Path(args.output) if args.output
                else outdir / f"alpha_sweep_{args.variant}.csv")

    fieldnames = [
        "variant", "gamma", "alpha", "seed",
        "e_phi_final", "e_rho_final", "e_f_final", "energy_final",
        "mode1_final", "mode1_truth_final", "e_j_final",
    ]

    with out_path.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()

        for seed in args.seeds:
            for alpha in args.alphas:
                cfg_r = copy.deepcopy(cfg)
                cfg_r.seed = seed
                cfg_r.nudge.set_legacy_variant(args.variant, args.gamma)
                cfg_r.observation.time_derivative.alpha = alpha
                # alpha == 0 disables the augmentation entirely (matches
                # snapshot-only behaviour bit-exactly via the regression test).
                cfg_r.observation.time_derivative.enabled = (alpha != 0.0)
                # Mirror the load-time shim: when the augmented observation
                # is on and the variant is velocity, also enable velocity_dtobs
                # so the new channel-based dispatch picks it up.
                if args.variant == "velocity" and alpha != 0.0:
                    from mfda.config import ChannelCfg
                    cfg_r.nudge.velocity_dtobs = ChannelCfg(
                        enabled=True, gamma=args.gamma, alpha=alpha,
                    )
                print(f"[sweep] variant={args.variant} gamma={args.gamma} "
                      f"alpha={alpha} seed={seed}")
                out = run(cfg_r)
                modes_a = np.asarray(out.assim_log["modes"])
                modes_t = np.asarray(out.truth_log["modes"])
                def _m1(m: np.ndarray) -> float:
                    return float(m[-1, 1]) if m.ndim == 2 and m.shape[1] > 1 else float("nan")
                e_j_final = float("nan")
                extras = out.assim_log.get("extras", {})
                if "e_j" in extras and len(extras["e_j"]) > 0:
                    e_j_final = float(np.asarray(extras["e_j"])[-1])
                wr.writerow({
                    "variant": args.variant,
                    "gamma": args.gamma,
                    "alpha": alpha,
                    "seed": seed,
                    "e_phi_final": out.assim_log["e_phi"][-1],
                    "e_rho_final": out.assim_log["e_rho"][-1],
                    "e_f_final": out.assim_log["e_f"][-1],
                    "energy_final": out.assim_log["energy"][-1],
                    "mode1_final": _m1(modes_a),
                    "mode1_truth_final": _m1(modes_t),
                    "e_j_final": e_j_final,
                })
                f.flush()

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
