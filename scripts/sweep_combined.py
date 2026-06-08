"""Sweep the combined nudge over (gamma, alpha) at fixed channel assignment.

Mirrors scripts/sweep_alpha.py but mutates the gamma and alpha of all
*enabled* channels simultaneously (the doc's combined default has
position_snapshot + position_dtobs + velocity_dtobs share both knobs).

Usage:
    python scripts/sweep_combined.py \
        --config configs/test0_identifiability_strong_combined.yaml \
        --gammas 1.0 \
        --alphas 1.0 \
        --seeds 0

Output (default):
    results/<cfg.name>/combined_sweep.csv
"""
from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import numpy as np

from mfda.assimilation import run
from mfda.config import load


def _enabled_channels(nudge) -> str:
    """Canonical short string for the enabled channels (e.g. 'ps+pd+vd')."""
    short = {
        "position_snapshot": "ps",
        "velocity_snapshot": "vs",
        "position_dtobs": "pd",
        "velocity_dtobs": "vd",
        "position_d2tobs": "p2d",
        "velocity_d2tobs": "v2d",
    }
    parts: list[str] = []
    for name, key in short.items():
        if getattr(nudge, name).enabled:
            parts.append(key)
    return "+".join(parts) if parts else "none"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gammas", nargs="+", type=float,
                    default=[1.0])
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[1.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = load(args.config)
    outdir = Path(cfg.outputs_dir) / cfg.name
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = (Path(args.output) if args.output
                else outdir / "combined_sweep.csv")

    channels = _enabled_channels(cfg.nudge)

    fieldnames = [
        "channels", "gamma", "alpha", "seed",
        "e_phi_final", "e_rho_final", "e_f_final", "energy_final",
        "mode1_final", "mode1_truth_final", "e_j_final",
    ]

    with out_path.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()

        for seed in args.seeds:
            for gamma in args.gammas:
                for alpha in args.alphas:
                    cfg_r = copy.deepcopy(cfg)
                    cfg_r.seed = seed
                    # Apply (gamma, alpha) to every enabled channel.
                    for ch_name in (
                        "position_snapshot", "velocity_snapshot",
                        "position_dtobs", "velocity_dtobs",
                        "position_d2tobs", "velocity_d2tobs",
                    ):
                        ch = getattr(cfg_r.nudge, ch_name)
                        if ch.enabled:
                            ch.gamma = gamma
                            ch.alpha = alpha
                    # The augmented observation block carries its own alpha
                    # for the lowpass; mirror it for consistency.
                    cfg_r.observation.time_derivative.alpha = alpha
                    print(f"[combined] channels={channels} gamma={gamma} "
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
                        "channels": channels,
                        "gamma": gamma,
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
