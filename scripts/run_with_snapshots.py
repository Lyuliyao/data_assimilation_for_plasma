"""Single assim run that saves intermediate particle snapshots for visualisation.

Usage:
    python scripts/run_with_snapshots.py \
        --config configs/test0_identifiability_linear_combined5_best.yaml \
        --n-snaps 6

Output:
    results/<cfg.name>/snapshots.npz    — particle (x, v, w) at the chosen
                                          step indices, both truth and assim.
    results/<cfg.name>/sweep_row.csv    — single-row metrics CSV.

Then plot via:
    python scripts/plot_from_snapshots.py --config <same yaml>
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mfda.assimilation import run
from mfda.config import load


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-snaps", type=int, default=6,
                    help="Number of evenly spaced snapshot steps including t=0.")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load(args.config)
    if args.seed is not None:
        cfg.seed = args.seed

    n_steps = cfg.pic.n_steps
    snap_steps = sorted({
        int(round(i * n_steps / (args.n_snaps - 1)))
        for i in range(args.n_snaps)
    })
    print(f"[run+snaps] snapshot steps: {snap_steps}")

    out = run(cfg, snapshot_steps=snap_steps)

    outdir = Path(cfg.outputs_dir) / cfg.name
    outdir.mkdir(parents=True, exist_ok=True)

    # Pack snapshots into a flat npz. Keys: truth_<step>_{x,v,w,phi,t} and
    # assim_<step>_{x,v,w,phi,t}.
    npz_payload: dict[str, np.ndarray] = {}
    npz_payload["snap_steps"] = np.asarray(snap_steps, dtype=np.int64)
    for step, snap in (out.snapshots_truth or {}).items():
        for key, arr in snap.items():
            npz_payload[f"truth_{step}_{key}"] = np.asarray(arr)
    for step, snap in (out.snapshots_assim or {}).items():
        for key, arr in snap.items():
            npz_payload[f"assim_{step}_{key}"] = np.asarray(arr)
    # Also persist the full per-diagnostic-step time series for both
    # truth and assim (e_phi, e_rho, e_f, energy, M_total, e_j, e_M,
    # e_var_v, etc.). This is what makes M_total(t) plotting possible
    # without re-simulating.
    npz_payload["t"] = np.asarray(out.t)
    for k, arr in out.assim_log.items():
        if isinstance(arr, np.ndarray):
            npz_payload[f"assim_log_{k}"] = arr
    for k, arr in out.assim_log.get("extras", {}).items():
        npz_payload[f"assim_log_extra_{k}"] = np.asarray(arr)
    for k, arr in out.truth_log.items():
        if isinstance(arr, np.ndarray):
            npz_payload[f"truth_log_{k}"] = arr
    for k, arr in out.truth_log.get("extras", {}).items():
        npz_payload[f"truth_log_extra_{k}"] = np.asarray(arr)
    np.savez_compressed(outdir / "snapshots.npz", **npz_payload)
    print(f"[run+snaps] wrote {outdir / 'snapshots.npz'}")

    # One-row metrics CSV.
    modes_a = np.asarray(out.assim_log["modes"])
    modes_t = np.asarray(out.truth_log["modes"])

    def _m1(m: np.ndarray) -> float:
        return float(m[-1, 1]) if m.ndim == 2 and m.shape[1] > 1 else float("nan")

    extras = out.assim_log.get("extras", {})

    def _last(key: str) -> float:
        if key in extras and len(extras[key]) > 0:
            return float(np.asarray(extras[key])[-1])
        return float("nan")

    row = {
        "config": cfg.name,
        "seed": cfg.seed,
        "e_phi_final": float(out.assim_log["e_phi"][-1]),
        "e_rho_final": float(out.assim_log["e_rho"][-1]),
        "e_f_final": float(out.assim_log["e_f"][-1]),
        "energy_final": float(out.assim_log["energy"][-1]),
        "mode1_final": _m1(modes_a),
        "mode1_truth_final": _m1(modes_t),
        "e_j_final": _last("e_j"),
        "e_M_final": _last("e_M"),
        "e_var_v_final": _last("e_var_v"),
    }
    with (outdir / "sweep_row.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(row.keys()))
        wr.writeheader()
        wr.writerow(row)
    (outdir / "manifest.json").write_text(json.dumps(out.config_snapshot, indent=2))
    print(f"[run+snaps] wrote {outdir / 'sweep_row.csv'}")


if __name__ == "__main__":
    main()
