"""Run the assimilated simulation (truth + assim in lock-step) and save diagnostics.

Usage:
    # Run a config as-is (combined channels enabled in the YAML):
    python scripts/run_assimilation.py \
        --config configs/test0_identifiability_strong_combined.yaml

    # Disable a single channel for ablations (repeatable):
    python scripts/run_assimilation.py \
        --config configs/test0_identifiability_strong_combined.yaml \
        --disable position_dtobs

    # Legacy CLI (still works via the config shim and set_legacy_variant):
    python scripts/run_assimilation.py \
        --config configs/test1_linear_landau.yaml \
        --nudge velocity --gamma 1.0

Output: results/<name>/assim_<canonical>.npz, where <canonical> is the
hyphen-joined enabled-channel codes (e.g. "ps+pd+vd"). The *_manifest.json
records the resolved config and the git SHA implicitly via the assim run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mfda.assimilation import run
from mfda.config import load


CHANNEL_CODES = {
    "position_snapshot": "ps",
    "velocity_snapshot": "vs",
    "position_dtobs": "pd",
    "velocity_dtobs": "vd",
    "position_d2tobs": "p2d",
    "velocity_d2tobs": "v2d",
}


def _canonical_channels(nudge) -> str:
    parts = [code for name, code in CHANNEL_CODES.items()
             if getattr(nudge, name).enabled]
    return "+".join(parts) if parts else "none"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--nudge", choices=["velocity", "position", "none"], default=None,
                    help="Legacy: shorthand for set_legacy_variant.")
    ap.add_argument("--gamma", type=float, default=None,
                    help="Legacy: gamma for the legacy variant.")
    ap.add_argument("--disable", action="append", default=[],
                    choices=list(CHANNEL_CODES.keys()),
                    help="Disable a channel by name (repeatable).")
    ap.add_argument("--obs-kind", choices=["full", "noisy", "coarse"], default=None)
    ap.add_argument("--obs-sigma", type=float, default=None)
    ap.add_argument("--obs-every-m", type=int, default=None)
    ap.add_argument("--obs-every-q", type=int, default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = load(args.config)

    # Legacy CLI overrides
    if args.nudge is not None:
        gamma = args.gamma if args.gamma is not None else 1.0
        cfg.nudge.set_legacy_variant(args.nudge, gamma)
    elif args.gamma is not None:
        # No --nudge but a --gamma: scale gamma on every enabled channel.
        for name in CHANNEL_CODES:
            ch = getattr(cfg.nudge, name)
            if ch.enabled:
                ch.gamma = args.gamma

    # Ablation disables
    for name in args.disable:
        getattr(cfg.nudge, name).enabled = False

    if args.obs_kind is not None:
        cfg.observation.kind = args.obs_kind
    if args.obs_sigma is not None:
        cfg.observation.sigma = args.obs_sigma
    if args.obs_every_m is not None:
        cfg.observation.every_m = args.obs_every_m
    if args.obs_every_q is not None:
        cfg.observation.every_q = args.obs_every_q

    out = run(cfg)

    canonical = _canonical_channels(cfg.nudge)
    out_path = (
        Path(args.output) if args.output
        else Path(cfg.outputs_dir) / cfg.name / f"assim_{canonical}.npz"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = dict(t=out.t)
    payload.update(
        {f"assim_{k}": v for k, v in out.assim_log.items() if not isinstance(v, list)}
    )
    payload.update(
        {f"truth_{k}": v for k, v in out.truth_log.items() if not isinstance(v, list)}
    )
    payload.update(dict(
        final_assim_x=out.final_assim["x"],
        final_assim_v=out.final_assim["v"],
        final_truth_x=out.final_truth["x"],
        final_truth_v=out.final_truth["v"],
    ))
    # Intermediate-step snapshots (default ON; pass [] in run() to disable).
    if out.snapshots_truth:
        payload["snap_steps"] = np.asarray(sorted(out.snapshots_truth.keys()),
                                           dtype=np.int64)
        for step, snap in out.snapshots_truth.items():
            for key, arr in snap.items():
                payload[f"truth_{step}_{key}"] = np.asarray(arr)
        for step, snap in out.snapshots_assim.items():
            for key, arr in snap.items():
                payload[f"assim_{step}_{key}"] = np.asarray(arr)
    np.savez_compressed(out_path, **payload)
    (out_path.parent / f"assim_{canonical}_manifest.json").write_text(
        json.dumps(out.config_snapshot, indent=2)
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
