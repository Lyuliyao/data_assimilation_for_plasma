"""Smooth a truth.npz's phi(t,x), compute z = d_t phi and w = d_t^2 phi
from the smoothed series, and save a self-contained observation_smoothed.npz
the assim side can later consume.

Output schema (in the same dir as truth.npz):
    observation_smoothed.npz
        t            : (Nt,)            time array
        phi          : (Nt, Nx)          smoothed observation y(t, x)
        z            : (Nt, Nx)          d_t phi (smoothed)
        w            : (Nt, Nx)          d_t^2 phi (smoothed)
        phi_raw      : (Nt, Nx)          unsmoothed phi for reference
        method, ...  : metadata

Default smoothing method: temporal Savitzky-Golay backward, window=21,
polyorder=2, identified empirically as the best causal smoother in
scripts/smoothing_study.py.

Usage:
    python scripts/save_smoothed_observation.py \
        --truth-dir results/position_mismatch \
        --method temporal_sg_w21
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Re-use the same smoothing primitives as the smoothing study so the
# saved observation is bit-equivalent to what's plotted there.
from smoothing_study import (   # type: ignore  (sibling-script import)
    apply_method, METHODS,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth-dir", required=True,
                    help="dir containing truth.npz produced by run_truth.py")
    ap.add_argument("--method", default="temporal_sg_w21",
                    choices=METHODS,
                    help="smoothing method (default: temporal_sg_w21)")
    args = ap.parse_args()

    truth_dir = Path(args.truth_dir)
    data = np.load(truth_dir / "truth.npz")
    phi = np.asarray(data["phi"])              # (Nt, Nx) raw observation
    t = np.asarray(data["t"])
    Nt, Nx = phi.shape
    dt = float(t[1] - t[0])

    # Recover L from manifest if available, else default 2*pi/k.
    import json, math
    L = 4.0 * math.pi
    mp = truth_dir / "truth_manifest.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        cfg = m.get("config", {})
        dom = cfg.get("domain", {})
        if dom.get("L") is not None:
            L = float(dom["L"])
        elif dom.get("k") is not None:
            L = 2.0 * math.pi / float(dom["k"])

    print(f"[obs-save] Nt={Nt} Nx={Nx} dt={dt} L={L:.4f} method={args.method}")
    phi_smooth, z, w = apply_method(phi, dt, L, args.method)

    out_path = truth_dir / "observation_smoothed.npz"
    np.savez_compressed(
        out_path,
        t=t,
        phi=phi_smooth,
        z=z,
        w=w,
        phi_raw=phi,
        method=np.array(args.method),
        # Not strictly needed but useful for downstream consumers.
        L=np.float64(L),
        dt=np.float64(dt),
    )
    print(f"[obs-save] wrote {out_path}")


# Make the smoothing_study script importable even when run from anywhere.
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
