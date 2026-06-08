"""Focused 3x2 figure showing the three observation channels (phi, z, w)
with raw finite differencing vs the best causal filter (Savitzky-Golay
backward, window=21).

Reads results/<truth-dir>/smoothed_observations.npz produced by
scripts/smoothing_study.py.

Usage:
    python scripts/plot_obs_with_filter.py --truth-dir results/landau_langevin
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth-dir", required=True)
    ap.add_argument("--filter-method", default="temporal_sg_w21",
                    choices=["spatial", "temporal_ema_a0p3", "temporal_ema_a0p1",
                             "temporal_sg_w11", "temporal_sg_w21", "combined"])
    ap.add_argument("--x-frac", type=float, default=0.25,
                    help="Position to sample (fraction of L). Default 0.25.")
    args = ap.parse_args()

    truth_dir = Path(args.truth_dir)
    data = np.load(truth_dir / "smoothed_observations.npz")
    t = data["t"]
    Nx = data["raw_phi"].shape[1]
    x_idx = int(round(args.x_frac * Nx))

    # raw and filtered traces for phi, z, w at the chosen x
    raw = {k: data[f"raw_{k}"][:, x_idx] for k in ("phi", "z", "w")}
    flt = {k: data[f"{args.filter_method}_{k}"][:, x_idx] for k in ("phi", "z", "w")}

    fig, axes = plt.subplots(3, 2, figsize=(11, 7), sharex=True)
    titles = (
        (r"$\phi(t, x_*)$  raw",        r"$\phi(t, x_*)$  filtered"),
        (r"$z = \partial_t \phi$  raw",   r"$z = \partial_t \phi$  filtered"),
        (r"$w = \partial_t^2 \phi$  raw", r"$w = \partial_t^2 \phi$  filtered"),
    )
    colors = ("C0", "C1", "C3")
    for i, key in enumerate(("phi", "z", "w")):
        axes[i, 0].plot(t, raw[key], color=colors[i], lw=0.7)
        axes[i, 0].set_title(titles[i][0])
        axes[i, 1].plot(t, flt[key], color=colors[i], lw=0.9)
        axes[i, 1].set_title(titles[i][1])
        for ax in axes[i, :]:
            ax.grid(alpha=0.3)
        # share y-range across raw/filtered so the noise vs signal contrast
        # is visible
        ymin = min(axes[i, 0].get_ylim()[0], axes[i, 1].get_ylim()[0])
        ymax = max(axes[i, 0].get_ylim()[1], axes[i, 1].get_ylim()[1])
        for ax in axes[i, :]:
            ax.set_ylim(ymin, ymax)
    for ax in axes[-1, :]:
        ax.set_xlabel("t")

    fig.suptitle(
        f"{truth_dir.name}: observation y(t) and its time derivatives  "
        f"(filter = {args.filter_method},  x_* = {args.x_frac:.2f} L)"
    )
    fig.tight_layout()
    out_path = truth_dir / "obs_filter_panels.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[plot-obs-filter] wrote {out_path}")


if __name__ == "__main__":
    main()
