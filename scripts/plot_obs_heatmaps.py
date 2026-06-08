"""3x2 heatmaps of phi(t,x), z(t,x), w(t,x) — raw vs filtered.

Shows the full spatial-temporal field, so the x-dependence is visible
unlike the fixed-x time series in plot_obs_with_filter.py.

Reads results/<truth-dir>/smoothed_observations.npz produced by
scripts/smoothing_study.py.

Usage:
    python scripts/plot_obs_heatmaps.py --truth-dir results/landau_langevin
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth-dir", required=True)
    ap.add_argument("--filter-method", default="temporal_sg_w21")
    args = ap.parse_args()

    truth_dir = Path(args.truth_dir)
    data = np.load(truth_dir / "smoothed_observations.npz")
    t = data["t"]
    Nt, Nx = data["raw_phi"].shape
    # Recover L from manifest (default 2pi/k)
    import json, math
    L = 4.0 * math.pi
    mp = truth_dir / "truth_manifest.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        L = float(m["config"]["domain"]["L"])

    raw = {k: data[f"raw_{k}"] for k in ("phi", "z", "w")}
    flt = {k: data[f"{args.filter_method}_{k}"] for k in ("phi", "z", "w")}

    fig, axes = plt.subplots(3, 2, figsize=(11, 8.5), sharex=True, sharey=True)
    titles = (
        (r"$\phi(t, x)$  raw",            r"$\phi(t, x)$  filtered"),
        (r"$z(t, x) = \partial_t \phi$  raw",   r"$z(t, x) = \partial_t \phi$  filtered"),
        (r"$w(t, x) = \partial_t^2 \phi$  raw", r"$w(t, x) = \partial_t^2 \phi$  filtered"),
    )

    for i, key in enumerate(("phi", "z", "w")):
        # Use a symmetric color scale matched to filtered amplitude (so the
        # signal is visible in both columns; raw shows noise on the same
        # scale, which IS the point).
        vmax = float(np.percentile(np.abs(flt[key]), 99.5))
        if vmax <= 0:
            vmax = float(np.max(np.abs(flt[key])))
        kwargs = dict(
            origin="lower", aspect="auto",
            extent=(0.0, t[-1], 0.0, L),
            cmap="RdBu_r", vmin=-vmax, vmax=+vmax,
        )
        # arr is (Nt, Nx); imshow expects (row=y, col=x), so transpose to put
        # x on the y-axis and t on the x-axis.
        im0 = axes[i, 0].imshow(raw[key].T, **kwargs)
        axes[i, 0].set_title(titles[i][0])
        im1 = axes[i, 1].imshow(flt[key].T, **kwargs)
        axes[i, 1].set_title(titles[i][1])
        # one colorbar per row, attached to the right column
        cbar = fig.colorbar(im1, ax=axes[i, :], shrink=0.8, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("t")
    for ax in axes[:, 0]:
        ax.set_ylabel("x")
    fig.suptitle(
        f"{truth_dir.name}: full phi(t,x) field — raw vs {args.filter_method}"
    )
    out_path = truth_dir / "obs_filter_heatmaps.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot-obs-heatmaps] wrote {out_path}")


if __name__ == "__main__":
    main()
