"""Compare smoothed observation derivatives (z, w from filtering phi(t,x))
against the *exact* continuity-based derivatives reconstructed from the
truth's first two velocity moments (j, M) and electric field E.

Math (from docs/algorithm_implementation.md §2):

    z_continuity(t, x) = ∂_t phi  via  -Δ(∂_t phi) = -∂_x j
                       = solve_poisson_from_div(j(t, x), L)

    w_continuity(t, x) = ∂_t² phi via  -Δ(∂_t² phi) = ∂_x² M - ∂_x(ρ E)
                       = solve_poisson_from_d2(M(t, x), ρ(t, x), E(t, x), L)

These are NOISE-FREE references (modulo CIC shape-function smoothing on
the deposits) for what the smoothed observation derivatives should
look like.

Reads:
    results/<truth-dir>/truth.npz                — has phi, rho, E, j, M
    results/<truth-dir>/smoothed_observations.npz — has phi_s/z/w per method

Writes:
    results/<truth-dir>/zw_continuity_vs_obs.png — 2x4 grid:
        rows: z, w
        cols: continuity reference / SG_w21 / SG_w41 / combined_heavy
        each panel: heatmap of (t, x)
    results/<truth-dir>/zw_continuity_vs_obs_atx.png — same data, fixed-x time
        series so amplitude is visible.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mfda.poisson import solve_poisson_from_div, solve_poisson_from_d2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth-dir", required=True)
    args = ap.parse_args()

    truth_dir = Path(args.truth_dir)
    truth = np.load(truth_dir / "truth.npz")
    smooth = np.load(truth_dir / "smoothed_observations.npz")

    if "j" not in truth.files or "M" not in truth.files:
        raise SystemExit(
            f"truth.npz at {truth_dir} does not contain 'j' and 'M'. "
            f"Re-run scripts/run_truth.py with the updated version that "
            f"saves the first two velocity moments."
        )

    t = truth["t"]
    phi = truth["phi"]
    rho = truth["rho"]
    E = truth["E"]
    j = truth["j"]
    M = truth["M"]
    Nt, Nx = phi.shape

    # Recover L from manifest.
    mp = truth_dir / "truth_manifest.json"
    L = 4.0 * math.pi
    if mp.exists():
        m = json.loads(mp.read_text())
        cfg = m.get("config", {})
        dom = cfg.get("domain", {})
        if dom.get("L") is not None:
            L = float(dom["L"])
        elif dom.get("k") is not None:
            L = 2.0 * math.pi / float(dom["k"])

    print(f"[compare] reconstructing z, w via continuity, Nt={Nt}, Nx={Nx}, L={L:.4f}")

    # Continuity-based derivatives — one Poisson solve per time step.
    z_cont = np.zeros_like(phi)
    w_cont = np.zeros_like(phi)
    for n in range(Nt):
        z_cont[n] = solve_poisson_from_div(j[n], L)
        w_cont[n] = solve_poisson_from_d2(M[n], rho[n], E[n], L)

    # Smoothed-observation candidates.
    candidates = ("temporal_sg_w21", "temporal_sg_w41", "temporal_sg_w81", "combined_heavy")

    # ---- Heatmap figure: 2 rows (z, w) x N cols ----
    n_cols = 1 + len(candidates)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 6.5),
                             sharex=True, sharey=True)
    extent = (0.0, t[-1], 0.0, L)
    for row, (key, ref, title) in enumerate([
        ("z", z_cont, r"$z = \partial_t \phi$"),
        ("w", w_cont, r"$w = \partial_t^2 \phi$"),
    ]):
        # Color scale from the continuity reference (clean signal).
        vmax = float(np.percentile(np.abs(ref), 99.5))
        if vmax <= 0:
            vmax = float(np.max(np.abs(ref)))
        kwargs = dict(origin="lower", aspect="auto", extent=extent,
                      cmap="RdBu_r", vmin=-vmax, vmax=+vmax)
        # Column 0: continuity reference.
        im = axes[row, 0].imshow(ref.T, **kwargs)
        axes[row, 0].set_title(f"{title}  continuity (truth ref)")
        # Columns 1..: candidate smoothing methods.
        for col, name in enumerate(candidates, start=1):
            arr = smooth[f"{name}_{key}"]
            axes[row, col].imshow(arr.T, **kwargs)
            axes[row, col].set_title(f"{title}  {name}")
        cb = fig.colorbar(im, ax=axes[row, :], shrink=0.8, pad=0.02)
        cb.ax.tick_params(labelsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("t")
    for ax in axes[:, 0]:
        ax.set_ylabel("x")
    fig.suptitle(f"{truth_dir.name}: smoothed-obs derivatives vs continuity reference")
    out_a = truth_dir / "zw_continuity_vs_obs.png"
    fig.savefig(out_a, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {out_a}")

    # ---- Time series at fixed x ----
    x_idx = Nx // 4
    fig2, axes2 = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax, key, ref, title in [
        (axes2[0], "z", z_cont, r"$z = \partial_t \phi$  at $x_*$"),
        (axes2[1], "w", w_cont, r"$w = \partial_t^2 \phi$  at $x_*$"),
    ]:
        ax.plot(t, ref[:, x_idx], color="k", lw=1.6, label="continuity (truth ref)")
        for c, name in zip(("C1", "C2", "C3", "C4"), candidates):
            ax.plot(t, smooth[f"{name}_{key}"][:, x_idx],
                    color=c, lw=1.0, alpha=0.8, label=name)
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    axes2[-1].set_xlabel("t")
    fig2.suptitle(f"{truth_dir.name}: same comparison at x_* = {x_idx*L/Nx:.2f}")
    fig2.tight_layout()
    out_b = truth_dir / "zw_continuity_vs_obs_atx.png"
    fig2.savefig(out_b, dpi=140)
    plt.close(fig2)
    print(f"[compare] wrote {out_b}")


if __name__ == "__main__":
    main()
