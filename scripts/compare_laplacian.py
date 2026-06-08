"""Forward (no inverse Poisson) comparison of:

  (1)  Δ ∂_t φ        vs    ∂_x j
  (2)  Δ ∂_t² φ       vs    -∂_x² M - ∂_x(ρ E)

Math (1D, Poisson −∂_x² φ = ρ − 1):

  ∂_t (−Δφ) = ∂_t ρ = −∂_x j           [continuity]
  ⇒  −Δ ∂_t φ = −∂_x j   ⇒   Δ ∂_t φ = ∂_x j

  ∂_t² (−Δφ) = ∂_t² ρ
            = −∂_x ∂_t j
            = −∂_x (−∂_x M − ρ E)       [1D momentum: ∂_t j = −∂_x M − ρ E]
            = ∂_x² M + ∂_x(ρ E)
  ⇒  −Δ ∂_t² φ = ∂_x² M + ∂_x(ρ E)
  ⇒   Δ ∂_t² φ = −∂_x² M − ∂_x(ρ E)

So perfect agreement of the smoothed obs derivatives with the moment
source means:

  LHS_1 := Δ z_smooth(t, x)                =  ∂_x j(t, x)         =: RHS_1
  LHS_2 := Δ w_smooth(t, x)                = −Δ M(t, x) − ∂_x(ρ E)  =: RHS_2

Reads:
  results/<truth-dir>/truth.npz                — phi, rho, E, j, M
  results/<truth-dir>/observation_smoothed.npz  — z, w (smoothed)

Writes:
  results/<truth-dir>/laplacian_compare.png      — heatmap LHS vs RHS
  results/<truth-dir>/laplacian_compare_atx.png  — fixed-x time series

Usage:
  python scripts/compare_laplacian.py --truth-dir results/position_mismatch
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mfda.filtering import lowpass_filter
from mfda.poisson import grad_1d


def _laplacian_per_time(arr_tx: np.ndarray, L: float) -> np.ndarray:
    """Compute d²/dx² of arr(t, x) per time slice via two spectral grads."""
    return np.array([grad_1d(grad_1d(row, L), L) for row in arr_tx])


def _grad_per_time(arr_tx: np.ndarray, L: float) -> np.ndarray:
    """d/dx per time slice."""
    return np.array([grad_1d(row, L) for row in arr_tx])


def _spatial_lowpass_per_time(arr_tx: np.ndarray, L: float, k_cut_frac: float) -> np.ndarray:
    """Apply Hou-Li spatial lowpass to each time slice."""
    return np.array([lowpass_filter(row, L, k_cut_frac=k_cut_frac) for row in arr_tx])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth-dir", required=True)
    args = ap.parse_args()

    truth_dir = Path(args.truth_dir)
    truth = np.load(truth_dir / "truth.npz")
    obs = np.load(truth_dir / "observation_smoothed.npz")

    if "j" not in truth.files or "M" not in truth.files:
        raise SystemExit(
            f"truth.npz at {truth_dir} missing 'j' / 'M'. Re-run "
            f"scripts/run_truth.py with the latest version."
        )

    t = obs["t"]
    z = np.asarray(obs["z"])              # smoothed ∂_t φ
    w = np.asarray(obs["w"])              # smoothed ∂_t² φ
    rho = np.asarray(truth["rho"])
    E = np.asarray(truth["E"])
    j = np.asarray(truth["j"])
    M = np.asarray(truth["M"])
    Nt, Nx = z.shape

    # Recover L from the manifest.
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
    print(f"[lap] Nt={Nt}, Nx={Nx}, L={L:.4f}")

    # The moment fields j and M are CIC-deposited from ~1e6 particles
    # onto a 128-cell grid. Per-cell shot noise is ~1/sqrt(Np/Nx) ~ 1%.
    # Taking ∂_x of these amplifies high-k noise by ~Nx/L = ~10x; ∂_x²
    # amplifies by ~100x. Without spatial lowpass on the moments, the
    # RHS would be totally noise-dominated even though the LHS (Δ of
    # already-smoothed observation derivatives) is smooth.
    #
    # Apply the same spatial lowpass that the observation pipeline uses
    # on z and w BEFORE taking spatial derivatives of the moments.
    K_CUT_Z = 0.20      # matches combined_heavy lowpass on z
    K_CUT_W = 0.10      # matches combined_heavy lowpass on w
    j_s = _spatial_lowpass_per_time(j, L, K_CUT_Z)
    M_s = _spatial_lowpass_per_time(M, L, K_CUT_W)
    rhoE_s = _spatial_lowpass_per_time(rho * E, L, K_CUT_W)

    # LHS_1: Laplacian of smoothed z = ∂_t φ
    LHS_1 = _laplacian_per_time(z, L)
    # RHS_1: ∂_x j from saved moment (now spatially-lowpassed).
    RHS_1 = _grad_per_time(j_s, L)

    # LHS_2: Laplacian of smoothed w = ∂_t² φ
    LHS_2 = _laplacian_per_time(w, L)
    # RHS_2: -∂_x² M - ∂_x(ρ E)
    d2M = _laplacian_per_time(M_s, L)
    d_rhoE = _grad_per_time(rhoE_s, L)
    RHS_2 = -d2M - d_rhoE

    # ---- Heatmap figure: 2 rows (eq.1, eq.2) x 3 cols (LHS, RHS, diff) ----
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    extent = (0.0, t[-1], 0.0, L)

    for row, (lhs, rhs, lhs_label, rhs_label, eq_title) in enumerate([
        (LHS_1, RHS_1,
         r"$\Delta\,z = \Delta\,\partial_t\phi$  (smoothed obs)",
         r"$\partial_x j$  (truth moment)",
         r"continuity:  $\Delta\,\partial_t\phi = \partial_x j$"),
        (LHS_2, RHS_2,
         r"$\Delta\,w = \Delta\,\partial_t^2\phi$  (smoothed obs)",
         r"$-\Delta M - \partial_x(\rho E)$  (truth moments)",
         r"momentum:  $\Delta\,\partial_t^2\phi = -\Delta M - \partial_x(\rho E)$"),
    ]):
        # Color scale from RHS (the clean reference).
        vmax = float(np.percentile(np.abs(rhs), 99.5))
        if vmax <= 0:
            vmax = float(np.max(np.abs(rhs)))
        kwargs = dict(origin="lower", aspect="auto", extent=extent,
                      cmap="RdBu_r", vmin=-vmax, vmax=+vmax)
        im = axes[row, 0].imshow(lhs.T, **kwargs)
        axes[row, 0].set_title(lhs_label, fontsize=9)
        axes[row, 1].imshow(rhs.T, **kwargs)
        axes[row, 1].set_title(rhs_label, fontsize=9)
        # Difference at the same scale to make the residual visible.
        axes[row, 2].imshow((lhs - rhs).T, **kwargs)
        axes[row, 2].set_title(f"LHS − RHS   ({eq_title})", fontsize=9)
        cb = fig.colorbar(im, ax=axes[row, :], shrink=0.8, pad=0.02)
        cb.ax.tick_params(labelsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("t")
    for ax in axes[:, 0]:
        ax.set_ylabel("x")
    fig.suptitle(
        f"{truth_dir.name}: forward Laplacian comparison "
        f"(smoothed obs vs moment source)"
    )
    out_a = truth_dir / "laplacian_compare.png"
    fig.savefig(out_a, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[lap] wrote {out_a}")

    # ---- Time series at fixed x ----
    x_idx = Nx // 4
    fig2, axes2 = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    axes2[0].plot(t, RHS_1[:, x_idx], color="k", lw=1.4, label=r"$\partial_x j$  (truth)")
    axes2[0].plot(t, LHS_1[:, x_idx], color="C3", lw=1.0, alpha=0.8,
                  label=r"$\Delta\,z$  (smoothed obs)")
    axes2[0].set_ylabel(r"continuity:  $\Delta\,\partial_t\phi$")
    axes2[0].grid(alpha=0.3); axes2[0].legend(fontsize=9)
    axes2[1].plot(t, RHS_2[:, x_idx], color="k", lw=1.4,
                  label=r"$-\Delta M - \partial_x(\rho E)$  (truth)")
    axes2[1].plot(t, LHS_2[:, x_idx], color="C3", lw=1.0, alpha=0.8,
                  label=r"$\Delta\,w$  (smoothed obs)")
    axes2[1].set_ylabel(r"momentum:  $\Delta\,\partial_t^2\phi$")
    axes2[1].set_xlabel("t")
    axes2[1].grid(alpha=0.3); axes2[1].legend(fontsize=9)
    fig2.suptitle(f"{truth_dir.name}: forward Laplacian comparison at "
                  f"x_* = {x_idx*L/Nx:.2f}")
    fig2.tight_layout()
    out_b = truth_dir / "laplacian_compare_atx.png"
    fig2.savefig(out_b, dpi=140)
    plt.close(fig2)
    print(f"[lap] wrote {out_b}")


if __name__ == "__main__":
    main()
