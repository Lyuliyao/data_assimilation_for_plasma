"""Apply different temporal+spatial smoothing methods to truth phi(t, x)
and produce z = d/dt phi and w = d^2/dt^2 phi for each. Saves all
processed arrays to disk and produces side-by-side visualisations so we
can eyeball "is this derivative smooth enough".

Reads:  results/<truth_dir>/truth.npz  (must have phi[n_steps+1, Nx], t[n_steps+1])
Writes: results/<truth_dir>/smoothed_observations.npz
        results/<truth_dir>/smoothing_comparison.png

Methods tested:
  raw                  : direct finite difference of unsmoothed phi
  spatial              : current production approach — phi unsmoothed,
                         apply spatial lowpass to z and w post-difference
  temporal_ema_a0p3    : exponential moving average on phi (alpha=0.3)
                         before differentiation, then spatial lowpass
  temporal_ema_a0p1    : EMA alpha=0.1 (more aggressive temporal smooth)
  temporal_sg_w11      : Savitzky-Golay backward, window=11, polyorder=2
                         on phi before differentiation
  temporal_sg_w21      : SG backward, window=21
  combined             : EMA alpha=0.3 + spatial lowpass

Usage:
    python scripts/smoothing_study.py \
        --truth-dir results/test0_identifiability_strong__np1000000

The script is purely post-processing on saved arrays — runs in seconds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mfda.filtering import lowpass_filter


# ---------- temporal smoothers (causal — past samples only) ----------

def smooth_ema(phi: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average on the time axis: phi_smooth_n =
    alpha * phi_n + (1 - alpha) * phi_smooth_{n-1}.  alpha=1 → no smoothing.
    """
    out = np.empty_like(phi)
    out[0] = phi[0]
    for n in range(1, phi.shape[0]):
        out[n] = alpha * phi[n] + (1.0 - alpha) * out[n - 1]
    return out


def _sg_backward_coeffs(window: int, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing coefficients for the LAST sample of a
    backward window of size `window`. Polynomial fit of order polyorder
    over [n-window+1, ..., n], evaluate at n. Returns 1D array of length
    `window`."""
    if window <= polyorder:
        raise ValueError("window must be > polyorder")
    # Time axis points -window+1, ..., 0  (right-aligned at the latest sample).
    tau = np.arange(-window + 1, 1, dtype=float)
    A = np.vstack([tau ** k for k in range(polyorder + 1)]).T   # (window, p+1)
    # Smoothing = polynomial fit, evaluate at t=0.
    AtA_inv = np.linalg.inv(A.T @ A)
    # Coefficients to project a column of length `window` onto the value
    # of the fitted polynomial at t=0:  e_0^T (A^T A)^{-1} A^T = (A^T A)^{-1}_[0,:] @ A^T
    h = AtA_inv[0, :] @ A.T
    return h


def smooth_sg_backward(phi: np.ndarray, window: int = 11, polyorder: int = 2) -> np.ndarray:
    """Causal Savitzky-Golay: at each time n, fit a polynomial of order
    `polyorder` over the past `window` samples and evaluate at the most
    recent point. For n < window-1, fall back to a partial-window EMA-style
    smoothing (just clip).
    """
    h = _sg_backward_coeffs(window, polyorder)        # length window
    out = np.empty_like(phi)
    out[: window - 1] = phi[: window - 1]              # warm-up: no smoothing
    for n in range(window - 1, phi.shape[0]):
        out[n] = np.tensordot(h, phi[n - window + 1 : n + 1], axes=(0, 0))
    return out


# ---------- combined smoother application ----------

def apply_method(
    phi: np.ndarray, dt: float, L: float, method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (phi_smooth, z, w) for the given method.
    z and w are central time differences (interior) and clipped at endpoints
    so we get arrays of the same length as phi for plotting."""
    if method == "raw":
        phi_s = phi
        z, w = _finite_diff_zw(phi_s, dt)
    elif method == "spatial":
        phi_s = phi
        z_raw, w_raw = _finite_diff_zw(phi_s, dt)
        z = np.array([lowpass_filter(zr, L, k_cut_frac=0.25) for zr in z_raw])
        w = np.array([lowpass_filter(wr, L, k_cut_frac=0.15) for wr in w_raw])
    elif method.startswith("temporal_ema_a"):
        alpha = float(method.split("a")[-1].replace("p", "."))
        phi_s = smooth_ema(phi, alpha)
        z_raw, w_raw = _finite_diff_zw(phi_s, dt)
        z = np.array([lowpass_filter(zr, L, k_cut_frac=0.25) for zr in z_raw])
        w = np.array([lowpass_filter(wr, L, k_cut_frac=0.15) for wr in w_raw])
    elif method.startswith("temporal_sg_w"):
        window = int(method.split("w")[-1])
        phi_s = smooth_sg_backward(phi, window=window, polyorder=2)
        z_raw, w_raw = _finite_diff_zw(phi_s, dt)
        z = np.array([lowpass_filter(zr, L, k_cut_frac=0.25) for zr in z_raw])
        w = np.array([lowpass_filter(wr, L, k_cut_frac=0.15) for wr in w_raw])
    elif method == "combined":
        phi_s = smooth_ema(phi, alpha=0.3)
        z_raw, w_raw = _finite_diff_zw(phi_s, dt)
        z = np.array([lowpass_filter(zr, L, k_cut_frac=0.25) for zr in z_raw])
        w = np.array([lowpass_filter(wr, L, k_cut_frac=0.15) for wr in w_raw])
    elif method == "combined_heavy":
        # SG with w=41 + tighter spatial lowpass on w (k_cut=0.10).
        # Heavier than the default `combined`. Phase lag ~ 0.20 time units.
        phi_s = smooth_sg_backward(phi, window=41, polyorder=2)
        z_raw, w_raw = _finite_diff_zw(phi_s, dt)
        z = np.array([lowpass_filter(zr, L, k_cut_frac=0.20) for zr in z_raw])
        w = np.array([lowpass_filter(wr, L, k_cut_frac=0.10) for wr in w_raw])
    else:
        raise ValueError(f"unknown method: {method}")
    return phi_s, z, w


def _finite_diff_zw(phi: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Backward 1st difference (z) and central 2nd difference (w) of phi.
    z[n] = (phi[n] - phi[n-1]) / dt;  w[n] = (phi[n] - 2 phi[n-1] + phi[n-2]) / dt^2.
    Endpoints filled with the closest valid value (no fancy boundary)."""
    z = np.zeros_like(phi)
    z[1:] = (phi[1:] - phi[:-1]) / dt
    z[0] = z[1]
    w = np.zeros_like(phi)
    w[2:] = (phi[2:] - 2.0 * phi[1:-1] + phi[:-2]) / (dt * dt)
    w[:2] = w[2]
    return z, w


# ---------- visualisation ----------

METHODS = (
    "raw",
    "spatial",
    "temporal_ema_a0p3",
    "temporal_ema_a0p1",
    "temporal_sg_w11",
    "temporal_sg_w21",
    "temporal_sg_w41",        # 0.41 time units window — heavier smoothing
    "temporal_sg_w81",        # 0.81 time units window — much heavier
    "combined",
    "combined_heavy",         # SG_w41 + tighter spatial lowpass on w
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth-dir", required=True,
                    help="directory containing truth.npz")
    args = ap.parse_args()

    truth_dir = Path(args.truth_dir)
    data = np.load(truth_dir / "truth.npz")
    phi = data["phi"]                    # shape (Nt, Nx)
    t = data["t"]                        # shape (Nt,)
    Nt, Nx = phi.shape
    dt = float(t[1] - t[0])
    # Recover L from grid spacing (assuming uniform). Nx grid points cover [0, L).
    # We don't actually need L for the temporal methods but lowpass needs it.
    # Read it from the manifest if available, else assume L = 4*pi (k=0.5 default).
    manifest_path = truth_dir / "truth_manifest.json"
    L = 4.0 * np.pi
    if manifest_path.exists():
        import json
        m = json.loads(manifest_path.read_text())
        L = float(m["config"]["domain"]["L"])
    print(f"[smoothing_study] Nt={Nt}, Nx={Nx}, dt={dt}, L={L:.4f}")

    # Apply every method, accumulate results.
    out_payload: dict[str, np.ndarray] = {"t": t}
    results: dict[str, dict[str, np.ndarray]] = {}
    for method in METHODS:
        print(f"[smoothing_study] applying method={method}")
        phi_s, z, w = apply_method(phi, dt, L, method)
        results[method] = {"phi": phi_s, "z": z, "w": w}
        out_payload[f"{method}_phi"] = phi_s
        out_payload[f"{method}_z"] = z
        out_payload[f"{method}_w"] = w
    np.savez_compressed(truth_dir / "smoothed_observations.npz", **out_payload)
    print(f"[smoothing_study] wrote {truth_dir / 'smoothed_observations.npz'}")

    # ---- Visualisation: time-series at a fixed x ----
    # Three rows (phi, z, w), len(METHODS) columns.
    x_idx = Nx // 4   # pick a non-trivial spatial location
    n_methods = len(METHODS)
    fig, axes = plt.subplots(3, n_methods, figsize=(2.2 * n_methods, 6.4),
                             sharex=True)
    for j, method in enumerate(METHODS):
        r = results[method]
        axes[0, j].plot(t, r["phi"][:, x_idx], color="C0", lw=0.8)
        axes[0, j].set_title(method, fontsize=8)
        axes[1, j].plot(t, r["z"][:, x_idx], color="C1", lw=0.8)
        axes[2, j].plot(t, r["w"][:, x_idx], color="C3", lw=0.8)
        for ax in axes[:, j]:
            ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel(r"$\phi(t, x_*)$")
    axes[1, 0].set_ylabel(r"$z = \partial_t \phi$")
    axes[2, 0].set_ylabel(r"$w = \partial_t^2 \phi$")
    for ax in axes[-1, :]:
        ax.set_xlabel("t")
    fig.suptitle(f"{truth_dir.name}: smoothing-method comparison at x={x_idx*L/Nx:.2f}")
    fig.tight_layout()
    fig.savefig(truth_dir / "smoothing_comparison.png", dpi=140)
    plt.close(fig)
    print(f"[smoothing_study] wrote {truth_dir / 'smoothing_comparison.png'}")

    # ---- Spectral signature: |FFT|(omega, k_*) for each method ----
    # Pick the k=2pi/L mode (lowest non-zero), look at its time-frequency power.
    fig, axes = plt.subplots(3, n_methods, figsize=(2.2 * n_methods, 6.4),
                             sharex=True, sharey="row")
    omega = 2.0 * np.pi * np.fft.rfftfreq(Nt, d=dt)
    for j, method in enumerate(METHODS):
        r = results[method]
        for i, key in enumerate(("phi", "z", "w")):
            arr = r[key][:, x_idx]
            ps = np.abs(np.fft.rfft(arr - arr.mean())) ** 2
            axes[i, j].loglog(omega[1:], ps[1:], color="C0" if i == 0 else ("C1" if i == 1 else "C3"))
            axes[i, j].grid(alpha=0.3, which="both")
        axes[0, j].set_title(method, fontsize=8)
    axes[0, 0].set_ylabel(r"$|\hat\phi|^2$")
    axes[1, 0].set_ylabel(r"$|\hat z|^2$")
    axes[2, 0].set_ylabel(r"$|\hat w|^2$")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\omega$")
    fig.suptitle(f"{truth_dir.name}: power spectrum of phi, z, w at x={x_idx*L/Nx:.2f}")
    fig.tight_layout()
    fig.savefig(truth_dir / "smoothing_spectrum.png", dpi=140)
    plt.close(fig)
    print(f"[smoothing_study] wrote {truth_dir / 'smoothing_spectrum.png'}")


if __name__ == "__main__":
    main()
