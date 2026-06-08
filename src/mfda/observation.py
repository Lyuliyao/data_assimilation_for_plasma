"""Observation operators for potential-only data.

Matches section 2 of the note:
    (a) full-field noiseless
    (b) noisy full-field
    (c) coarse spatial
    (d) sparse-in-time

Spatial observation operators act on a Poisson-grid potential phi of shape
(Nx,). Sparse-in-time is handled in the assimilation loop (we just control
how often y is fed in), not here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ObservationSpec:
    kind: str  # "full", "noisy", "coarse"
    sigma: float = 0.0          # used by "noisy"
    every_m: int = 1            # used by "coarse": keep grid points 0, m, 2m, ...
    reconstruction: str = "fourier"  # "fourier" or "linear"
    # Sparse-in-time is a property of the assim loop, not a per-step operator.
    every_q: int = 1            # assimilate every q time steps
    rng_seed: int = 0


def observe(phi_true: np.ndarray, spec: ObservationSpec, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return y from phi_true according to spec.

    For "full": y = phi_true.
    For "noisy": y = phi_true + sigma * xi with xi ~ N(0, 1) pointwise.
    For "coarse": y has Nx samples but the (m-1) grid points between observed
                  ones are filled in by `reconstruction`:
                    - "fourier": keep only low-k modes consistent with the
                                 coarse sample rate.
                    - "linear":  piecewise-linear interpolation from the
                                 observed points.
    """
    if rng is None:
        rng = np.random.default_rng(spec.rng_seed)
    if spec.kind == "full":
        return phi_true.copy()

    if spec.kind == "noisy":
        return phi_true + spec.sigma * rng.standard_normal(phi_true.shape)

    if spec.kind == "coarse":
        Nx = phi_true.shape[0]
        m = max(1, int(spec.every_m))
        idx = np.arange(0, Nx, m)
        samples = phi_true[idx]
        if spec.reconstruction == "linear":
            # Linear interp, respecting periodicity.
            x_full = np.arange(Nx, dtype=float)
            x_obs = idx.astype(float)
            # Append wrap-around point.
            x_obs_ext = np.concatenate([x_obs, [Nx + x_obs[0]]])
            samples_ext = np.concatenate([samples, samples[:1]])
            return np.interp(x_full, x_obs_ext, samples_ext)
        if spec.reconstruction == "fourier":
            # Zero-pad in Fourier from the coarse DFT, i.e. low-k reconstruction.
            Nc = samples.shape[0]
            Yc = np.fft.rfft(samples)
            Yf = np.zeros(Nx // 2 + 1, dtype=complex)
            k_keep = Yc.shape[0]
            Yf[:k_keep] = Yc
            # Scaling: rfft of a sub-sample is off by a factor Nx / Nc.
            Yf *= Nx / Nc
            return np.fft.irfft(Yf, n=Nx)
        raise ValueError(f"Unknown reconstruction: {spec.reconstruction!r}")

    raise ValueError(f"Unknown observation kind: {spec.kind!r}")


def rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a * a)))


def noise_sigma_from_snr(phi_true_rms: float, snr_db: float) -> float:
    """Convert a target signal-to-noise ratio (in dB) to a sigma for white noise.

    sigma = rms_signal * 10^(-snr_db / 20).
    """
    return phi_true_rms * 10.0 ** (-snr_db / 20.0)
