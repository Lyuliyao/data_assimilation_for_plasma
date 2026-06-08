"""2D spectral low-pass filter for residuals (mirror of filtering.py)."""
from __future__ import annotations

import numpy as np


def lowpass_filter_2d(
    r: np.ndarray, Lx: float, Ly: float,
    k_cut_frac: float = 0.25, sharpness: float = 16.0,
) -> np.ndarray:
    """Smooth spectral low-pass on a 2D periodic field.

    H(kx, ky) = exp(-(|k| / k_c)^sharpness)
    where |k|^2 = kx^2 + ky^2 and k_c = k_cut_frac * max(k_max_x, k_max_y).
    """
    Nx, Ny = r.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
    ky = 2.0 * np.pi * np.fft.rfftfreq(Ny, d=Ly / Ny)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k_mag = np.sqrt(KX * KX + KY * KY)
    k_max = max(np.abs(kx).max(), np.abs(ky).max())
    k_c = k_cut_frac * k_max
    with np.errstate(divide="ignore", invalid="ignore"):
        H = np.exp(-((k_mag / k_c) ** sharpness))
    H[0, 0] = 1.0  # don't touch the DC mode
    r_hat = np.fft.rfft2(r)
    return np.fft.irfft2(H * r_hat, s=(Nx, Ny))
