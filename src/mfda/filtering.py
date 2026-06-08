"""Low-pass spectral filtering of residuals.

Practical recommendation #4 of the note: begin with low-pass filtered residuals.
Unfiltered PIC residuals can inject particle noise back into the dynamics.

We filter the potential residual r(x) = phi_f(x) - y(x) with a smooth spectral
low-pass before feeding it into the adjoint Poisson solve.
"""
from __future__ import annotations

import numpy as np


def lowpass_filter(r: np.ndarray, L: float, k_cut_frac: float = 0.25, sharpness: float = 16.0) -> np.ndarray:
    """Smooth spectral low-pass filter.

    Parameters
    ----------
    r : (Nx,) array
        Real-valued signal on a periodic grid of length L.
    k_cut_frac : float
        Cutoff wavenumber as a fraction of the Nyquist k_max. Default 0.25.
    sharpness : float
        Steepness of the exponential roll-off. Larger = sharper.

    Returns
    -------
    r_f : (Nx,) array
        Filtered signal.

    Notes
    -----
    Uses the filter   H(k) = exp(-((k/k_c)^sharpness)).
    This is the classical 'Hou–Li' / 2/3-rule style filter and is smooth enough
    to avoid Gibbs ringing while killing high-k particle noise.
    """
    Nx = r.shape[0]
    k = 2.0 * np.pi * np.fft.rfftfreq(Nx, d=L / Nx)
    k_max = k.max()
    k_c = k_cut_frac * k_max
    with np.errstate(divide="ignore", invalid="ignore"):
        H = np.exp(-((k / k_c) ** sharpness))
    H[0] = 1.0  # don't touch the mean
    r_hat = np.fft.rfft(r)
    return np.fft.irfft(H * r_hat, n=Nx)
