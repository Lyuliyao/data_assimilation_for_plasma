"""Time-derivative observation z = d phi / dt, built from successive y snapshots.

Section 1 of v12 of the note augments the observation functional with

    (alpha / 2) * integral |d phi_f / dt - z|^2 dx

where z(x, t) := (y(x, t) - y(x, t - dt_obs)) / dt_obs.

The time derivative amplifies high-frequency noise by 1 / dt^2 (~ 10^4 at
the canonical dt = 1e-2). Lowpass filtering of z is therefore mandatory, not
optional - see pitfall #1 in docs/time_derivative_observation_plan.md.
"""
from __future__ import annotations

import numpy as np

from .filtering import lowpass_filter


def time_second_derivative_observation(
    y_now: np.ndarray,
    y_prev: np.ndarray,
    y_prev2: np.ndarray,
    dt_obs: float,
    L: float,
    lowpass_k_cut_frac: float = 0.15,
    lowpass_sharpness: float = 16.0,
) -> np.ndarray:
    """Build the lowpass-filtered second finite-difference of y.

        w(x, t) := (y_n - 2 y_{n-1} + y_{n-2}) / dt_obs^2

    Approximates d^2 y / dt^2 at t_{n-1}. Noise variance amplifies as
    4 * var(y) / dt^4 ~ 10^8 at the canonical dt = 1e-2, six orders worse
    than z. Lowpass with k_cut_frac = 0.15 by default (vs. 0.25 for z and
    0.40 for the snapshot residual) — see pitfall #4.1 of
    docs/second_derivative_observation_plan.md. Mean-subtracted because
    d^2 phi / dt^2 is mean-free in the periodic gauge.
    """
    if dt_obs <= 0.0:
        raise ValueError(f"dt_obs must be positive, got {dt_obs}")
    w = (y_now - 2.0 * y_prev + y_prev2) / (dt_obs * dt_obs)
    w = lowpass_filter(w, L, k_cut_frac=lowpass_k_cut_frac, sharpness=lowpass_sharpness)
    return w - w.mean()


def time_derivative_observation(
    y_now: np.ndarray,
    y_prev: np.ndarray,
    dt_obs: float,
    L: float,
    lowpass_k_cut_frac: float = 0.25,
    lowpass_sharpness: float = 16.0,
) -> np.ndarray:
    """Build the lowpass-filtered finite-difference time derivative of y.

    Parameters
    ----------
    y_now, y_prev : (Nx,) arrays
        Two consecutive observation snapshots.
    dt_obs : float
        Time elapsed between y_prev and y_now (= every_q * dt for snapshot
        observation).
    L : float
        Spatial period.
    lowpass_k_cut_frac, lowpass_sharpness : float
        Forwarded to mfda.filtering.lowpass_filter.

    Returns
    -------
    z : (Nx,) array
        Mean-free, lowpass-filtered estimate of d y / dt at the midpoint
        of the interval. The mean is subtracted because the underlying
        d phi / dt is mean-free in the periodic gauge used elsewhere.
    """
    if dt_obs <= 0.0:
        raise ValueError(f"dt_obs must be positive, got {dt_obs}")
    z = (y_now - y_prev) / dt_obs
    z = lowpass_filter(z, L, k_cut_frac=lowpass_k_cut_frac, sharpness=lowpass_sharpness)
    return z - z.mean()
