"""Unit tests for observation_time.time_derivative_observation.

We check:
  - For low-frequency analytic input the lowpass barely attenuates and the
    finite difference recovers the analytic derivative to high precision.
  - The output is mean-free.
  - dt_obs <= 0 raises ValueError.
"""
from __future__ import annotations

import numpy as np
import pytest

from mfda.observation_time import time_derivative_observation


def test_recovers_analytic_low_frequency_derivative() -> None:
    Nx = 128
    L = 2.0 * np.pi
    x = np.linspace(0.0, L, Nx, endpoint=False)
    omega = 1.5
    g = np.cos(x)  # low-frequency spatial profile (mode 1)
    dt = 1e-3      # finite-difference truncation O(dt^2) on midpoint -> tiny
    t0 = 0.4
    t1 = t0 + dt

    y_prev = np.sin(omega * t0) * g
    y_now = np.sin(omega * t1) * g

    z = time_derivative_observation(
        y_now, y_prev, dt, L,
        lowpass_k_cut_frac=0.5,  # well above the mode-1 we're feeding in
        lowpass_sharpness=16.0,
    )
    # Finite-difference midpoint estimate of d/dt[sin(w t) g] at (t0+t1)/2
    z_exact = omega * np.cos(omega * (t0 + t1) / 2.0) * g
    z_exact = z_exact - z_exact.mean()
    assert np.allclose(z, z_exact, atol=5e-4)


def test_output_is_mean_free() -> None:
    rng = np.random.default_rng(0)
    Nx = 64
    L = 1.0
    y_prev = rng.standard_normal(Nx)
    y_now = rng.standard_normal(Nx)
    z = time_derivative_observation(y_now, y_prev, dt_obs=0.01, L=L)
    assert abs(z.mean()) < 1e-12


def test_invalid_dt_raises() -> None:
    Nx = 8
    y = np.zeros(Nx)
    with pytest.raises(ValueError):
        time_derivative_observation(y, y, dt_obs=0.0, L=1.0)
    with pytest.raises(ValueError):
        time_derivative_observation(y, y, dt_obs=-1.0, L=1.0)
