"""Unit tests for time_second_derivative_observation (compute_w).

For y(t) = sin(omega t) * cos(k x), the analytic d^2 y / dt^2 is
    d^2/dt^2 [sin(omega t) cos(k x)] = -omega^2 sin(omega t) cos(k x).

A centered second finite difference with step dt approximates this with
O(dt^2) error: at the midpoint t we evaluate
    (y(t + dt) - 2 y(t) + y(t - dt)) / dt^2  ~  d^2y/dt^2 |_t  +  O(dt^2)

Pass the lowpass cutoff that doesn't suppress the analytic mode.
"""
from __future__ import annotations

import numpy as np
import pytest

from mfda.observation_time import time_second_derivative_observation


def test_recovers_analytic_low_frequency_d2() -> None:
    Nx = 128
    L = 2.0 * np.pi
    x = np.linspace(0.0, L, Nx, endpoint=False)
    omega = 0.5
    g = np.cos(x)
    dt = 1e-3
    t0 = 0.4
    t1 = t0 + dt
    t2 = t0 + 2.0 * dt

    y0 = np.sin(omega * t0) * g
    y1 = np.sin(omega * t1) * g
    y2 = np.sin(omega * t2) * g

    # The function expects (y_now=y2, y_prev=y1, y_prev2=y0). At dt_obs=dt
    # this evaluates the second difference centered at t1.
    w = time_second_derivative_observation(
        y2, y1, y0, dt, L,
        lowpass_k_cut_frac=0.5,
        lowpass_sharpness=16.0,
    )
    w_exact = -(omega ** 2) * np.sin(omega * t1) * g
    w_exact = w_exact - w_exact.mean()
    # Loose tolerance (1e-4) — finite-difference truncation O(dt^2) at
    # dt = 1e-3 gives ~1e-6 error before lowpass; lowpass adds bias.
    assert np.allclose(w, w_exact, atol=1e-4)


def test_output_is_mean_free() -> None:
    rng = np.random.default_rng(0)
    Nx = 64
    L = 1.0
    y0, y1, y2 = (rng.standard_normal(Nx) for _ in range(3))
    w = time_second_derivative_observation(y2, y1, y0, dt_obs=0.01, L=L)
    assert abs(w.mean()) < 1e-12


def test_invalid_dt_raises() -> None:
    Nx = 8
    y = np.zeros(Nx)
    with pytest.raises(ValueError):
        time_second_derivative_observation(y, y, y, dt_obs=0.0, L=1.0)
    with pytest.raises(ValueError):
        time_second_derivative_observation(y, y, y, dt_obs=-1.0, L=1.0)
