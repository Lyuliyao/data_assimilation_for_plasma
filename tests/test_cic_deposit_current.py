"""Unit tests for backend_reference.cic_deposit_current.

Sanity checks:
  - Uniform stream (v_p == u0, x uniform): j should be ~ u0 * mean(rho)
    everywhere, within shot noise.
  - Setting v == 0 reduces to deposit of zero -> j = 0.
  - Linearity in v: deposit(v=2*u0) == 2 * deposit(v=u0).
"""
from __future__ import annotations

import numpy as np

from mfda.backend_reference import cic_deposit, cic_deposit_current


def test_uniform_stream_recovers_u0_times_rho() -> None:
    rng = np.random.default_rng(0)
    L = 2.0 * np.pi
    Nx = 64
    Np = 200_000
    u0 = 1.7
    x = rng.uniform(0.0, L, Np)
    v = np.full(Np, u0)
    w = np.full(Np, L / Np)  # neutralises mean rho to 1.0
    rho = cic_deposit(x, w, L, Nx)
    j = cic_deposit_current(x, v, w, L, Nx)
    # Pointwise tolerance is shot-noise limited (~ 1/sqrt(Np/Nx)).
    expected = u0 * rho
    assert np.allclose(j, expected, atol=1e-10)


def test_zero_velocity_gives_zero_current() -> None:
    rng = np.random.default_rng(1)
    L = 1.0
    Nx = 16
    Np = 1000
    x = rng.uniform(0.0, L, Np)
    v = np.zeros(Np)
    w = np.ones(Np) * L / Np
    j = cic_deposit_current(x, v, w, L, Nx)
    assert np.allclose(j, 0.0, atol=1e-15)


def test_linearity_in_v() -> None:
    rng = np.random.default_rng(2)
    L = 5.0
    Nx = 32
    Np = 5000
    x = rng.uniform(0.0, L, Np)
    v = rng.standard_normal(Np)
    w = np.full(Np, L / Np)
    j1 = cic_deposit_current(x, v, w, L, Nx)
    j2 = cic_deposit_current(x, 2.0 * v, w, L, Nx)
    assert np.allclose(j2, 2.0 * j1, atol=1e-12)
