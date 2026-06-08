"""Unit tests for kinetic_stress.cic_deposit_kinetic_stress.

Sanity checks:
  - Particles all at v = u0 with uniform x: M(x) = u0^2 * rho(x) within
    shot noise.
  - v = 0: M = 0.
  - Quadratic in v: deposit(v=2*u) = 4 * deposit(v=u).
"""
from __future__ import annotations

import numpy as np

from mfda.backend_reference import cic_deposit
from mfda.kinetic_stress import cic_deposit_kinetic_stress


def test_uniform_v_recovers_v2_times_rho() -> None:
    rng = np.random.default_rng(0)
    L = 2.0 * np.pi
    Nx = 64
    Np = 200_000
    u0 = 1.7
    x = rng.uniform(0.0, L, Np)
    v = np.full(Np, u0)
    w = np.full(Np, L / Np)
    rho = cic_deposit(x, w, L, Nx)
    M = cic_deposit_kinetic_stress(x, v, w, L, Nx)
    expected = (u0 * u0) * rho
    assert np.allclose(M, expected, atol=1e-10)


def test_zero_velocity_gives_zero_stress() -> None:
    rng = np.random.default_rng(1)
    L = 1.0
    Nx = 16
    Np = 1000
    x = rng.uniform(0.0, L, Np)
    v = np.zeros(Np)
    w = np.ones(Np) * L / Np
    M = cic_deposit_kinetic_stress(x, v, w, L, Nx)
    assert np.allclose(M, 0.0, atol=1e-15)


def test_quadratic_in_v() -> None:
    rng = np.random.default_rng(2)
    L = 5.0
    Nx = 32
    Np = 5000
    x = rng.uniform(0.0, L, Np)
    v = rng.standard_normal(Np)
    w = np.full(Np, L / Np)
    M1 = cic_deposit_kinetic_stress(x, v, w, L, Nx)
    M2 = cic_deposit_kinetic_stress(x, 2.0 * v, w, L, Nx)
    assert np.allclose(M2, 4.0 * M1, atol=1e-12)
