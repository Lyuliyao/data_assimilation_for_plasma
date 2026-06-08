"""Unit tests for poisson.solve_poisson_from_div.

We solve  -d^2 u / dx^2 = -dj/dx  with periodic BC and zero mean.
For j(x) = sin(k x), -dj/dx = -k cos(k x) and u(x) = -cos(k x) / k - which
is mean-free.  The implementation should recover this to FFT precision.
"""
from __future__ import annotations

import numpy as np

from mfda.poisson import solve_poisson_from_div


def test_recovers_analytic_sin_current() -> None:
    Nx = 128
    L = 2.0 * np.pi
    x = np.linspace(0.0, L, Nx, endpoint=False)
    for n_mode in (1, 2, 3, 5):
        k = 2.0 * np.pi * n_mode / L
        j = np.sin(k * x)
        u = solve_poisson_from_div(j, L)
        u_exact = -np.cos(k * x) / k
        u_exact = u_exact - u_exact.mean()
        assert np.allclose(u, u_exact, atol=1e-12), f"n_mode={n_mode}"


def test_zero_mean_enforced() -> None:
    Nx = 64
    L = 4.0
    rng = np.random.default_rng(0)
    j = rng.standard_normal(Nx)
    u = solve_poisson_from_div(j, L)
    assert abs(u.mean()) < 1e-12


def test_constant_current_is_zero() -> None:
    """A constant j has zero divergence; ψ should therefore be zero."""
    Nx = 32
    L = 1.0
    j = np.full(Nx, 0.7)
    u = solve_poisson_from_div(j, L)
    assert np.allclose(u, 0.0, atol=1e-12)
