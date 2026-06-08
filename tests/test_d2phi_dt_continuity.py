"""Unit tests for poisson.solve_poisson_from_d2.

The 1D continuity-derived second time derivative of phi is
    -Delta (d^2 phi / dt^2) = d^2 M / dx^2 - d/dx (rho * E)

For M(x) = sin(2 x), rho * E = 0:
    d^2 M / dx^2 = -4 sin(2 x)
    -Delta u = -4 sin(2 x)
    -> u(x) = -sin(2 x)        (check: -u'' = -(-(-4) sin(2x)) = -4 sin(2x) ✓)

So d^2 phi / dt^2 = -sin(2 x) for that input.
"""
from __future__ import annotations

import numpy as np

from mfda.poisson import solve_poisson_from_d2


def test_recovers_analytic_for_sin_M() -> None:
    Nx = 256
    L = 2.0 * np.pi
    x = np.linspace(0.0, L, Nx, endpoint=False)
    for n_mode in (1, 2, 3):
        kx = n_mode  # physical wavenumber for L=2pi grid
        M = np.sin(kx * x)
        rho = np.ones(Nx)
        E = np.zeros(Nx)
        d2phi_dt = solve_poisson_from_d2(M, rho, E, L)
        # Source = -kx^2 sin(kx x).  Solving -Delta u = source gives
        # u = source / kx^2 in Fourier = -sin(kx x).
        analytic = -np.sin(kx * x)
        assert np.allclose(d2phi_dt, analytic, atol=1e-12), f"n_mode={n_mode}"


def test_rho_E_term_contributes_correctly() -> None:
    """If d^2 M / dx^2 = 0 and rho * E = sin(x), source = -d/dx sin(x) = -cos(x).
    Then -Delta u = -cos(x) -> u = -cos(x).
    """
    Nx = 128
    L = 2.0 * np.pi
    x = np.linspace(0.0, L, Nx, endpoint=False)
    M = np.zeros(Nx)
    rho = np.ones(Nx)
    E = np.sin(x)  # rho * E = sin(x)
    d2phi_dt = solve_poisson_from_d2(M, rho, E, L)
    analytic = -np.cos(x)
    analytic = analytic - analytic.mean()
    assert np.allclose(d2phi_dt, analytic, atol=1e-12)


def test_zero_inputs_give_zero() -> None:
    Nx = 32
    L = 1.0
    z = np.zeros(Nx)
    out = solve_poisson_from_d2(z, np.ones(Nx), z, L)
    assert np.allclose(out, 0.0, atol=1e-14)
