"""Verify 2D periodic Poisson and spectral gradient."""
from __future__ import annotations

import numpy as np

from mfda.poisson_2d import (
    electric_field_from_density_2d,
    grad_2d,
    potential_from_density_2d,
    solve_poisson_2d,
)


def test_solve_poisson_2d_recovers_known_eigenmode() -> None:
    """For source = cos(kx X) cos(ky Y), solution is source / (kx^2 + ky^2)
    when both kx, ky are integer multiples of (2 pi / L)."""
    Lx = 2.0 * np.pi
    Ly = 4.0 * np.pi
    Nx = 64
    Ny = 128
    kx = 1.0  # mode 1 of Lx
    ky = 0.5  # mode 1 of Ly (period 4 pi)
    x = np.arange(Nx) * Lx / Nx
    y = np.arange(Ny) * Ly / Ny
    X, Y = np.meshgrid(x, y, indexing="ij")
    source = np.cos(kx * X) * np.cos(ky * Y)
    expected = source / (kx * kx + ky * ky)
    u = solve_poisson_2d(source, Lx, Ly)
    np.testing.assert_allclose(u, expected, atol=1e-10)


def test_solve_poisson_2d_zero_mean() -> None:
    """The solution always has zero spatial mean."""
    rng = np.random.default_rng(0)
    Lx = Ly = 2.0 * np.pi
    Nx = Ny = 32
    source = rng.standard_normal((Nx, Ny))
    source -= source.mean()
    u = solve_poisson_2d(source, Lx, Ly)
    assert abs(u.mean()) < 1e-12


def test_grad_2d_recovers_known_gradient() -> None:
    """Gradient of f = sin(x) cos(y) is (cos(x) cos(y), -sin(x) sin(y))."""
    L = 2.0 * np.pi
    Nx = Ny = 64
    x = np.arange(Nx) * L / Nx
    y = np.arange(Ny) * L / Ny
    X, Y = np.meshgrid(x, y, indexing="ij")
    f = np.sin(X) * np.cos(Y)
    dfdx_expected = np.cos(X) * np.cos(Y)
    dfdy_expected = -np.sin(X) * np.sin(Y)
    dfdx, dfdy = grad_2d(f, L, L)
    np.testing.assert_allclose(dfdx, dfdx_expected, atol=1e-10)
    np.testing.assert_allclose(dfdy, dfdy_expected, atol=1e-10)


def test_uniform_density_zero_field() -> None:
    """For uniform rho = background, both Ex and Ey are exactly zero."""
    Lx = Ly = 2.0 * np.pi
    Nx = Ny = 32
    rho = np.ones((Nx, Ny))
    Ex, Ey = electric_field_from_density_2d(rho, Lx, Ly)
    np.testing.assert_allclose(Ex, 0.0, atol=1e-12)
    np.testing.assert_allclose(Ey, 0.0, atol=1e-12)


def test_potential_consistency_with_field() -> None:
    """E = -grad phi recovered from rho should match field-from-density helper."""
    rng = np.random.default_rng(0)
    Lx = Ly = 2.0 * np.pi
    Nx = Ny = 32
    rho = 1.0 + 0.3 * rng.standard_normal((Nx, Ny))
    phi = potential_from_density_2d(rho, Lx, Ly)
    dphidx, dphidy = grad_2d(phi, Lx, Ly)
    Ex, Ey = electric_field_from_density_2d(rho, Lx, Ly)
    np.testing.assert_allclose(Ex, -dphidx, atol=1e-12)
    np.testing.assert_allclose(Ey, -dphidy, atol=1e-12)
