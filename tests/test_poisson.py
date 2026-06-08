"""Unit tests for the FFT Poisson solver."""
from __future__ import annotations

import numpy as np
import pytest

from mfda.poisson import (
    adjoint_potential,
    electric_field_from_density,
    grad_1d,
    potential_from_density,
    solve_poisson_1d,
)


@pytest.mark.parametrize("Nx", [32, 64, 128])
def test_solve_recovers_mms(Nx: int) -> None:
    L = 2.0 * np.pi
    x = np.linspace(0.0, L, Nx, endpoint=False)
    # u = cos(k x) + 0.5 sin(2 k x) has -u'' = k^2 cos(k x) + 2 k^2 sin(2 k x)
    k = 1.0
    u_true = np.cos(k * x) + 0.5 * np.sin(2.0 * k * x)
    source = (k * k) * np.cos(k * x) + (2.0 * k) ** 2 * 0.5 * np.sin(2.0 * k * x)
    u = solve_poisson_1d(source, L)
    # u_true is zero-mean, so direct comparison works
    assert np.allclose(u, u_true, atol=1e-10)


def test_grad_1d_constant_is_zero() -> None:
    L = 2.0 * np.pi
    Nx = 64
    u = np.ones(Nx) * 3.14
    assert np.allclose(grad_1d(u, L), 0.0, atol=1e-12)


def test_grad_1d_sine() -> None:
    L = 2.0 * np.pi
    Nx = 64
    x = np.linspace(0.0, L, Nx, endpoint=False)
    u = np.sin(x)
    du = grad_1d(u, L)
    assert np.allclose(du, np.cos(x), atol=1e-10)


def test_zero_mean_enforced() -> None:
    L = 4.0
    Nx = 64
    x = np.linspace(0.0, L, Nx, endpoint=False)
    # Non-zero-mean source should still yield zero-mean u (k=0 is dropped).
    source = np.cos(2.0 * np.pi * x / L) + 0.7
    u = solve_poisson_1d(source, L)
    assert abs(u.mean()) < 1e-12


def test_adjoint_consistency() -> None:
    """If phi_f = y + delta, then adjoint psi satisfies -Delta psi = delta."""
    L = 2.0 * np.pi
    Nx = 64
    x = np.linspace(0.0, L, Nx, endpoint=False)
    y = np.cos(x)
    delta = 0.1 * np.cos(2.0 * x)
    phi_f = y + delta
    psi = adjoint_potential(phi_f, y, L)
    # Numerically verify -Delta psi = delta via spectral derivative.
    # d2 psi / dx^2 = -delta => -grad(grad(psi)) = delta (mean-free).
    dpsi = grad_1d(psi, L)
    ddpsi = grad_1d(dpsi, L)
    assert np.allclose(-ddpsi, delta - delta.mean(), atol=1e-10)


def test_electric_field_from_cosine_density() -> None:
    """rho = 1 + a cos(k x) -> phi = (a/k^2) cos(k x), E = (a/k) sin(k x)."""
    L = 2.0 * np.pi
    Nx = 128
    x = np.linspace(0.0, L, Nx, endpoint=False)
    a = 0.01
    k = 1.0
    rho = 1.0 + a * np.cos(k * x)
    phi = potential_from_density(rho, L)
    E = electric_field_from_density(rho, L)
    assert np.allclose(phi, (a / k ** 2) * np.cos(k * x), atol=1e-10)
    assert np.allclose(E, (a / k) * np.sin(k * x), atol=1e-10)
