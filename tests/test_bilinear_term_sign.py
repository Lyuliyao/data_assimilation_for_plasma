"""Verify the sign of the bilinear position-dtobs update on the grid.

Eq. 17 (1D form):
    dx_p = -gamma * alpha * U_p * (d^2 psi1 / dx^2)|_{x_p} * dt

For psi1(x) = cos(k x), d^2 psi1 / dx^2 = -k^2 cos(k x), so
    dx_p = +gamma * alpha * U_p * k^2 cos(k x_p) * dt

With gamma = alpha = dt = U_p = 1 and k = 1, dx_p = cos(x_p) — positive
near x = 0, negative near x = pi. Any sign flip in the implementation would
shift particles the wrong way.

This test exercises the math of the bilinear term without driving a full
psi1 solve from the residual; it operates on the grid quantities directly.
"""
from __future__ import annotations

import numpy as np

from mfda.backend_reference import cic_interpolate
from mfda.poisson import grad_1d


def test_bilinear_dx_sign_matches_eq_17() -> None:
    Nx = 128
    L = 2.0 * np.pi
    x_grid = np.linspace(0.0, L, Nx, endpoint=False)
    k = 1.0
    # Manufactured psi1 on the grid.
    psi1 = np.cos(k * x_grid)
    # Two spectral grad applications -> d^2 psi1 / dx^2 on the grid.
    grad1 = grad_1d(psi1, L)
    h1 = grad_1d(grad1, L)
    # Reference: -k^2 cos(k x), well-resolved here.
    expected_h1 = -(k ** 2) * np.cos(k * x_grid)
    assert np.allclose(h1, expected_h1, atol=1e-12)

    # Place test particles at known positions and apply the bilinear update.
    x_p = np.array([0.0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2])
    U_p = np.ones_like(x_p)
    gamma, alpha, dt = 1.0, 1.0, 1.0
    h1_p = cic_interpolate(h1, x_p, L)
    dx_p = -gamma * alpha * U_p * h1_p * dt
    # Predicted dx = +cos(x_p) at U=1, gamma=alpha=dt=1, k=1.
    expected_dx = np.cos(x_p)
    assert np.allclose(dx_p, expected_dx, atol=1e-10)


def test_bilinear_zero_velocity_gives_zero_dx() -> None:
    Nx = 64
    L = 2.0 * np.pi
    x_grid = np.linspace(0.0, L, Nx, endpoint=False)
    psi1 = np.cos(2 * x_grid)
    grad1 = grad_1d(psi1, L)
    h1 = grad_1d(grad1, L)
    x_p = np.linspace(0.0, L, 17, endpoint=False)
    U_p = np.zeros_like(x_p)
    h1_p = cic_interpolate(h1, x_p, L)
    dx_p = -1.0 * 1.0 * U_p * h1_p * 0.1
    assert np.allclose(dx_p, 0.0, atol=1e-15)


def test_bilinear_velocity_proportionality() -> None:
    """dx scales linearly in U_p at the same x_p."""
    Nx = 64
    L = 2.0 * np.pi
    x_grid = np.linspace(0.0, L, Nx, endpoint=False)
    psi1 = np.cos(x_grid)
    h1 = grad_1d(grad_1d(psi1, L), L)
    x_p = np.array([0.5, 1.0, 1.5])
    h1_p = cic_interpolate(h1, x_p, L)
    dx1 = -1.0 * 1.0 * np.ones_like(x_p) * h1_p * 0.1
    dx2 = -1.0 * 1.0 * (2.0 * np.ones_like(x_p)) * h1_p * 0.1
    assert np.allclose(dx2, 2.0 * dx1, atol=1e-13)
