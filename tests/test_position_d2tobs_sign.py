"""Sign and magnitude verification for the three position_d2tobs sub-terms.

Per docs/second_derivative_observation_plan.md §1.3, the position-channel
update from the second-derivative-observation term is

    dx_p = -gamma * beta * [ U^2 * H_3(psi_2)            (sub-term A)
                            + d/dx (E * H_1(psi_2))      (sub-term B)
                            + H_1(chi)                   (sub-term C)
                          ] * dt

where chi solves -Delta chi = grad . (rho grad psi_2).

For psi_2(x) = cos(k x):
    H_1(psi_2) = -k sin(k x)
    H_2(psi_2) = -k^2 cos(k x)
    H_3(psi_2) =  k^3 sin(k x)

At x = pi/(2k) (so k*x = pi/2): sin = 1, cos = 0.
    H_1(psi_2) = -k
    H_2(psi_2) = 0
    H_3(psi_2) = +k^3

For rho = 1, ψ_2 = cos(k x): chi = -cos(k x)  (see test_chi_solve.py).
    H_1(chi) = k sin(k x), so at x = pi/(2k): H_1(chi) = +k.

Sub-term A at U = 1, gamma = beta = 1:
    -1 * 1 * 1 * k^3 = -k^3   (decelerates in x)
Sub-term C at gamma = beta = 1:
    -1 * 1 * k = -k
"""
from __future__ import annotations

import numpy as np

from mfda.backend_reference import cic_interpolate
from mfda.poisson import grad_1d, solve_chi


def _grids(k: float, Nx: int = 256, L: float = 2.0 * np.pi):
    x_grid = np.linspace(0.0, L, Nx, endpoint=False)
    psi2 = np.cos(k * x_grid)
    h1 = grad_1d(psi2, L)
    h2 = grad_1d(h1, L)
    h3 = grad_1d(h2, L)
    return x_grid, psi2, h1, h2, h3


def test_h3_recovers_analytic_for_cos_psi2() -> None:
    """h3 = d^3 psi_2 / dx^3 should equal k^3 sin(k x) for psi_2 = cos(k x)."""
    k = 1.0
    x_grid, psi2, h1, h2, h3 = _grids(k)
    expected = (k ** 3) * np.sin(k * x_grid)
    # h3 is three composed grad_1d calls; FP noise at the ~1e-10 level
    # leaks through the Nyquist zero-out, hence a looser tolerance.
    assert np.allclose(h3, expected, atol=1e-9)


def test_subterm_A_sign() -> None:
    """U^2 * H_3(psi_2) at x = pi/2 should be +k^3 for psi_2 = cos(k x)."""
    k = 1.0
    x_grid, psi2, h1, h2, h3 = _grids(k)
    x_p = np.array([np.pi / (2.0 * k)])
    U_p = np.array([1.0])
    h3_p = cic_interpolate(h3, x_p, 2.0 * np.pi)
    assert np.allclose(h3_p, k ** 3, atol=1e-10)
    A = U_p * U_p * h3_p
    # The minus sign in dx = ... - gamma*beta*A*dt makes the particle
    # decelerate in x at this position. Just check A > 0 here.
    assert A[0] > 0


def test_subterm_C_chi_h1_sign() -> None:
    """H_1(chi) at x = pi/(2k) for psi_2=cos(kx), rho=1 should be +k."""
    k = 1.0
    L = 2.0 * np.pi
    Nx = 256
    x_grid, psi2, h1, h2, h3 = _grids(k, Nx=Nx, L=L)
    rho = np.ones(Nx)
    chi = solve_chi(rho, h1, L)
    expected_chi = -np.cos(k * x_grid)
    assert np.allclose(chi, expected_chi, atol=1e-12)
    grad_chi = grad_1d(chi, L)
    expected_grad_chi = k * np.sin(k * x_grid)
    assert np.allclose(grad_chi, expected_grad_chi, atol=1e-12)
    # Interpolate at x = pi/(2k)
    x_p = np.array([np.pi / (2.0 * k)])
    grad_chi_p = cic_interpolate(grad_chi, x_p, L)
    assert np.allclose(grad_chi_p, k, atol=1e-10)


def test_subterm_B_zero_when_E_is_zero() -> None:
    """If E = 0, d/dx(E * H_1(psi_2)) = 0 identically."""
    k = 1.0
    L = 2.0 * np.pi
    x_grid, psi2, h1, h2, h3 = _grids(k, L=L)
    E = np.zeros_like(x_grid)
    grad_E_h1 = grad_1d(E * h1, L)
    assert np.allclose(grad_E_h1, 0.0, atol=1e-14)


def test_subterm_B_with_nontrivial_E() -> None:
    """For E(x) = sin(k x) and h1(psi_2) = -k sin(k x):
        E * h1 = -k sin^2(k x) = -k/2 + k/2 cos(2 k x)
        d/dx (E * h1) = -k^2 sin(2 k x)
    """
    k = 1.0
    L = 2.0 * np.pi
    Nx = 256
    x_grid, psi2, h1, h2, h3 = _grids(k, Nx=Nx, L=L)
    E = np.sin(k * x_grid)
    grad_E_h1 = grad_1d(E * h1, L)
    expected = -(k ** 2) * np.sin(2.0 * k * x_grid)
    assert np.allclose(grad_E_h1, expected, atol=1e-10)


def test_full_position_d2tobs_dx_at_zero_E() -> None:
    """Combine sub-terms A and C (B=0 when E=0) at x = pi/(2k), U=1.
    Expect dx = -gamma * beta * (k^3 + k) * dt < 0.
    """
    k = 1.0
    L = 2.0 * np.pi
    x_grid, psi2, h1, h2, h3 = _grids(k, L=L)
    rho = np.ones_like(x_grid)
    chi = solve_chi(rho, h1, L)
    grad_chi = grad_1d(chi, L)
    x_p = np.array([np.pi / (2.0 * k)])
    U_p = np.array([1.0])
    h3_p = cic_interpolate(h3, x_p, L)
    grad_chi_p = cic_interpolate(grad_chi, x_p, L)
    p2d_term = U_p * U_p * h3_p + 0.0 + grad_chi_p
    gamma, beta, dt = 1.0, 1.0, 0.1
    dx = -gamma * beta * p2d_term * dt
    expected_dx = -(k ** 3 + k) * dt  # = -0.2 for k=1
    assert np.allclose(dx, expected_dx, atol=1e-9)
    assert dx[0] < 0
