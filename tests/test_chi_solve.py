"""Unit tests for poisson.solve_chi.

The chi adjoint solves
    -Delta chi = grad . (rho grad psi_2)

For psi_2 = cos(k x), rho = 1:
    grad psi_2 = -k sin(k x)
    rho * grad psi_2 = -k sin(k x)
    grad . (rho grad psi_2) = d/dx (-k sin(k x)) = -k^2 cos(k x)
    -Delta chi = -k^2 cos(k x)
    chi'' = k^2 cos(k x)
    chi = -cos(k x)        (mean-free periodic solution)
"""
from __future__ import annotations

import numpy as np

from mfda.poisson import grad_1d, solve_chi


def test_chi_recovers_analytic_for_cos_psi2() -> None:
    Nx = 128
    L = 2.0 * np.pi
    x = np.linspace(0.0, L, Nx, endpoint=False)
    for k_mode in (1, 2, 3):
        kx = k_mode
        psi2 = np.cos(kx * x)
        grad_psi2 = grad_1d(psi2, L)  # should be -kx sin(kx x)
        rho = np.ones(Nx)
        chi = solve_chi(rho, grad_psi2, L)
        analytic = -np.cos(kx * x)
        assert np.allclose(chi, analytic, atol=1e-12), f"k_mode={k_mode}"


def test_chi_zero_when_psi2_is_constant() -> None:
    Nx = 64
    L = 1.0
    rho = np.ones(Nx) * 1.5
    grad_psi2 = np.zeros(Nx)  # psi_2 constant -> zero gradient
    chi = solve_chi(rho, grad_psi2, L)
    assert np.allclose(chi, 0.0, atol=1e-14)
