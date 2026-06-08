"""Sign verification for the velocity_d2tobs term.

Per docs/second_derivative_observation_plan.md eq. for dU:

    dU = ... - 2 gamma * beta * U * H_2(psi_2) * dt

For psi_2(x) = cos(k x), H_2(psi_2)(x) = d^2 psi_2 / dx^2 = -k^2 cos(k x).
At x = 0, H_2 = -k^2 < 0. With a particle at U_p > 0, the velocity
correction is

    dU = -2 * gamma * beta * U_p * (-k^2) * dt = +2 gamma beta U_p k^2 dt > 0

So U increases, i.e. the particle is accelerated. This is the doc's
"anti-drag" case (locally concave psi_2). The opposite x position
(x = pi/k, where cos(kx) = -1) has H_2 > 0 and the particle is decelerated.

This test exercises the math directly without driving a full psi_2 solve.
"""
from __future__ import annotations

import numpy as np

from mfda.backend_reference import cic_interpolate
from mfda.poisson import grad_1d


def test_velocity_d2tobs_sign_at_concave_psi2() -> None:
    Nx = 128
    L = 2.0 * np.pi
    x_grid = np.linspace(0.0, L, Nx, endpoint=False)
    k = 1.0
    psi2 = np.cos(k * x_grid)
    h2 = grad_1d(grad_1d(psi2, L), L)
    expected_h2 = -(k ** 2) * np.cos(k * x_grid)
    assert np.allclose(h2, expected_h2, atol=1e-12)

    # Particle at x=0 (locally concave: cos(kx) > 0, so h2 < 0), U > 0.
    x_p = np.array([0.0, np.pi])  # second particle at x=pi (locally convex)
    U_p = np.array([1.5, 1.5])
    h2_p = cic_interpolate(h2, x_p, L)
    # h2_p[0] should be -k^2 = -1 (concave); h2_p[1] should be +1 (convex)
    assert h2_p[0] < 0
    assert h2_p[1] > 0

    gamma, beta, dt = 1.0, 1.0, 0.1
    # The minus sign in `assim.v = assim.v - dv_total * dt` is applied by
    # the assim loop, so dv_total carries the leading +2 gamma beta U H2.
    dv_total = 2.0 * gamma * beta * U_p * h2_p
    dU = -dv_total * dt  # what the loop actually applies
    # Particle 0 (concave psi_2): dU > 0 (accelerated)
    # Particle 1 (convex psi_2): dU < 0 (decelerated, viscous drag)
    assert dU[0] > 0, f"concave psi2: expected dU>0, got {dU[0]}"
    assert dU[1] < 0, f"convex psi2: expected dU<0, got {dU[1]}"


def test_velocity_d2tobs_zero_when_psi2_linear() -> None:
    """If psi_2 is linear in x, H_2 = 0 and the correction vanishes."""
    Nx = 64
    L = 2.0 * np.pi
    x_grid = np.linspace(0.0, L, Nx, endpoint=False)
    # A linear-in-x psi_2 isn't periodic; use psi_2 = sin(x) which has h2 = -sin(x)
    # — at x = pi/2 the second derivative is -1, not zero. Better: cos(0*x) = 1
    # constant -> h2 = 0 everywhere.
    psi2 = np.ones(Nx)
    h2 = grad_1d(grad_1d(psi2, L), L)
    assert np.allclose(h2, 0.0, atol=1e-12)
    x_p = np.linspace(0, L, 5, endpoint=False)
    h2_p = cic_interpolate(h2, x_p, L)
    dv = 2.0 * 1.0 * 1.0 * np.ones_like(x_p) * h2_p * 0.1
    assert np.allclose(dv, 0.0, atol=1e-12)


def test_velocity_d2tobs_proportional_to_U() -> None:
    """dU scales linearly in U at the same x."""
    Nx = 64
    L = 2.0 * np.pi
    x_grid = np.linspace(0.0, L, Nx, endpoint=False)
    psi2 = np.cos(2 * x_grid)
    h2 = grad_1d(grad_1d(psi2, L), L)
    x_p = np.array([0.5, 1.0, 1.5])
    h2_p = cic_interpolate(h2, x_p, L)
    dv1 = 2.0 * 1.0 * 1.0 * np.ones_like(x_p) * h2_p * 0.1
    dv2 = 2.0 * 1.0 * 1.0 * (3.0 * np.ones_like(x_p)) * h2_p * 0.1
    assert np.allclose(dv2, 3.0 * dv1, atol=1e-13)
