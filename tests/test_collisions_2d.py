"""Verify 2D BGK + LB collision substeps."""
from __future__ import annotations

import numpy as np

from mfda.backend_reference_2d import make_state_2d
from mfda.collisions_2d import bgk_substep_2d, lb_substep_2d
from mfda.diagnostics_2d import grid_moments_2d2v


def _state(rng, Lx, Ly, Nx, Ny, Np, sigma=1.0, ux0=0.0, uy0=0.0, dt=0.1):
    x = rng.uniform(0.0, Lx, Np)
    y = rng.uniform(0.0, Ly, Np)
    vx = sigma * rng.standard_normal(Np) + ux0
    vy = sigma * rng.standard_normal(Np) + uy0
    w = np.ones(Np)
    return make_state_2d(x, y, vx, vy, w, Lx, Ly, Nx, Ny, dt)


def test_bgk_2d_conserves_mass_exactly() -> None:
    rng = np.random.default_rng(0)
    state = _state(rng, 2 * np.pi, 2 * np.pi, 32, 32, 5000)
    m0 = float(state.w.sum())
    rng_b = np.random.default_rng(1)
    for _ in range(20):
        bgk_substep_2d(state, nu=1.0, rng=rng_b)
    assert state.w.sum() == m0


def test_lb_2d_conserves_mass_exactly() -> None:
    rng = np.random.default_rng(0)
    state = _state(rng, 2 * np.pi, 2 * np.pi, 32, 32, 5000)
    m0 = float(state.w.sum())
    rng_l = np.random.default_rng(1)
    for _ in range(20):
        lb_substep_2d(state, nu=1.0, rng=rng_l)
    assert state.w.sum() == m0


def test_lb_zero_nu_is_noop_2d() -> None:
    rng = np.random.default_rng(0)
    state = _state(rng, 2 * np.pi, 2 * np.pi, 32, 32, 1000)
    vx_before = state.vx.copy()
    vy_before = state.vy.copy()
    lb_substep_2d(state, nu=0.0, rng=np.random.default_rng(1))
    assert np.array_equal(state.vx, vx_before)
    assert np.array_equal(state.vy, vy_before)


def test_bgk_2d_homogeneous_maxwellian_fixed_point() -> None:
    """Homogeneous Maxwellian at (ux*, uy*, T*) stays so on average under BGK."""
    rng = np.random.default_rng(0)
    Lx = Ly = 2 * np.pi
    Nx = Ny = 32
    Np = 200_000
    ux0, uy0, sigma = 0.3, -0.2, np.sqrt(1.4)
    state = _state(rng, Lx, Ly, Nx, Ny, Np, sigma=sigma, ux0=ux0, uy0=uy0, dt=0.1)
    nu = 1.0
    rng_b = np.random.default_rng(7)
    n_steps = int(np.ceil(5.0 / (nu * state.dt)))
    for _ in range(n_steps):
        bgk_substep_2d(state, nu, rng_b)
    rho, ux, uy, T = grid_moments_2d2v(state)
    ux_bulk = float(np.sum(rho * ux) / np.sum(rho))
    uy_bulk = float(np.sum(rho * uy) / np.sum(rho))
    T_bulk = float(np.sum(rho * T) / np.sum(rho))
    assert abs(ux_bulk - ux0) < 0.02 * abs(ux0) + 5e-3
    assert abs(uy_bulk - uy0) < 0.02 * abs(uy0) + 5e-3
    assert abs(T_bulk - 1.4) / 1.4 < 0.02


def test_lb_2d_homogeneous_maxwellian_fixed_point() -> None:
    rng = np.random.default_rng(0)
    Lx = Ly = 2 * np.pi
    Nx = Ny = 32
    Np = 200_000
    ux0, uy0, sigma = 0.3, -0.2, np.sqrt(1.4)
    state = _state(rng, Lx, Ly, Nx, Ny, Np, sigma=sigma, ux0=ux0, uy0=uy0, dt=0.1)
    nu = 1.0
    rng_l = np.random.default_rng(7)
    n_steps = int(np.ceil(5.0 / (nu * state.dt)))
    for _ in range(n_steps):
        lb_substep_2d(state, nu, rng_l)
    rho, ux, uy, T = grid_moments_2d2v(state)
    ux_bulk = float(np.sum(rho * ux) / np.sum(rho))
    uy_bulk = float(np.sum(rho * uy) / np.sum(rho))
    T_bulk = float(np.sum(rho * T) / np.sum(rho))
    assert abs(ux_bulk - ux0) < 0.02 * abs(ux0) + 5e-3
    assert abs(uy_bulk - uy0) < 0.02 * abs(uy0) + 5e-3
    assert abs(T_bulk - 1.4) / 1.4 < 0.02
