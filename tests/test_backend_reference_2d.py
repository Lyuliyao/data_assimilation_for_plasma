"""Verify 2D2V CIC deposit/interpolate roundtrip + leapfrog conservation."""
from __future__ import annotations

import numpy as np

from mfda.backend_reference_2d import (
    cic_deposit_2d,
    cic_deposit_scalar_2d,
    cic_interpolate_2d,
    field_solve_2d,
    make_state_2d,
    push_leapfrog_drift_2d,
    push_leapfrog_half_2d,
    step_2d,
)
from mfda.diagnostics_2d import grid_moments_2d2v


def _rng_state(rng, Lx, Ly, Nx, Ny, Np, dt=0.05, sigma=1.0, u0=(0.0, 0.0)):
    x = rng.uniform(0.0, Lx, Np)
    y = rng.uniform(0.0, Ly, Np)
    vx = sigma * rng.standard_normal(Np) + u0[0]
    vy = sigma * rng.standard_normal(Np) + u0[1]
    w = np.ones(Np)
    return make_state_2d(x, y, vx, vy, w, Lx, Ly, Nx, Ny, dt)


def test_cic_deposit_total_mass_preserved() -> None:
    rng = np.random.default_rng(0)
    Lx = Ly = 2.0 * np.pi
    Nx = Ny = 32
    Np = 1000
    x = rng.uniform(0.0, Lx, Np)
    y = rng.uniform(0.0, Ly, Np)
    w = np.ones(Np)
    rho = cic_deposit_2d(x, y, w, Lx, Ly, Nx, Ny)
    cell_area = (Lx / Nx) * (Ly / Ny)
    total_mass = rho.sum() * cell_area
    assert np.isclose(total_mass, w.sum())


def test_cic_interpolate_constant_field() -> None:
    Nx = Ny = 32
    Lx = Ly = 2.0 * np.pi
    field = np.full((Nx, Ny), 3.14)
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, Lx, 200)
    y = rng.uniform(0.0, Ly, 200)
    vals = cic_interpolate_2d(field, x, y, Lx, Ly)
    np.testing.assert_allclose(vals, 3.14, atol=1e-12)


def test_cic_interpolate_linear_in_x() -> None:
    """Interpolating a field that is linear in x recovers x at each particle
    (within one CIC cell on the boundary)."""
    Nx = Ny = 64
    Lx = Ly = 2.0 * np.pi
    x_grid = np.arange(Nx) * Lx / Nx
    field = np.broadcast_to(x_grid[:, None], (Nx, Ny)).copy()
    rng = np.random.default_rng(0)
    Np = 500
    # Avoid the wrap region where periodic interpolation collapses x ~ L back to 0.
    x = rng.uniform(0.0, Lx - Lx / Nx, Np)
    y = rng.uniform(0.0, Ly, Np)
    vals = cic_interpolate_2d(field, x, y, Lx, Ly)
    # CIC on a linear field is exact.
    np.testing.assert_allclose(vals, x, atol=Lx / Nx * 1e-10)


def test_field_solve_uniform_distribution_low_E() -> None:
    """Uniform 2D Poisson source gives Ex, Ey dominated by PIC shot noise (small)."""
    rng = np.random.default_rng(0)
    Lx = Ly = 2.0 * np.pi
    Nx = Ny = 32
    Np = 100_000
    state = _rng_state(rng, Lx, Ly, Nx, Ny, Np)
    field_solve_2d(state)
    # Mean |E| should be small compared to typical thermal speed (~1).
    assert float(np.mean(np.abs(state.Ex))) < 0.05
    assert float(np.mean(np.abs(state.Ey))) < 0.05


def test_grid_moments_recover_uniform_maxwellian() -> None:
    """A spatially uniform 2D Maxwellian with sigma=1 gives rho ~ 1, u ~ 0, T ~ 1."""
    rng = np.random.default_rng(0)
    Lx = Ly = 2.0 * np.pi
    Nx = Ny = 32
    Np = 200_000
    state = _rng_state(rng, Lx, Ly, Nx, Ny, Np, sigma=1.0, u0=(0.0, 0.0))
    rho, ux, uy, T = grid_moments_2d2v(state)
    rho_bulk = float(rho.mean())
    ux_bulk = float(np.sum(rho * ux) / np.sum(rho))
    uy_bulk = float(np.sum(rho * uy) / np.sum(rho))
    T_bulk = float(np.sum(rho * T) / np.sum(rho))
    assert abs(rho_bulk - 1.0) < 0.02
    assert abs(ux_bulk) < 0.02
    assert abs(uy_bulk) < 0.02
    assert abs(T_bulk - 1.0) < 0.02


def test_leapfrog_step_preserves_total_mass() -> None:
    """Step 10 leapfrog updates -- total particle weight must be invariant."""
    rng = np.random.default_rng(0)
    Lx = Ly = 2.0 * np.pi
    Nx = Ny = 32
    Np = 5000
    state = _rng_state(rng, Lx, Ly, Nx, Ny, Np)
    m0 = float(state.w.sum())
    for _ in range(10):
        step_2d(state)
    assert state.w.sum() == m0
