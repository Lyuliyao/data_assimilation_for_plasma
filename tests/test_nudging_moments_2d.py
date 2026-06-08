"""Verify 2D2V nudging A_var + metriplectic B."""
from __future__ import annotations

import numpy as np

from mfda.backend_reference_2d import make_state_2d, normalize_weights_2d
from mfda.diagnostics_2d import grid_moments_2d2v
from mfda.nudging_moments_2d import (
    FormulationAVariantParams2D,
    FormulationBParams2D,
    apply_formulation_A_variant_2d,
    apply_formulation_B_2d,
)


def _state_with_assim_obs_match():
    """Build assim particles, read its own moments back as 'obs'.
    Then any A_var / B residual is identically zero."""
    rng = np.random.default_rng(0)
    Lx = Ly = 2.0 * np.pi
    Nx = Ny = 32
    Np = 50_000
    ux_star, uy_star, T_star = 0.2, -0.15, 1.0
    x = rng.uniform(0.0, Lx, Np)
    y = rng.uniform(0.0, Ly, Np)
    sigma = np.sqrt(T_star)
    vx = ux_star + sigma * rng.standard_normal(Np)
    vy = uy_star + sigma * rng.standard_normal(Np)
    w = np.ones(Np)
    state = make_state_2d(x, y, vx, vy, w, Lx, Ly, Nx, Ny, dt=0.1)
    rho_obs, ux_obs, uy_obs, T_obs = grid_moments_2d2v(state)
    return state, rho_obs, ux_obs, uy_obs, T_obs, Lx, Ly, Nx, Ny


def test_A_variant_2d_zero_residual_no_update() -> None:
    state, rho_o, ux_o, uy_o, T_o, Lx, Ly, Nx, Ny = _state_with_assim_obs_match()
    params = FormulationAVariantParams2D(lowpass_k_cut_frac=1.0)
    x_new, y_new, vx_new, vy_new = apply_formulation_A_variant_2d(
        state.x, state.y, state.vx, state.vy, state.w, Lx, Ly, Nx, Ny,
        rho_o, ux_o, uy_o, T_o, params, dt=0.05,
    )
    assert np.allclose(x_new, state.x, atol=1e-8)
    assert np.allclose(y_new, state.y, atol=1e-8)
    assert np.allclose(vx_new, state.vx, atol=1e-8)
    assert np.allclose(vy_new, state.vy, atol=1e-8)


def test_B_2d_zero_residual_no_update() -> None:
    state, rho_o, ux_o, uy_o, T_o, Lx, Ly, Nx, Ny = _state_with_assim_obs_match()
    params = FormulationBParams2D(lowpass_k_cut_frac=1.0)
    x_new, y_new, vx_new, vy_new = apply_formulation_B_2d(
        state.x, state.y, state.vx, state.vy, state.w, Lx, Ly, Nx, Ny,
        rho_o, ux_o, uy_o, T_o, params, dt=0.05,
    )
    assert np.allclose(x_new, state.x, atol=1e-8)
    assert np.allclose(y_new, state.y, atol=1e-8)
    assert np.allclose(vx_new, state.vx, atol=1e-8)
    assert np.allclose(vy_new, state.vy, atol=1e-8)


def test_2d_smoke_shapes_finite() -> None:
    rng = np.random.default_rng(0)
    Lx = Ly = 2.0 * np.pi
    Nx = Ny = 16
    Np = 500
    x = rng.uniform(0.0, Lx, Np)
    y = rng.uniform(0.0, Ly, Np)
    vx = rng.standard_normal(Np)
    vy = rng.standard_normal(Np)
    w = normalize_weights_2d(np.ones(Np), Lx, Ly)
    # Synthetic obs with mild structure.
    x_g = np.arange(Nx) * Lx / Nx
    y_g = np.arange(Ny) * Ly / Ny
    XG, YG = np.meshgrid(x_g, y_g, indexing="ij")
    rho_obs = 1.0 + 0.1 * np.cos(XG + YG)
    ux_obs = 0.05 * np.sin(XG)
    uy_obs = 0.05 * np.sin(YG)
    T_obs = np.ones((Nx, Ny))
    xa, ya, vxa, vya = apply_formulation_A_variant_2d(
        x, y, vx, vy, w, Lx, Ly, Nx, Ny,
        rho_obs, ux_obs, uy_obs, T_obs, FormulationAVariantParams2D(), dt=0.05,
    )
    assert np.all(np.isfinite(xa)) and np.all(np.isfinite(ya))
    assert np.all(np.isfinite(vxa)) and np.all(np.isfinite(vya))
    xb, yb, vxb, vyb = apply_formulation_B_2d(
        x, y, vx, vy, w, Lx, Ly, Nx, Ny,
        rho_obs, ux_obs, uy_obs, T_obs, FormulationBParams2D(), dt=0.05,
    )
    assert np.all(np.isfinite(xb)) and np.all(np.isfinite(yb))
    assert np.all(np.isfinite(vxb)) and np.all(np.isfinite(vyb))
