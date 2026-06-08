"""Collision substeps for the 2D2V reference backend.

Mirrors collisions.py for 1D1V. The local hydrodynamic moments are now
(rho, ux, uy, T) on the 2D grid; the local Maxwellian is the isotropic 2D
Gaussian with mean (ux, uy) and variance T per component.

Two operators:
  - bgk_substep_2d : MC replacement, v_p <- N((u_local), T_local I_2)
                     with probability p = 1 - exp(-nu * dt).
  - lb_substep_2d  : exact-OU Langevin step on (vx, vy):
                       v_new_alpha = u_local_alpha
                                     + (v_alpha - u_local_alpha) * exp(-nu*dt)
                                     + sqrt(T_local * (1 - exp(-2 nu dt))) * Z_alpha,
                     Z_x, Z_y ~ N(0, 1) independent.

Both preserve mass exactly (no position update); momentum and energy on
average. Same equilibrium = isotropic local Maxwellian.
"""
from __future__ import annotations

import numpy as np

from .backend_reference_2d import (
    ReferenceState2D,
    cic_interpolate_2d,
)
from .diagnostics_2d import (
    RHO_FLOOR_DEFAULT,
    T_FLOOR_DEFAULT,
    grid_moments_2d2v,
)


def bgk_substep_2d(
    state: ReferenceState2D,
    nu: float,
    rng: np.random.Generator,
    rho_floor: float = RHO_FLOOR_DEFAULT,
    T_floor: float = T_FLOOR_DEFAULT,
) -> None:
    """Apply one MC BGK substep in-place on (vx, vy)."""
    if nu <= 0.0:
        return
    dt = state.dt
    rho_g, ux_g, uy_g, T_g = grid_moments_2d2v(state, rho_floor, T_floor)
    ux_p = cic_interpolate_2d(ux_g, state.x, state.y, state.Lx, state.Ly)
    uy_p = cic_interpolate_2d(uy_g, state.x, state.y, state.Lx, state.Ly)
    T_p = cic_interpolate_2d(T_g, state.x, state.y, state.Lx, state.Ly)
    T_p = np.maximum(T_p, T_floor)
    p_replace = 1.0 - np.exp(-nu * dt)
    u = rng.uniform(size=state.vx.shape[0])
    mask = u < p_replace
    if not mask.any():
        return
    sigma_p = np.sqrt(T_p[mask])
    n_mask = int(mask.sum())
    state.vx[mask] = ux_p[mask] + sigma_p * rng.standard_normal(n_mask)
    state.vy[mask] = uy_p[mask] + sigma_p * rng.standard_normal(n_mask)


def lb_substep_2d(
    state: ReferenceState2D,
    nu: float,
    rng: np.random.Generator,
    rho_floor: float = RHO_FLOOR_DEFAULT,
    T_floor: float = T_FLOOR_DEFAULT,
) -> None:
    """Apply one exact-OU LB (Lenard-Bernstein / Dougherty) substep on (vx, vy)."""
    if nu <= 0.0:
        return
    dt = state.dt
    rho_g, ux_g, uy_g, T_g = grid_moments_2d2v(state, rho_floor, T_floor)
    ux_p = cic_interpolate_2d(ux_g, state.x, state.y, state.Lx, state.Ly)
    uy_p = cic_interpolate_2d(uy_g, state.x, state.y, state.Lx, state.Ly)
    T_p = cic_interpolate_2d(T_g, state.x, state.y, state.Lx, state.Ly)
    T_p = np.maximum(T_p, T_floor)
    decay = float(np.exp(-nu * dt))
    diffusion_var = T_p * (1.0 - decay * decay)
    sigma_diff = np.sqrt(diffusion_var)
    Np = state.vx.shape[0]
    zx = rng.standard_normal(Np)
    zy = rng.standard_normal(Np)
    state.vx[:] = ux_p + (state.vx - ux_p) * decay + sigma_diff * zx
    state.vy[:] = uy_p + (state.vy - uy_p) * decay + sigma_diff * zy
