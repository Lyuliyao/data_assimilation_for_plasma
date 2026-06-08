"""Collision substeps for the reference PIC backend.

This module currently implements the **BGK relaxation operator** (note v3
§2.1, eq. (with M[f] denoting the local Maxwellian)):

    Q_BGK(f) = nu * (M[f] - f).

In a PIC code BGK is applied as a Monte-Carlo substep: at each timestep,
each particle is replaced (with probability p = 1 - exp(-nu * dt)) by a
fresh draw from the *local* Maxwellian M[f](x_p, .). The Maxwellian is
parameterised by the local hydrodynamic moments (rho, u, T) computed from
the particle distribution itself — this is exactly what makes BGK
non-linear and is also why it preserves mass, momentum, and energy
*on average* (a single MC realisation has stochastic fluctuations).

Conventions
-----------
- 1D1V (the reference backend's setting). Velocity is a scalar v.
- Local moments are deposited on the Poisson grid via CIC and interpolated
  back to particle positions, matching the rest of the backend.
- The temperature is floored at T_floor (default 1e-3 in plasma units) to
  avoid numerical issues in low-density cells where the empirical T can
  be small / negative due to particle noise.

Why MC-BGK vs. deterministic relaxation
---------------------------------------
Two implementations are common:
  1. MC ('replacement'): with probability p, replace v with a sample from
     M[f]. Stochastic but bias-free for any dt; standard in PIC codes.
  2. Deterministic relaxation: v <- v + p * (sample - v) with sample drawn
     once per particle. Lower variance but biased at large p.
We use option 1 for cleanest fidelity to the BGK ODE; the variance is
absorbed by Np ~ 1e5-1e6.
"""
from __future__ import annotations

import numpy as np

from .backend_reference import (
    ReferenceState,
    cic_deposit,
    cic_deposit_current,
    cic_interpolate,
)


T_FLOOR_DEFAULT = 1.0e-3
RHO_FLOOR_DEFAULT = 1.0e-3


def grid_moments_1d1v(
    state: ReferenceState,
    rho_floor: float = RHO_FLOOR_DEFAULT,
    T_floor: float = T_FLOOR_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute local (rho, u, T) on the Poisson grid via CIC moments.

    Returns
    -------
    rho_g : (Nx,) — particle density (background-independent: just the moment).
    u_g   : (Nx,) — bulk velocity = j_g / max(rho_g, rho_floor).
    T_g   : (Nx,) — temperature, computed from the second moment with the
                    bulk-velocity contribution removed:
                       T_g = max(<v^2>_g - u_g^2, T_floor)
                    where <v^2>_g = M_g / max(rho_g, rho_floor) and M_g is the
                    CIC-deposited second velocity moment.
    """
    L = state.L
    Nx = state.Nx
    rho_g = cic_deposit(state.x, state.w, L, Nx)
    j_g = cic_deposit_current(state.x, state.v, state.w, L, Nx)
    # Second-moment deposit. We avoid pulling kinetic_stress here to keep
    # this module self-contained; the operation is identical to
    # cic_deposit_current with weight v^2 instead of v.
    dx = L / Nx
    xi = state.x / dx
    i0 = np.floor(xi).astype(np.int64)
    frac = xi - i0
    i0 = np.mod(i0, Nx)
    i1 = np.mod(i0 + 1, Nx)
    v2w = state.v * state.v * state.w
    M_g = (np.bincount(i0, weights=v2w * (1.0 - frac), minlength=Nx)
           + np.bincount(i1, weights=v2w * frac, minlength=Nx))
    M_g /= dx
    rho_safe = np.maximum(rho_g, rho_floor)
    u_g = j_g / rho_safe
    var_g = M_g / rho_safe - u_g * u_g
    T_g = np.maximum(var_g, T_floor)
    return rho_g, u_g, T_g


def bgk_substep(
    state: ReferenceState,
    nu: float,
    rng: np.random.Generator,
    rho_floor: float = RHO_FLOOR_DEFAULT,
    T_floor: float = T_FLOOR_DEFAULT,
) -> None:
    """Apply one Monte-Carlo BGK collision substep in-place.

    For each particle, with probability p = 1 - exp(-nu * dt) replace its
    velocity by a fresh sample from N(u_local, T_local). The local moments
    are evaluated at the particle's current position via CIC interpolation
    of grid moments computed from the *same* state (BGK is non-linear in f).

    Mass-, momentum-, and energy-conservation hold in expectation; finite-Np
    realisations have O(1/sqrt(Np)) fluctuations.
    """
    if nu <= 0.0:
        return
    dt = state.dt
    rho_g, u_g, T_g = grid_moments_1d1v(state, rho_floor, T_floor)
    u_p = cic_interpolate(u_g, state.x, state.L)
    T_p = cic_interpolate(T_g, state.x, state.L)
    # Clamp interpolated T (CIC can mildly under-shoot near sharp gradients).
    T_p = np.maximum(T_p, T_floor)
    p_replace = 1.0 - np.exp(-nu * dt)
    u = rng.uniform(size=state.x.shape[0])
    mask = u < p_replace
    if not mask.any():
        return
    # Resample velocity from local Maxwellian for masked particles.
    sigma_p = np.sqrt(T_p[mask])
    new_v = u_p[mask] + sigma_p * rng.standard_normal(int(mask.sum()))
    state.v[mask] = new_v


def lb_substep(
    state: ReferenceState,
    nu: float,
    rng: np.random.Generator,
    rho_floor: float = RHO_FLOOR_DEFAULT,
    T_floor: float = T_FLOOR_DEFAULT,
) -> None:
    """Apply one Lenard-Bernstein collision substep in-place.

    The LB operator is the Fokker-Planck-type drift-diffusion

        Q_LB(f) = nu * d_v[(v - u_local) f + T_local d_v f],

    whose stationary distribution is the local Maxwellian M[f]. In a PIC
    code we realise it as the Langevin equation

        dv = -nu (v - u_local) dt + sqrt(2 nu T_local) dW.

    For constant (u_local, T_local) over a step dt this is an Ornstein-
    Uhlenbeck process whose exact discrete update is

        v_new = u_local + (v - u_local) * exp(-nu*dt)
                       + sqrt(T_local * (1 - exp(-2*nu*dt))) * Z,
        Z ~ N(0, 1).

    We use this exact OU step rather than Euler-Maruyama so the scheme is
    accurate at any nu*dt. Like BGK, mass/momentum/energy conservation is
    only in expectation.
    """
    if nu <= 0.0:
        return
    dt = state.dt
    rho_g, u_g, T_g = grid_moments_1d1v(state, rho_floor, T_floor)
    u_p = cic_interpolate(u_g, state.x, state.L)
    T_p = cic_interpolate(T_g, state.x, state.L)
    T_p = np.maximum(T_p, T_floor)
    decay = float(np.exp(-nu * dt))
    diffusion_var = T_p * (1.0 - decay * decay)
    z = rng.standard_normal(state.v.shape[0])
    state.v[:] = u_p + (state.v - u_p) * decay + np.sqrt(diffusion_var) * z
