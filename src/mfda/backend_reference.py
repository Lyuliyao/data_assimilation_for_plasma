"""Numpy 1D1V ES-PIC reference backend.

Purpose
-------
A tiny, hackable leapfrog electrostatic PIC used for:
  - unit tests,
  - algorithm development of the nudging kernels,
  - end-to-end smoke runs in CI.

It is NOT the production backend. Use WarpX (backend_warpx.py) for the
experimental campaign, especially when scaling up or extending to 2D/3D.

Conventions
-----------
- Periodic x-domain [0, L), uniform Poisson grid of Nx cells with dx = L / Nx.
- CIC deposition and interpolation.
- Leapfrog integrator. Velocity is offset by -dt/2 at t=0 to initialise the
  leapfrog.
- Units: m_e = q_e = 1, omega_p = 1, ion background = 1.
- Particle array shapes: x (Np,), v (Np,), w (Np,).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .poisson import electric_field_from_density, potential_from_density


# Type alias for a time-dependent external E-field on the Poisson grid.
# Callable signature: E_ext_func(x_grid, t) -> E_ext_grid (same shape as x_grid).
EExtFunc = Callable[[np.ndarray, float], np.ndarray]


@dataclass
class ReferenceState:
    x: np.ndarray
    v: np.ndarray
    w: np.ndarray
    L: float
    Nx: int
    dt: float
    t: float = 0.0
    # Cached last E and phi, valid after a deposit-solve.
    E: np.ndarray | None = None
    phi: np.ndarray | None = None
    # Optional time-dependent external E-field (note v3 §2 setup with driving).
    # If set, the value is evaluated on the Poisson grid at the half-kick time
    # and ADDED to the self-consistent E_p before the velocity update. This
    # leaves state.E (the self-consistent field) untouched — clean separation.
    E_ext_func: EExtFunc | None = None


def cic_deposit(x: np.ndarray, w: np.ndarray, L: float, Nx: int) -> np.ndarray:
    """Deposit weighted particles to a periodic grid with CIC (linear) shape.

    Returns rho such that mean(rho) = total weight / L.

    Uses np.bincount for the scatter add (C-implemented, ~5-10x faster
    than the previous np.add.at).
    """
    dx = L / Nx
    xi = x / dx
    i0 = np.floor(xi).astype(np.int64)
    frac = xi - i0
    i0 = np.mod(i0, Nx)
    i1 = np.mod(i0 + 1, Nx)
    rho = (np.bincount(i0, weights=w * (1.0 - frac), minlength=Nx)
           + np.bincount(i1, weights=w * frac, minlength=Nx))
    rho /= dx
    return rho


def cic_deposit_current(
    x: np.ndarray, v: np.ndarray, w: np.ndarray, L: float, Nx: int,
) -> np.ndarray:
    """Deposit weighted particle current j = sum_p v_p w_p S(x - x_p) on a periodic grid.

    Same CIC shape as cic_deposit but the per-particle weight is v * w instead
    of w. Returns j on the same Nx-cell grid as the density.
    """
    dx = L / Nx
    xi = x / dx
    i0 = np.floor(xi).astype(np.int64)
    frac = xi - i0
    i0 = np.mod(i0, Nx)
    i1 = np.mod(i0 + 1, Nx)
    vw = v * w
    j = (np.bincount(i0, weights=vw * (1.0 - frac), minlength=Nx)
         + np.bincount(i1, weights=vw * frac, minlength=Nx))
    j /= dx
    return j


def cic_interpolate(field: np.ndarray, x: np.ndarray, L: float) -> np.ndarray:
    """Interpolate a grid field to particle positions with CIC."""
    Nx = field.shape[0]
    dx = L / Nx
    xi = x / dx
    i0 = np.floor(xi).astype(np.int64)
    frac = xi - i0
    i0 = np.mod(i0, Nx)
    i1 = np.mod(i0 + 1, Nx)
    return field[i0] * (1.0 - frac) + field[i1] * frac


def normalize_weights(w: np.ndarray, L: float) -> np.ndarray:
    """Rescale weights so that the mean density equals 1.0 (neutral background)."""
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("Non-positive total particle weight.")
    target = L  # since mean(rho) = total_weight / L should equal 1
    return w * (target / total)


def field_solve(state: ReferenceState) -> None:
    """Deposit rho, solve Poisson, cache phi and E on the state."""
    rho = cic_deposit(state.x, state.w, state.L, state.Nx)
    state.phi = potential_from_density(rho, state.L)
    state.E = electric_field_from_density(rho, state.L)


def push_leapfrog_half(state: ReferenceState, half: float = 0.5) -> None:
    """Half-kick of the velocity using the current E (plus optional E_ext).

    If state.E_ext_func is set, the external field is evaluated on the grid
    at the current state.t and added to E before interpolation to particles.
    The cached state.E (self-consistent only) is not modified.
    """
    if state.E is None:
        field_solve(state)
    if state.E_ext_func is not None:
        dx = state.L / state.Nx
        x_grid = np.arange(state.Nx) * dx
        E_total_grid = state.E + state.E_ext_func(x_grid, state.t)
        E_p = cic_interpolate(E_total_grid, state.x, state.L)
    else:
        E_p = cic_interpolate(state.E, state.x, state.L)
    state.v = state.v + half * state.dt * E_p


def push_leapfrog_drift(state: ReferenceState) -> None:
    """Full drift of the positions."""
    state.x = np.mod(state.x + state.dt * state.v, state.L)


def step(state: ReferenceState) -> None:
    """One symmetric leapfrog step: half-kick, drift, field solve, half-kick.

    Nudging should be applied *between* the drift and the second half-kick by
    the assimilation loop, i.e. after the field solve and before the next
    velocity update.
    """
    push_leapfrog_half(state, 0.5)
    push_leapfrog_drift(state)
    field_solve(state)
    # Second half-kick is done by the assim loop AFTER any nudging
    push_leapfrog_half(state, 0.5)
    state.t += state.dt


def make_state(
    x: np.ndarray, v: np.ndarray, w: np.ndarray,
    L: float, Nx: int, dt: float,
    E_ext_func: EExtFunc | None = None,
) -> ReferenceState:
    w = normalize_weights(w.copy(), L)
    state = ReferenceState(
        x=x.copy(), v=v.copy(), w=w, L=L, Nx=Nx, dt=dt,
        E_ext_func=E_ext_func,
    )
    field_solve(state)
    # Offset velocity by -dt/2 for leapfrog start. Include E_ext at t=0 if set.
    if state.E_ext_func is not None:
        dx = L / Nx
        x_grid = np.arange(Nx) * dx
        E_total_grid = state.E + state.E_ext_func(x_grid, state.t)
        E_p = cic_interpolate(E_total_grid, state.x, state.L)
    else:
        E_p = cic_interpolate(state.E, state.x, state.L)
    state.v = state.v - 0.5 * state.dt * E_p
    return state
