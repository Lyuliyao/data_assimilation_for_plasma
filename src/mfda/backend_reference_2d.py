"""2D2V reference PIC backend.

Mirrors backend_reference.py for 1D1V. Particle state is (x, y, vx, vy, w).
Grid is (Nx, Ny) on [0, Lx] x [0, Ly], periodic. CIC bilinear deposition /
interpolation. Symplectic leapfrog with optional time-dependent external
E-field F_ext(x, y, t) = (Ex_ext, Ey_ext).

Same sign convention as 1D: -Laplacian phi = rho - 1, E = -grad phi,
dv/dt = E (i.e. q/m = +1 in normalised units).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .poisson_2d import (
    electric_field_from_density_2d,
    potential_from_density_2d,
)


# E_ext(x_grid, y_grid, t) -> (Ex_grid (Nx,Ny), Ey_grid (Nx,Ny))
EExtFunc2D = Callable[[np.ndarray, np.ndarray, float], tuple[np.ndarray, np.ndarray]]


@dataclass
class ReferenceState2D:
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    w: np.ndarray
    Lx: float
    Ly: float
    Nx: int
    Ny: int
    dt: float
    t: float = 0.0
    Ex: np.ndarray | None = None
    Ey: np.ndarray | None = None
    phi: np.ndarray | None = None
    E_ext_func: EExtFunc2D | None = None


# ---------------------------------------------------------------------------
# CIC deposition / interpolation.
# ---------------------------------------------------------------------------


def _cic_indices_2d(
    x: np.ndarray, y: np.ndarray, Lx: float, Ly: float, Nx: int, Ny: int,
):
    """Return (i0, i1, j0, j1, fx, fy) for bilinear CIC."""
    dx = Lx / Nx
    dy = Ly / Ny
    xi = x / dx
    eta = y / dy
    i0 = np.floor(xi).astype(np.int64)
    j0 = np.floor(eta).astype(np.int64)
    fx = xi - i0
    fy = eta - j0
    i0 = np.mod(i0, Nx)
    j0 = np.mod(j0, Ny)
    i1 = np.mod(i0 + 1, Nx)
    j1 = np.mod(j0 + 1, Ny)
    return i0, i1, j0, j1, fx, fy


def cic_deposit_2d(
    x: np.ndarray, y: np.ndarray, w: np.ndarray,
    Lx: float, Ly: float, Nx: int, Ny: int,
) -> np.ndarray:
    """Deposit weighted particles to a 2D periodic grid via bilinear CIC.

    Returns rho (Nx, Ny) with mean(rho) = total_weight / (Lx * Ly).
    """
    dx = Lx / Nx
    dy = Ly / Ny
    i0, i1, j0, j1, fx, fy = _cic_indices_2d(x, y, Lx, Ly, Nx, Ny)
    minlength = Nx * Ny
    rho_flat = (
        np.bincount(i0 * Ny + j0, weights=w * (1.0 - fx) * (1.0 - fy), minlength=minlength)
        + np.bincount(i1 * Ny + j0, weights=w * fx * (1.0 - fy), minlength=minlength)
        + np.bincount(i0 * Ny + j1, weights=w * (1.0 - fx) * fy, minlength=minlength)
        + np.bincount(i1 * Ny + j1, weights=w * fx * fy, minlength=minlength)
    )
    return rho_flat.reshape(Nx, Ny) / (dx * dy)


def cic_deposit_scalar_2d(
    x: np.ndarray, y: np.ndarray, vals: np.ndarray, w: np.ndarray,
    Lx: float, Ly: float, Nx: int, Ny: int,
) -> np.ndarray:
    """Deposit sum_p (vals_p * w_p) S(x - x_p) S(y - y_p) on a 2D grid."""
    dx = Lx / Nx
    dy = Ly / Ny
    i0, i1, j0, j1, fx, fy = _cic_indices_2d(x, y, Lx, Ly, Nx, Ny)
    vw = vals * w
    minlength = Nx * Ny
    field_flat = (
        np.bincount(i0 * Ny + j0, weights=vw * (1.0 - fx) * (1.0 - fy), minlength=minlength)
        + np.bincount(i1 * Ny + j0, weights=vw * fx * (1.0 - fy), minlength=minlength)
        + np.bincount(i0 * Ny + j1, weights=vw * (1.0 - fx) * fy, minlength=minlength)
        + np.bincount(i1 * Ny + j1, weights=vw * fx * fy, minlength=minlength)
    )
    return field_flat.reshape(Nx, Ny) / (dx * dy)


def cic_interpolate_2d(
    field: np.ndarray, x: np.ndarray, y: np.ndarray, Lx: float, Ly: float,
) -> np.ndarray:
    """Interpolate a 2D grid field to particle positions via bilinear CIC."""
    Nx, Ny = field.shape
    i0, i1, j0, j1, fx, fy = _cic_indices_2d(x, y, Lx, Ly, Nx, Ny)
    return (
        field[i0, j0] * (1.0 - fx) * (1.0 - fy)
        + field[i1, j0] * fx * (1.0 - fy)
        + field[i0, j1] * (1.0 - fx) * fy
        + field[i1, j1] * fx * fy
    )


def normalize_weights_2d(w: np.ndarray, Lx: float, Ly: float) -> np.ndarray:
    """Rescale so mean(rho) = 1."""
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("Non-positive total particle weight.")
    target = Lx * Ly
    return w * (target / total)


# ---------------------------------------------------------------------------
# Field solve + leapfrog.
# ---------------------------------------------------------------------------


def field_solve_2d(state: ReferenceState2D) -> None:
    rho = cic_deposit_2d(
        state.x, state.y, state.w, state.Lx, state.Ly, state.Nx, state.Ny,
    )
    state.phi = potential_from_density_2d(rho, state.Lx, state.Ly)
    state.Ex, state.Ey = electric_field_from_density_2d(rho, state.Lx, state.Ly)


def _eval_E_ext_grid(
    state: ReferenceState2D,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the external E-field on the Poisson grid at state.t."""
    dx = state.Lx / state.Nx
    dy = state.Ly / state.Ny
    x_grid = np.arange(state.Nx) * dx
    y_grid = np.arange(state.Ny) * dy
    return state.E_ext_func(x_grid, y_grid, state.t)


def push_leapfrog_half_2d(state: ReferenceState2D, half: float = 0.5) -> None:
    """Half-kick of (vx, vy) using current (Ex, Ey) plus optional E_ext."""
    if state.Ex is None:
        field_solve_2d(state)
    if state.E_ext_func is not None:
        Ex_ext, Ey_ext = _eval_E_ext_grid(state)
        Ex_total = state.Ex + Ex_ext
        Ey_total = state.Ey + Ey_ext
    else:
        Ex_total = state.Ex
        Ey_total = state.Ey
    Ex_p = cic_interpolate_2d(Ex_total, state.x, state.y, state.Lx, state.Ly)
    Ey_p = cic_interpolate_2d(Ey_total, state.x, state.y, state.Lx, state.Ly)
    state.vx = state.vx + half * state.dt * Ex_p
    state.vy = state.vy + half * state.dt * Ey_p


def push_leapfrog_drift_2d(state: ReferenceState2D) -> None:
    state.x = np.mod(state.x + state.dt * state.vx, state.Lx)
    state.y = np.mod(state.y + state.dt * state.vy, state.Ly)


def step_2d(state: ReferenceState2D) -> None:
    """One symmetric leapfrog step. Nudging is applied between drift and the
    second half-kick by the assimilation loop (mirrors backend_reference.step)."""
    push_leapfrog_half_2d(state, 0.5)
    push_leapfrog_drift_2d(state)
    field_solve_2d(state)
    push_leapfrog_half_2d(state, 0.5)
    state.t += state.dt


def make_state_2d(
    x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray, w: np.ndarray,
    Lx: float, Ly: float, Nx: int, Ny: int, dt: float,
    E_ext_func: EExtFunc2D | None = None,
) -> ReferenceState2D:
    w = normalize_weights_2d(w.copy(), Lx, Ly)
    state = ReferenceState2D(
        x=x.copy(), y=y.copy(), vx=vx.copy(), vy=vy.copy(), w=w,
        Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, dt=dt, E_ext_func=E_ext_func,
    )
    field_solve_2d(state)
    if state.E_ext_func is not None:
        Ex_ext, Ey_ext = _eval_E_ext_grid(state)
        Ex_total = state.Ex + Ex_ext
        Ey_total = state.Ey + Ey_ext
    else:
        Ex_total = state.Ex
        Ey_total = state.Ey
    Ex_p = cic_interpolate_2d(Ex_total, state.x, state.y, state.Lx, state.Ly)
    Ey_p = cic_interpolate_2d(Ey_total, state.x, state.y, state.Lx, state.Ly)
    state.vx = state.vx - 0.5 * state.dt * Ex_p
    state.vy = state.vy - 0.5 * state.dt * Ey_p
    return state
