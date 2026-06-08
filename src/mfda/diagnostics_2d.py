"""2D2V diagnostic helpers: grid moments + ABC error metrics.

Mirrors the 1D `collisions.grid_moments_1d1v` and `diagnostics` routines.
For d=2, the isotropic temperature is

    T = (1/2) (<vx^2> + <vy^2>) - (1/2) (ux^2 + uy^2),

so a Maxwellian f(vx, vy) ~ exp(-(vx^2 + vy^2) / (2 T)) gives mean
<vx^2> = <vy^2> = T.
"""
from __future__ import annotations

import numpy as np

from .backend_reference_2d import (
    ReferenceState2D,
    cic_deposit_2d,
    cic_deposit_scalar_2d,
)


T_FLOOR_DEFAULT = 1.0e-3
RHO_FLOOR_DEFAULT = 1.0e-3


def grid_moments_2d2v(
    state: ReferenceState2D,
    rho_floor: float = RHO_FLOOR_DEFAULT,
    T_floor: float = T_FLOOR_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (rho, ux, uy, T) on the 2D grid.

    Returns
    -------
    rho : (Nx, Ny)
    ux  : (Nx, Ny)
    uy  : (Nx, Ny)
    T   : (Nx, Ny) -- isotropic temperature, T = (1/2)(<vx^2 + vy^2> - u^2)
    """
    Lx = state.Lx
    Ly = state.Ly
    Nx = state.Nx
    Ny = state.Ny
    rho = cic_deposit_2d(state.x, state.y, state.w, Lx, Ly, Nx, Ny)
    jx = cic_deposit_scalar_2d(state.x, state.y, state.vx, state.w, Lx, Ly, Nx, Ny)
    jy = cic_deposit_scalar_2d(state.x, state.y, state.vy, state.w, Lx, Ly, Nx, Ny)
    v2 = state.vx * state.vx + state.vy * state.vy
    M = cic_deposit_scalar_2d(state.x, state.y, v2, state.w, Lx, Ly, Nx, Ny)
    rho_safe = np.maximum(rho, rho_floor)
    ux = jx / rho_safe
    uy = jy / rho_safe
    var = M / rho_safe - ux * ux - uy * uy
    T = np.maximum(0.5 * var, T_floor)
    return rho_safe, ux, uy, T


def density_error_2d(
    rho_assim: np.ndarray, rho_truth: np.ndarray,
) -> float:
    """Relative L2 density error normalised by ||rho_truth||."""
    diff = rho_assim - rho_truth
    return float(np.sqrt(np.mean(diff * diff)) / max(np.sqrt(np.mean(rho_truth * rho_truth)), 1e-30))


def velocity_error_2d(
    rho_assim: np.ndarray, ux_assim: np.ndarray, uy_assim: np.ndarray,
    rho_truth: np.ndarray, ux_truth: np.ndarray, uy_truth: np.ndarray,
    T_truth: np.ndarray | None = None,
) -> float:
    """Bulk-velocity error normalised by the thermal velocity scale.

    e_u = sqrt(< rho_truth * |u_assim - u_truth|^2 >)
          / max(sqrt(< rho_truth * T_truth >), |u_truth|_RMS).

    The denominator uses the thermal velocity sqrt(<T>) (or, if T_truth is
    None, falls back to the RMS bulk velocity), preventing the divide-by-zero
    blow-up when truth's u is essentially noise around zero.
    """
    diffx = rho_truth * (ux_assim - ux_truth)
    diffy = rho_truth * (uy_assim - uy_truth)
    num = float(np.sqrt(np.mean(diffx * diffx + diffy * diffy)))
    if T_truth is not None:
        scale = float(np.sqrt(np.mean(rho_truth * T_truth)))
    else:
        scale = float(np.sqrt(np.mean(rho_truth * rho_truth * (ux_truth ** 2 + uy_truth ** 2))))
    return num / max(scale, 1e-30)


def temperature_error_2d(
    rho_assim: np.ndarray, T_assim: np.ndarray,
    rho_truth: np.ndarray, T_truth: np.ndarray,
) -> float:
    """Truth-density-weighted temperature error.
    e_T = || rho_truth * (T_a - T_t) || / || rho_truth * T_truth ||.
    """
    diff = rho_truth * (T_assim - T_truth)
    num = float(np.sqrt(np.mean(diff * diff)))
    denom_sq = np.mean(rho_truth * rho_truth * T_truth * T_truth)
    return num / max(np.sqrt(denom_sq), 1e-30)


def potential_error_2d(
    phi_assim: np.ndarray, phi_truth: np.ndarray,
) -> float:
    diff = phi_assim - phi_truth
    return float(np.sqrt(np.mean(diff * diff)) / max(np.sqrt(np.mean(phi_truth * phi_truth)), 1e-30))


def electric_energy_2d(Ex: np.ndarray, Ey: np.ndarray, Lx: float, Ly: float) -> float:
    """Total electric energy (1/2) integral (Ex^2 + Ey^2) dx dy."""
    return 0.5 * float(np.mean(Ex * Ex + Ey * Ey)) * Lx * Ly
