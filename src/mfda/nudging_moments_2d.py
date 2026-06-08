"""Particle-level nudging from 2D moment observations.

Implements two formulations in 2D2V (mirror of nudging_moments.py):

  A_var: hydrodynamic-projection mobility (arxiv eqs. 17-18 in 2D).
  B    : metriplectic redirection (arxiv eqs. 19-22 in 2D).

In 2D the moment residuals are
  r_0   = rho_f - rho_obs,
  r_1x  = jx_f - jx_obs,
  r_1y  = jy_f - jy_obs,
  r_2   = E_f - E_obs   (with E = (1/2) <vx^2 + vy^2>).

The variational derivative of the moment loss is
  Phi_M(x, y, vx, vy) = gamma_1 (K_h * r_0)
                        + gamma_2 [vx (K_h * r_1x) + vy (K_h * r_1y)]
                        + (gamma_3 / 2) (vx^2 + vy^2) (K_h * r_2).

Phase-space gradients (1D-style notation extended to 2D):
  d_vx Phi_M = gamma_2 (K_h * r_1x) + gamma_3 vx (K_h * r_2)
  d_vy Phi_M = gamma_2 (K_h * r_1y) + gamma_3 vy (K_h * r_2)
  d_x  Phi_M = gamma_1 g0_x + gamma_2 (vx g1xx + vy g1yx)
               + (gamma_3/2)(vx^2 + vy^2) g2_x
  d_y  Phi_M = gamma_1 g0_y + gamma_2 (vx g1xy + vy g1yy)
               + (gamma_3/2)(vx^2 + vy^2) g2_y

The hydrodynamic projection acts as v -> u_f, vx^2+vy^2 -> ux_f^2+uy_f^2+2 T_f
(2D isotropic Maxwellian moments).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backend_reference_2d import (
    cic_deposit_2d,
    cic_deposit_scalar_2d,
    cic_interpolate_2d,
)
from .diagnostics_2d import (
    RHO_FLOOR_DEFAULT,
    T_FLOOR_DEFAULT,
)
from .filtering_2d import lowpass_filter_2d
from .poisson_2d import grad_2d


# ---------------------------------------------------------------------------
# Helpers shared across A_var and B.
# ---------------------------------------------------------------------------


def _model_moments_2d(
    x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray, w: np.ndarray,
    Lx: float, Ly: float, Nx: int, Ny: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deposit conserved moments (rho, jx, jy, E) of the assim particles."""
    rho_f = cic_deposit_2d(x, y, w, Lx, Ly, Nx, Ny)
    jx_f = cic_deposit_scalar_2d(x, y, vx, w, Lx, Ly, Nx, Ny)
    jy_f = cic_deposit_scalar_2d(x, y, vy, w, Lx, Ly, Nx, Ny)
    half_v2 = 0.5 * (vx * vx + vy * vy)
    E_f = cic_deposit_scalar_2d(x, y, half_v2, w, Lx, Ly, Nx, Ny)
    return rho_f, jx_f, jy_f, E_f


def _conserved_obs_2d(
    rho_obs: np.ndarray, ux_obs: np.ndarray, uy_obs: np.ndarray, T_obs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert primitive (rho, ux, uy, T) to conserved (jx, jy, E) (d=2)."""
    jx_obs = rho_obs * ux_obs
    jy_obs = rho_obs * uy_obs
    # E = (1/2) rho * (ux^2 + uy^2) + rho * T  for d=2 isotropic:
    #   <vx^2 + vy^2> = ux^2 + uy^2 + 2T => E = (1/2) rho (ux^2+uy^2 + 2T)
    E_obs = 0.5 * rho_obs * (ux_obs * ux_obs + uy_obs * uy_obs) + rho_obs * T_obs
    return jx_obs, jy_obs, E_obs


def _filter_or_passthrough_2d(
    field: np.ndarray, Lx: float, Ly: float,
    k_cut_frac: float, sharpness: float,
) -> np.ndarray:
    if k_cut_frac >= 1.0:
        return field
    return lowpass_filter_2d(field, Lx, Ly, k_cut_frac=k_cut_frac, sharpness=sharpness)


def _smoothed_primitive_moments_2d(
    x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray, w: np.ndarray,
    Lx: float, Ly: float, Nx: int, Ny: int,
    k_cut_frac: float, sharpness: float,
    rho_floor: float, T_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build (rho, ux, uy, T) on the 2D grid from K_h-smoothed conserved moments.
    For d=2 isotropic: T = (1/d) * (<v^2>/rho - |u|^2) = (1/2) (2 E/rho - u^2)."""
    rho_raw, jx_raw, jy_raw, E_raw = _model_moments_2d(
        x, y, vx, vy, w, Lx, Ly, Nx, Ny,
    )
    rho_t = _filter_or_passthrough_2d(rho_raw, Lx, Ly, k_cut_frac, sharpness)
    jx_t = _filter_or_passthrough_2d(jx_raw, Lx, Ly, k_cut_frac, sharpness)
    jy_t = _filter_or_passthrough_2d(jy_raw, Lx, Ly, k_cut_frac, sharpness)
    E_t = _filter_or_passthrough_2d(E_raw, Lx, Ly, k_cut_frac, sharpness)
    rho_safe = np.maximum(rho_t, rho_floor)
    ux_t = jx_t / rho_safe
    uy_t = jy_t / rho_safe
    # E_t = (1/2) rho_t (ux^2 + uy^2) + rho_t T  =>  T = E_t / rho - 0.5 (ux^2+uy^2)
    var_iso = E_t / rho_safe - 0.5 * (ux_t * ux_t + uy_t * uy_t)
    T_t = np.maximum(var_iso, T_floor)
    return rho_safe, ux_t, uy_t, T_t


# ---------------------------------------------------------------------------
# Formulation A -- weighted-W2 gradient flow (2D).
# ---------------------------------------------------------------------------


@dataclass
class FormulationAParams2D:
    """Strengths & metric scale for Formulation A in 2D2V (eq. 16, d=2)."""
    gamma_1: float = 1.0
    gamma_2: float = 1.0
    gamma_3: float = 1.0
    V_star: float = 1.0
    lowpass_k_cut_frac: float = 0.25
    lowpass_sharpness: float = 16.0


def apply_formulation_A_2d(
    x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray, w: np.ndarray,
    Lx: float, Ly: float, Nx: int, Ny: int,
    rho_obs: np.ndarray, ux_obs: np.ndarray, uy_obs: np.ndarray, T_obs: np.ndarray,
    params: FormulationAParams2D, dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A in 2D2V (eq. 16). v-direction drift same as A_var; position drift
    is v-dependent and divided by the weighted-W2 inverse metric.

    dx = -d_x Phi_M / (1 + |v - u_obs|^2 / V_*^2) dt
    dy = -d_y Phi_M / (1 + |v - u_obs|^2 / V_*^2) dt
    dvx = -(gamma_2 K_h r_1x + gamma_3 vx K_h r_2) dt
    dvy = -(gamma_2 K_h r_1y + gamma_3 vy K_h r_2) dt
    """
    rho_f, jx_f, jy_f, E_f = _model_moments_2d(x, y, vx, vy, w, Lx, Ly, Nx, Ny)
    jx_obs_g, jy_obs_g, E_obs_g = _conserved_obs_2d(rho_obs, ux_obs, uy_obs, T_obs)
    r0 = rho_f - rho_obs
    r1x = jx_f - jx_obs_g
    r1y = jy_f - jy_obs_g
    r2 = E_f - E_obs_g

    Kr0 = _filter_or_passthrough_2d(r0, Lx, Ly,
                                     params.lowpass_k_cut_frac, params.lowpass_sharpness)
    Kr1x = _filter_or_passthrough_2d(r1x, Lx, Ly,
                                      params.lowpass_k_cut_frac, params.lowpass_sharpness)
    Kr1y = _filter_or_passthrough_2d(r1y, Lx, Ly,
                                      params.lowpass_k_cut_frac, params.lowpass_sharpness)
    Kr2 = _filter_or_passthrough_2d(r2, Lx, Ly,
                                     params.lowpass_k_cut_frac, params.lowpass_sharpness)

    g0_x, g0_y = grad_2d(Kr0, Lx, Ly)
    g1xx, g1xy = grad_2d(Kr1x, Lx, Ly)
    g1yx, g1yy = grad_2d(Kr1y, Lx, Ly)
    g2_x, g2_y = grad_2d(Kr2, Lx, Ly)

    g0xp = cic_interpolate_2d(g0_x, x, y, Lx, Ly)
    g0yp = cic_interpolate_2d(g0_y, x, y, Lx, Ly)
    g1xxp = cic_interpolate_2d(g1xx, x, y, Lx, Ly)
    g1xyp = cic_interpolate_2d(g1xy, x, y, Lx, Ly)
    g1yxp = cic_interpolate_2d(g1yx, x, y, Lx, Ly)
    g1yyp = cic_interpolate_2d(g1yy, x, y, Lx, Ly)
    g2xp = cic_interpolate_2d(g2_x, x, y, Lx, Ly)
    g2yp = cic_interpolate_2d(g2_y, x, y, Lx, Ly)
    Kr1x_p = cic_interpolate_2d(Kr1x, x, y, Lx, Ly)
    Kr1y_p = cic_interpolate_2d(Kr1y, x, y, Lx, Ly)
    Kr2_p = cic_interpolate_2d(Kr2, x, y, Lx, Ly)
    ux_obs_p = cic_interpolate_2d(ux_obs, x, y, Lx, Ly)
    uy_obs_p = cic_interpolate_2d(uy_obs, x, y, Lx, Ly)

    g1 = params.gamma_1
    g2 = params.gamma_2
    g3 = params.gamma_3
    v2 = vx * vx + vy * vy
    dxPhi = g1 * g0xp + g2 * (vx * g1xxp + vy * g1yxp) + 0.5 * g3 * v2 * g2xp
    dyPhi = g1 * g0yp + g2 * (vx * g1xyp + vy * g1yyp) + 0.5 * g3 * v2 * g2yp
    dvxPhi = g2 * Kr1x_p + g3 * vx * Kr2_p
    dvyPhi = g2 * Kr1y_p + g3 * vy * Kr2_p

    # Weighted-W2 inverse metric, applied to position drift only.
    dvx_obs = vx - ux_obs_p
    dvy_obs = vy - uy_obs_p
    inv_w = 1.0 / (1.0 + (dvx_obs * dvx_obs + dvy_obs * dvy_obs) / (params.V_star ** 2))

    x_new = np.mod(x - dxPhi * inv_w * dt, Lx)
    y_new = np.mod(y - dyPhi * inv_w * dt, Ly)
    vx_new = vx - dvxPhi * dt
    vy_new = vy - dvyPhi * dt
    return x_new, y_new, vx_new, vy_new


# ---------------------------------------------------------------------------
# Formulation A variant -- hydrodynamic-projection mobility (2D).
# ---------------------------------------------------------------------------


@dataclass
class FormulationAVariantParams2D:
    gamma_1: float = 1.0
    gamma_2: float = 1.0
    gamma_3: float = 1.0
    lowpass_k_cut_frac: float = 0.25
    lowpass_sharpness: float = 16.0
    rho_floor: float = RHO_FLOOR_DEFAULT
    T_floor: float = T_FLOOR_DEFAULT


def apply_formulation_A_variant_2d(
    x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray, w: np.ndarray,
    Lx: float, Ly: float, Nx: int, Ny: int,
    rho_obs: np.ndarray, ux_obs: np.ndarray, uy_obs: np.ndarray, T_obs: np.ndarray,
    params: FormulationAVariantParams2D, dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A variant in 2D2V. Position drift is v-independent via Pi_f.

    dx = -[gamma_1 g0_x + gamma_2 (ux_f g1xx + uy_f g1yx)
           + (gamma_3/2)(ux_f^2 + uy_f^2 + 2 T_f) g2_x] dt
    dy = -[gamma_1 g0_y + gamma_2 (ux_f g1xy + uy_f g1yy)
           + (gamma_3/2)(ux_f^2 + uy_f^2 + 2 T_f) g2_y] dt
    dvx = -[gamma_2 K_h r_1x + gamma_3 vx K_h r_2] dt
    dvy = -[gamma_2 K_h r_1y + gamma_3 vy K_h r_2] dt
    """
    # Conserved residuals on the grid.
    rho_f, jx_f, jy_f, E_f = _model_moments_2d(
        x, y, vx, vy, w, Lx, Ly, Nx, Ny,
    )
    jx_obs_g, jy_obs_g, E_obs_g = _conserved_obs_2d(rho_obs, ux_obs, uy_obs, T_obs)
    r0 = rho_f - rho_obs
    r1x = jx_f - jx_obs_g
    r1y = jy_f - jy_obs_g
    r2 = E_f - E_obs_g

    Kr0 = _filter_or_passthrough_2d(r0, Lx, Ly,
                                     params.lowpass_k_cut_frac, params.lowpass_sharpness)
    Kr1x = _filter_or_passthrough_2d(r1x, Lx, Ly,
                                      params.lowpass_k_cut_frac, params.lowpass_sharpness)
    Kr1y = _filter_or_passthrough_2d(r1y, Lx, Ly,
                                      params.lowpass_k_cut_frac, params.lowpass_sharpness)
    Kr2 = _filter_or_passthrough_2d(r2, Lx, Ly,
                                     params.lowpass_k_cut_frac, params.lowpass_sharpness)

    # Spatial gradients of smoothed residuals.
    g0_x, g0_y = grad_2d(Kr0, Lx, Ly)
    g1xx, g1xy = grad_2d(Kr1x, Lx, Ly)
    g1yx, g1yy = grad_2d(Kr1y, Lx, Ly)
    g2_x, g2_y = grad_2d(Kr2, Lx, Ly)

    # Smoothed local hydro moments of f (for Pi_f).
    _, ux_f, uy_f, T_f = _smoothed_primitive_moments_2d(
        x, y, vx, vy, w, Lx, Ly, Nx, Ny,
        params.lowpass_k_cut_frac, params.lowpass_sharpness,
        params.rho_floor, params.T_floor,
    )

    # Interpolate everything to particles.
    g0xp = cic_interpolate_2d(g0_x, x, y, Lx, Ly)
    g0yp = cic_interpolate_2d(g0_y, x, y, Lx, Ly)
    g1xxp = cic_interpolate_2d(g1xx, x, y, Lx, Ly)
    g1xyp = cic_interpolate_2d(g1xy, x, y, Lx, Ly)
    g1yxp = cic_interpolate_2d(g1yx, x, y, Lx, Ly)
    g1yyp = cic_interpolate_2d(g1yy, x, y, Lx, Ly)
    g2xp = cic_interpolate_2d(g2_x, x, y, Lx, Ly)
    g2yp = cic_interpolate_2d(g2_y, x, y, Lx, Ly)
    Kr1x_p = cic_interpolate_2d(Kr1x, x, y, Lx, Ly)
    Kr1y_p = cic_interpolate_2d(Kr1y, x, y, Lx, Ly)
    Kr2_p = cic_interpolate_2d(Kr2, x, y, Lx, Ly)
    ux_f_p = cic_interpolate_2d(ux_f, x, y, Lx, Ly)
    uy_f_p = cic_interpolate_2d(uy_f, x, y, Lx, Ly)
    T_f_p = cic_interpolate_2d(T_f, x, y, Lx, Ly)

    # Pi_f * grad_x Phi_M : v -> u_f, (vx^2+vy^2) -> ux^2 + uy^2 + 2 T.
    u2T = ux_f_p * ux_f_p + uy_f_p * uy_f_p + 2.0 * T_f_p
    g1 = params.gamma_1
    g2 = params.gamma_2
    g3 = params.gamma_3
    dxPhi_proj = (g1 * g0xp
                  + g2 * (ux_f_p * g1xxp + uy_f_p * g1yxp)
                  + 0.5 * g3 * u2T * g2xp)
    dyPhi_proj = (g1 * g0yp
                  + g2 * (ux_f_p * g1xyp + uy_f_p * g1yyp)
                  + 0.5 * g3 * u2T * g2yp)
    # grad_v Phi_M (per-particle).
    dvxPhi = g2 * Kr1x_p + g3 * vx * Kr2_p
    dvyPhi = g2 * Kr1y_p + g3 * vy * Kr2_p

    x_new = np.mod(x - dxPhi_proj * dt, Lx)
    y_new = np.mod(y - dyPhi_proj * dt, Ly)
    vx_new = vx - dvxPhi * dt
    vy_new = vy - dvyPhi * dt
    return x_new, y_new, vx_new, vy_new


# ---------------------------------------------------------------------------
# Formulation B -- metriplectic redirection (2D).
# ---------------------------------------------------------------------------


@dataclass
class FormulationBParams2D:
    gamma: float = 1.0
    gamma_1: float = 1.0
    gamma_2: float = 1.0
    gamma_3: float = 1.0
    lowpass_k_cut_frac: float = 0.25
    lowpass_sharpness: float = 16.0
    rho_floor: float = RHO_FLOOR_DEFAULT
    T_floor: float = T_FLOOR_DEFAULT


def apply_formulation_B_2d(
    x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray, w: np.ndarray,
    Lx: float, Ly: float, Nx: int, Ny: int,
    rho_obs: np.ndarray, ux_obs: np.ndarray, uy_obs: np.ndarray, T_obs: np.ndarray,
    params: FormulationBParams2D, dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Metriplectic B in 2D2V.

    dx = +gamma * [gamma_2 K_h r_1x + gamma_3 ux_f K_h r_2] dt   (mod Lx)
    dy = +gamma * [gamma_2 K_h r_1y + gamma_3 uy_f K_h r_2] dt   (mod Ly)
    dvx = -gamma * [Pi_f d_x Phi_M + d_vx Phi_M] dt
    dvy = -gamma * [Pi_f d_y Phi_M + d_vy Phi_M] dt
    """
    rho_f, jx_f, jy_f, E_f = _model_moments_2d(
        x, y, vx, vy, w, Lx, Ly, Nx, Ny,
    )
    jx_obs_g, jy_obs_g, E_obs_g = _conserved_obs_2d(rho_obs, ux_obs, uy_obs, T_obs)
    r0 = rho_f - rho_obs
    r1x = jx_f - jx_obs_g
    r1y = jy_f - jy_obs_g
    r2 = E_f - E_obs_g

    Kr0 = _filter_or_passthrough_2d(r0, Lx, Ly,
                                     params.lowpass_k_cut_frac, params.lowpass_sharpness)
    Kr1x = _filter_or_passthrough_2d(r1x, Lx, Ly,
                                      params.lowpass_k_cut_frac, params.lowpass_sharpness)
    Kr1y = _filter_or_passthrough_2d(r1y, Lx, Ly,
                                      params.lowpass_k_cut_frac, params.lowpass_sharpness)
    Kr2 = _filter_or_passthrough_2d(r2, Lx, Ly,
                                     params.lowpass_k_cut_frac, params.lowpass_sharpness)

    g0_x, g0_y = grad_2d(Kr0, Lx, Ly)
    g1xx, g1xy = grad_2d(Kr1x, Lx, Ly)
    g1yx, g1yy = grad_2d(Kr1y, Lx, Ly)
    g2_x, g2_y = grad_2d(Kr2, Lx, Ly)

    _, ux_f, uy_f, T_f = _smoothed_primitive_moments_2d(
        x, y, vx, vy, w, Lx, Ly, Nx, Ny,
        params.lowpass_k_cut_frac, params.lowpass_sharpness,
        params.rho_floor, params.T_floor,
    )

    g0xp = cic_interpolate_2d(g0_x, x, y, Lx, Ly)
    g0yp = cic_interpolate_2d(g0_y, x, y, Lx, Ly)
    g1xxp = cic_interpolate_2d(g1xx, x, y, Lx, Ly)
    g1xyp = cic_interpolate_2d(g1xy, x, y, Lx, Ly)
    g1yxp = cic_interpolate_2d(g1yx, x, y, Lx, Ly)
    g1yyp = cic_interpolate_2d(g1yy, x, y, Lx, Ly)
    g2xp = cic_interpolate_2d(g2_x, x, y, Lx, Ly)
    g2yp = cic_interpolate_2d(g2_y, x, y, Lx, Ly)
    Kr1x_p = cic_interpolate_2d(Kr1x, x, y, Lx, Ly)
    Kr1y_p = cic_interpolate_2d(Kr1y, x, y, Lx, Ly)
    Kr2_p = cic_interpolate_2d(Kr2, x, y, Lx, Ly)
    ux_f_p = cic_interpolate_2d(ux_f, x, y, Lx, Ly)
    uy_f_p = cic_interpolate_2d(uy_f, x, y, Lx, Ly)
    T_f_p = cic_interpolate_2d(T_f, x, y, Lx, Ly)

    g = params.gamma
    g1 = params.gamma_1
    g2 = params.gamma_2
    g3 = params.gamma_3

    # Position kinematic correction (= Pi_f grad_v Phi_M, v->u_f).
    dx_corr = g2 * Kr1x_p + g3 * ux_f_p * Kr2_p
    dy_corr = g2 * Kr1y_p + g3 * uy_f_p * Kr2_p

    # Velocity force = Pi_f grad_x Phi_M + grad_v Phi_M.
    u2T = ux_f_p * ux_f_p + uy_f_p * uy_f_p + 2.0 * T_f_p
    dvxForce = (g1 * g0xp
                + g2 * (ux_f_p * g1xxp + uy_f_p * g1yxp)
                + 0.5 * g3 * u2T * g2xp
                + g2 * Kr1x_p
                + g3 * vx * Kr2_p)
    dvyForce = (g1 * g0yp
                + g2 * (ux_f_p * g1xyp + uy_f_p * g1yyp)
                + 0.5 * g3 * u2T * g2yp
                + g2 * Kr1y_p
                + g3 * vy * Kr2_p)

    x_new = np.mod(x + g * dx_corr * dt, Lx)
    y_new = np.mod(y + g * dy_corr * dt, Ly)
    vx_new = vx - g * dvxForce * dt
    vy_new = vy - g * dvyForce * dt
    return x_new, y_new, vx_new, vy_new


# ---------------------------------------------------------------------------
# Formulation C -- Maxwellian-projected KL nudging (2D, d=2).
# ---------------------------------------------------------------------------


@dataclass
class FormulationCParams2D:
    """Parameters for Formulation C in 2D2V (Maxwellian-projected KL, d=2).

    Bregman coefficients differ from 1D only in the dimension constant in A:
        A = log(rho_f̃/rho_obs) + 1 - (d/2) log(T_f̃/T_obs)
            + |u_obs|^2 / (2 T_obs) - |u_f̃|^2 / (2 T_f̃)
    For d=2 the leading coefficient on log(T_f̃/T_obs) is -1 (not -1/2).
    Bx, By scale with u_x, u_y (vector); C is scalar (isotropic T).

    Phase-space drift:
        dx, dy   = -lam * inv_metric * [grad A + vx grad Bx + vy grad By
                                         + (1/2)(vx^2 + vy^2) grad C]
        dvx      = -lam * (Bx + C vx)
        dvy      = -lam * (By + C vy)

    `inv_metric` = 1 / (1 + |v - u_obs|^2 / V_*^2) when use_weighted_metric.
    """
    lam: float = 1.0
    use_weighted_metric: bool = True
    V_star: float = 1.0
    lowpass_k_cut_frac: float = 0.25
    lowpass_sharpness: float = 16.0
    rho_floor: float = RHO_FLOOR_DEFAULT
    T_floor: float = T_FLOOR_DEFAULT


def _bregman_coefficients_2d(
    rho_f: np.ndarray, ux_f: np.ndarray, uy_f: np.ndarray, T_f: np.ndarray,
    rho_obs: np.ndarray, ux_obs: np.ndarray, uy_obs: np.ndarray, T_obs: np.ndarray,
    rho_floor: float, T_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (A, Bx, By, C) on the grid from primitive moments (2D, d=2)."""
    rho_obs_safe = np.maximum(rho_obs, rho_floor)
    T_obs_safe = np.maximum(T_obs, T_floor)
    u_obs2 = ux_obs * ux_obs + uy_obs * uy_obs
    u_f2 = ux_f * ux_f + uy_f * uy_f
    # d=2: coefficient on log(T_f/T_obs) is -d/2 = -1.
    A = (np.log(rho_f / rho_obs_safe) + 1.0
         - np.log(T_f / T_obs_safe)
         + 0.5 * u_obs2 / T_obs_safe
         - 0.5 * u_f2 / T_f)
    Bx = ux_f / T_f - ux_obs / T_obs_safe
    By = uy_f / T_f - uy_obs / T_obs_safe
    C = 1.0 / T_obs_safe - 1.0 / T_f
    return A, Bx, By, C


def apply_formulation_C_2d(
    x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray, w: np.ndarray,
    Lx: float, Ly: float, Nx: int, Ny: int,
    rho_obs: np.ndarray, ux_obs: np.ndarray, uy_obs: np.ndarray, T_obs: np.ndarray,
    params: FormulationCParams2D, dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Particle update from Formulation C in 2D2V."""
    rho_f, ux_f, uy_f, T_f = _smoothed_primitive_moments_2d(
        x, y, vx, vy, w, Lx, Ly, Nx, Ny,
        params.lowpass_k_cut_frac, params.lowpass_sharpness,
        params.rho_floor, params.T_floor,
    )
    A, Bx, By, C = _bregman_coefficients_2d(
        rho_f, ux_f, uy_f, T_f,
        rho_obs, ux_obs, uy_obs, T_obs,
        params.rho_floor, params.T_floor,
    )
    grad_A_x, grad_A_y = grad_2d(A, Lx, Ly)
    grad_Bx_x, grad_Bx_y = grad_2d(Bx, Lx, Ly)
    grad_By_x, grad_By_y = grad_2d(By, Lx, Ly)
    grad_C_x, grad_C_y = grad_2d(C, Lx, Ly)

    Bx_p = cic_interpolate_2d(Bx, x, y, Lx, Ly)
    By_p = cic_interpolate_2d(By, x, y, Lx, Ly)
    C_p = cic_interpolate_2d(C, x, y, Lx, Ly)
    grad_A_xp = cic_interpolate_2d(grad_A_x, x, y, Lx, Ly)
    grad_A_yp = cic_interpolate_2d(grad_A_y, x, y, Lx, Ly)
    grad_Bx_xp = cic_interpolate_2d(grad_Bx_x, x, y, Lx, Ly)
    grad_Bx_yp = cic_interpolate_2d(grad_Bx_y, x, y, Lx, Ly)
    grad_By_xp = cic_interpolate_2d(grad_By_x, x, y, Lx, Ly)
    grad_By_yp = cic_interpolate_2d(grad_By_y, x, y, Lx, Ly)
    grad_C_xp = cic_interpolate_2d(grad_C_x, x, y, Lx, Ly)
    grad_C_yp = cic_interpolate_2d(grad_C_y, x, y, Lx, Ly)

    v2 = vx * vx + vy * vy
    dx_drift = grad_A_xp + vx * grad_Bx_xp + vy * grad_By_xp + 0.5 * v2 * grad_C_xp
    dy_drift = grad_A_yp + vx * grad_Bx_yp + vy * grad_By_yp + 0.5 * v2 * grad_C_yp

    if params.use_weighted_metric:
        ux_obs_p = cic_interpolate_2d(ux_obs, x, y, Lx, Ly)
        uy_obs_p = cic_interpolate_2d(uy_obs, x, y, Lx, Ly)
        dvx_obs = vx - ux_obs_p
        dvy_obs = vy - uy_obs_p
        inv_metric = 1.0 / (1.0 + (dvx_obs * dvx_obs + dvy_obs * dvy_obs)
                                   / (params.V_star ** 2))
        dx_drift = inv_metric * dx_drift
        dy_drift = inv_metric * dy_drift

    vx_new = vx - params.lam * (Bx_p + C_p * vx) * dt
    vy_new = vy - params.lam * (By_p + C_p * vy) * dt
    x_new = np.mod(x - params.lam * dx_drift * dt, Lx)
    y_new = np.mod(y - params.lam * dy_drift * dt, Ly)
    return x_new, y_new, vx_new, vy_new


# ---------------------------------------------------------------------------
# Baseline -- classical constant-gain AOT moment nudging (2D2V).
# ---------------------------------------------------------------------------


@dataclass
class FormulationAOTParams2D:
    """Constant-gain Luenberger / AOT feedback on the primitive moments (2D).

    2D mirror of the 1D FormulationAOTParams: drives (rho_f, u_f, T_f) toward
    (rho_obs, u_obs, T_obs) with three independent constant gains, with no
    loss functional. Bulk velocity is shifted per component; temperature is
    set by a mass/momentum-preserving affine rescaling of the peculiar
    velocity about the cloud's own mean; density via a position nudge down the
    smoothed density-residual gradient. Deterministic (no RNG).
    """
    mu_rho: float = 1.0
    mu_u: float = 1.0
    mu_T: float = 1.0
    lowpass_k_cut_frac: float = 0.25
    lowpass_sharpness: float = 16.0
    rho_floor: float = RHO_FLOOR_DEFAULT
    T_floor: float = T_FLOOR_DEFAULT


def apply_aot_2d(
    x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray, w: np.ndarray,
    Lx: float, Ly: float, Nx: int, Ny: int,
    rho_obs: np.ndarray, ux_obs: np.ndarray, uy_obs: np.ndarray, T_obs: np.ndarray,
    params: FormulationAOTParams2D, dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Classical constant-gain AOT nudging on primitive moments over one dt (2D)."""
    rho_f, ux_f, uy_f, T_f = _smoothed_primitive_moments_2d(
        x, y, vx, vy, w, Lx, Ly, Nx, Ny,
        params.lowpass_k_cut_frac, params.lowpass_sharpness,
        params.rho_floor, params.T_floor,
    )
    # Density-residual gradient for the position channel.
    r0 = rho_f - rho_obs
    Kr0 = _filter_or_passthrough_2d(
        r0, Lx, Ly, params.lowpass_k_cut_frac, params.lowpass_sharpness)
    g0x, g0y = grad_2d(Kr0, Lx, Ly)
    g0xp = cic_interpolate_2d(g0x, x, y, Lx, Ly)
    g0yp = cic_interpolate_2d(g0y, x, y, Lx, Ly)

    # Primitive moment fields at particles.
    ux_f_p = cic_interpolate_2d(ux_f, x, y, Lx, Ly)
    uy_f_p = cic_interpolate_2d(uy_f, x, y, Lx, Ly)
    ux_obs_p = cic_interpolate_2d(ux_obs, x, y, Lx, Ly)
    uy_obs_p = cic_interpolate_2d(uy_obs, x, y, Lx, Ly)
    T_f_p = cic_interpolate_2d(T_f, x, y, Lx, Ly)
    T_obs_p = cic_interpolate_2d(np.maximum(T_obs, params.T_floor), x, y, Lx, Ly)

    # Velocity channel: bulk-velocity nudge + temperature (peculiar-velocity) rescale.
    temp_fac = 0.5 * params.mu_T * ((T_f_p - T_obs_p) / T_f_p)
    vx_new = vx - (params.mu_u * (ux_f_p - ux_obs_p) + temp_fac * (vx - ux_f_p)) * dt
    vy_new = vy - (params.mu_u * (uy_f_p - uy_obs_p) + temp_fac * (vy - uy_f_p)) * dt

    # Position channel: density nudge down the residual gradient.
    x_new = np.mod(x - params.mu_rho * g0xp * dt, Lx)
    y_new = np.mod(y - params.mu_rho * g0yp * dt, Ly)
    return x_new, y_new, vx_new, vy_new
