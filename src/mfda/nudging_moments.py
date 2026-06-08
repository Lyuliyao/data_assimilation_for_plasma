"""Particle-level nudging from hydrodynamic-moment observations.

Implements the three formulations of note v3 §3.2:

  A. Weighted Wasserstein gradient flow (eq. 16):
        b_A = ( -grad_x Phi_M / (1 + |v - u_obs|^2 / V_*^2),  -grad_v Phi_M )
     where Phi_M is the variational derivative of the moment loss J_M.

  B. Direction-split moment-loss gradient flow (revised PDF, eqs. 17-18):
        dX_extra = -gamma_x * d/dx Phi_M^rho(x_p) * dt
                 = -gamma_x * gamma_1 * g_0(x_p) * dt
        dV_extra = -gamma_v * d/dv Phi_M^{j,E}(x_p, v_p) * dt
                 = -gamma_v * (gamma_2 (K_h*r_1)(x_p) + gamma_3 v_p (K_h*r_2)(x_p)) * dt
     B is A's velocity-direction drift combined with the position drift
     of a density-only loss J_M^rho (no cross terms, no weighted metric).
     Both channels are residual-driven: r_0 = r_1 = r_2 = 0 implies both
     updates vanish identically. Fully deterministic (no Brownian noise).

  C. Maxwellian-projected KL nudging (latest PDF revision):
        H[f] = KL(Pi_M f || pi_obs)
        Phi_M^(c)(x, v) = A(x) + B(x)*v + (1/2)*C(x)*|v|^2
        with Bregman coefficients (1D, d=1):
          A = log(rho_f/rho_obs) + 1 - (1/2) log(T_f/T_obs)
              + (u_obs^2)/(2 T_obs) - (u_f^2)/(2 T_f)
          B = u_f/T_f - u_obs/T_obs
          C = 1/T_obs - 1/T_f
        Particle drift (with optional weighted-W2 metric on dx):
          dX_extra = -lam * [grad_x A + V * grad_x B + (1/2) V^2 * grad_x C] dt
          dV_extra = -lam * [B + C * V] dt
        Coefficients are evaluated on smoothed model moments
        (rho_f̃, u_f̃, T_f̃) = derived from K_h*(rho_f, j_f, E_f).
        H[f]=0 iff (rho_f, u_f, T_f) = (rho_obs, u_obs, T_obs); the
        drift then vanishes identically.

These are pure functions: they take particle arrays + grid moment
observations and return updated arrays. The reference backend's leapfrog
runs separately; the assimilation loop calls the chosen formulation
between drift and the second half-kick (consistent with how the existing
ψ-only nudging is wired in `assimilation.py`).

Conventions & implementation notes
----------------------------------
- 1D1V everywhere. v is a scalar; gradients in v are scalar derivatives.
- All grid-evaluated fields (residuals r_0, r_1, r_2; gradients;
  observations) are interpolated to particle positions via CIC, matching
  the rest of the backend.
- "K_h *" is implemented as a single application of `lowpass_filter`
  (filtering.py), with cutoff fraction taken from the formulation config.
  Note that residuals already incur one CIC smoothing from the deposit;
  the explicit lowpass adds the explicit K_h convolution.
- For Formulation A, the v^2 term in grad_x Phi_M is the destabilising
  term motivating the weighted W2 metric. The weight (1 + |v - u_obs|^2 /
  V_*^2)^{-1} divides only the position-direction drift, leaving the
  velocity-direction drift unchanged.
- For Formulation B, the velocity channel is the deterministic
  v-gradient of J_M^{j,E} (= the same v-gradient used by A). The
  position channel uses only the rho-residual gradient, dropping
  the cross terms that A keeps. No Brownian noise.
- For Formulation C, A/B/C are the Bregman coefficients of
  log/reciprocal type. They require strictly positive lower bounds
  rho_floor, T_floor on (rho_f̃, T_f̃) for the drift to be well
  defined. `Pi_M f` is the Maxwellian projection of f, so H[f]
  depends on f only through (rho_f, u_f, T_f) — three linear moments,
  same as A and B.

All three functions return (x_new, v_new). They never write into the
inputs in-place.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backend_reference import cic_deposit, cic_deposit_current, cic_interpolate
from .collisions import T_FLOOR_DEFAULT, RHO_FLOOR_DEFAULT
from .filtering import lowpass_filter
from .poisson import grad_1d


# ---------------------------------------------------------------------------
# Helpers shared across A and B: build the moment residuals r_0, r_1, r_2.
# ---------------------------------------------------------------------------


def _model_moments(
    x: np.ndarray, v: np.ndarray, w: np.ndarray, L: float, Nx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deposit the conserved moments (rho_f, j_f, E_f) of the assim particles.

    Returns
    -------
    rho_f : (Nx,)  density (= B_0 f).
    j_f   : (Nx,)  current  (= B_1 f).
    E_f   : (Nx,)  kinetic-energy density (= B_2 f) = (1/2) sum_p w_p v_p^2 S(x).
    """
    rho_f = cic_deposit(x, w, L, Nx)
    j_f = cic_deposit_current(x, v, w, L, Nx)
    # E_f = (1/2) sum w v^2 S(x)
    dx = L / Nx
    xi = x / dx
    i0 = np.floor(xi).astype(np.int64)
    frac = xi - i0
    i0 = np.mod(i0, Nx)
    i1 = np.mod(i0 + 1, Nx)
    half_v2_w = 0.5 * v * v * w
    E_f = (np.bincount(i0, weights=half_v2_w * (1.0 - frac), minlength=Nx)
           + np.bincount(i1, weights=half_v2_w * frac, minlength=Nx))
    E_f /= dx
    return rho_f, j_f, E_f


def _conserved_obs(
    rho_obs: np.ndarray, u_obs: np.ndarray, T_obs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert primitive (rho, u, T) to conserved (j, E) for residuals.

    j_obs = rho_obs * u_obs
    E_obs = (1/2) rho_obs * u_obs^2 + (1/2) rho_obs * T_obs   (d=1)
    """
    j_obs = rho_obs * u_obs
    E_obs = 0.5 * rho_obs * u_obs * u_obs + 0.5 * rho_obs * T_obs
    return j_obs, E_obs


def _filter_or_passthrough(
    field: np.ndarray, L: float, k_cut_frac: float, sharpness: float,
) -> np.ndarray:
    if k_cut_frac >= 1.0:
        return field
    return lowpass_filter(field, L, k_cut_frac=k_cut_frac, sharpness=sharpness)


# ---------------------------------------------------------------------------
# Formulation A — weighted-W2 gradient flow.
# ---------------------------------------------------------------------------


@dataclass
class FormulationAParams:
    """Strengths & metric scale for Formulation A (eq. 16)."""
    gamma_1: float = 1.0   # density residual weight
    gamma_2: float = 1.0   # current  residual weight
    gamma_3: float = 1.0   # energy   residual weight
    V_star: float = 1.0    # reference thermal speed for the W2 weight
    lowpass_k_cut_frac: float = 0.25
    lowpass_sharpness: float = 16.0


def apply_formulation_A(
    x: np.ndarray, v: np.ndarray, w: np.ndarray, L: float, Nx: int,
    rho_obs: np.ndarray, u_obs: np.ndarray, T_obs: np.ndarray,
    params: FormulationAParams, dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Particle update from Formulation A (eq. 16) over one dt.

    Variational derivative on the grid (1D, d=1):
      Phi_M(x, v) = gamma_1 (K_h * r_0)(x)
                    + gamma_2 v (K_h * r_1)(x)
                    + (gamma_3 / 2) v^2 (K_h * r_2)(x)
    so:
      d_x Phi_M(x, v) = gamma_1 g_0(x) + gamma_2 v g_1(x) + (gamma_3/2) v^2 g_2(x)
      d_v Phi_M(x, v) = gamma_2 (K_h*r_1)(x) + gamma_3 v (K_h*r_2)(x)
    where g_j = d/dx (K_h * r_j).

    Weighted-W2 metric divides the position drift by (1 + (v-u_obs)^2 / V_*^2):
      dx = - d_x Phi_M / (1 + (v - u_obs)^2 / V_*^2) * dt
      dv = - d_v Phi_M * dt
    """
    # Build residuals on the grid.
    rho_f, j_f, E_f = _model_moments(x, v, w, L, Nx)
    j_obs, E_obs = _conserved_obs(rho_obs, u_obs, T_obs)
    r0 = rho_f - rho_obs
    r1 = j_f - j_obs
    r2 = E_f - E_obs
    Kr0 = _filter_or_passthrough(r0, L, params.lowpass_k_cut_frac, params.lowpass_sharpness)
    Kr1 = _filter_or_passthrough(r1, L, params.lowpass_k_cut_frac, params.lowpass_sharpness)
    Kr2 = _filter_or_passthrough(r2, L, params.lowpass_k_cut_frac, params.lowpass_sharpness)
    g0 = grad_1d(Kr0, L)
    g1 = grad_1d(Kr1, L)
    g2 = grad_1d(Kr2, L)

    # Interpolate to particles.
    Kr1_p = cic_interpolate(Kr1, x, L)
    Kr2_p = cic_interpolate(Kr2, x, L)
    g0_p = cic_interpolate(g0, x, L)
    g1_p = cic_interpolate(g1, x, L)
    g2_p = cic_interpolate(g2, x, L)
    u_obs_p = cic_interpolate(u_obs, x, L)

    # Per-particle phase-space gradient of Phi_M.
    dxPhi = params.gamma_1 * g0_p + params.gamma_2 * v * g1_p + 0.5 * params.gamma_3 * v * v * g2_p
    dvPhi = params.gamma_2 * Kr1_p + params.gamma_3 * v * Kr2_p

    # Weighted W2 inverse-metric on position drift.
    inv_w = 1.0 / (1.0 + ((v - u_obs_p) ** 2) / (params.V_star ** 2))

    x_new = np.mod(x - dxPhi * inv_w * dt, L)
    v_new = v - dvPhi * dt
    return x_new, v_new


# ---------------------------------------------------------------------------
# Formulation B — direction-split SDE.
# ---------------------------------------------------------------------------


@dataclass
class FormulationBParams:
    """Strengths for Formulation B (revised PDF, eqs. 17-18).

    `gamma_x`, `gamma_v` are outer multipliers on the position- and
    velocity-channel drifts. `gamma_1, gamma_2, gamma_3` are the
    residual weights inside the split losses J_M^rho and J_M^{j,E}
    (matching A's notation; B uses gamma_1 for r_0, gamma_2 for r_1,
    gamma_3 for r_2).
    """
    gamma_x: float = 1.0
    gamma_v: float = 1.0
    gamma_1: float = 1.0
    gamma_2: float = 1.0
    gamma_3: float = 1.0
    lowpass_k_cut_frac: float = 0.25
    lowpass_sharpness: float = 16.0


def apply_formulation_B(
    x: np.ndarray, v: np.ndarray, w: np.ndarray, L: float, Nx: int,
    rho_obs: np.ndarray, u_obs: np.ndarray, T_obs: np.ndarray,
    params: FormulationBParams, dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Particle update from Formulation B (revised PDF, eqs. 17-18).

    Split losses:
      Phi_M^rho(x)     = gamma_1 (K_h * r_0)(x)                        (v-independent)
      Phi_M^{j,E}(x,v) = gamma_2 v (K_h * r_1)(x) + (gamma_3/2) v^2 (K_h * r_2)(x)

    Position channel uses x-gradient of Phi_M^rho:
      dx = -gamma_x * d/dx Phi_M^rho(x_p) * dt
         = -gamma_x * gamma_1 * g_0(x_p) * dt

    Velocity channel uses v-gradient of Phi_M^{j,E}:
      dv = -gamma_v * d/dv Phi_M^{j,E}(x_p, v_p) * dt
         = -gamma_v * (gamma_2 (K_h * r_1)(x_p) + gamma_3 v_p (K_h * r_2)(x_p)) * dt

    Both drifts vanish identically when r_0 = r_1 = r_2 = 0. The
    velocity-direction drift is identical to A's d_v Phi_M; B differs
    from A only by (i) dropping A's weighted-W2 metric on the position
    channel and (ii) dropping A's cross terms in the position drift.
    """
    rho_f, j_f, E_f = _model_moments(x, v, w, L, Nx)
    j_obs, E_obs = _conserved_obs(rho_obs, u_obs, T_obs)
    r0 = rho_f - rho_obs
    r1 = j_f - j_obs
    r2 = E_f - E_obs
    Kr0 = _filter_or_passthrough(r0, L, params.lowpass_k_cut_frac, params.lowpass_sharpness)
    Kr1 = _filter_or_passthrough(r1, L, params.lowpass_k_cut_frac, params.lowpass_sharpness)
    Kr2 = _filter_or_passthrough(r2, L, params.lowpass_k_cut_frac, params.lowpass_sharpness)
    g0 = grad_1d(Kr0, L)

    # Interpolate to particles.
    g0_p = cic_interpolate(g0, x, L)
    Kr1_p = cic_interpolate(Kr1, x, L)
    Kr2_p = cic_interpolate(Kr2, x, L)

    # Position drift: only r_0 contribution, no weighted metric, no cross terms.
    x_new = np.mod(x - params.gamma_x * params.gamma_1 * g0_p * dt, L)

    # Velocity drift: same v-gradient as A, scaled by outer gamma_v.
    dvPhi = params.gamma_2 * Kr1_p + params.gamma_3 * v * Kr2_p
    v_new = v - params.gamma_v * dvPhi * dt
    return x_new, v_new


# ---------------------------------------------------------------------------
# Formulation C — Maxwellian-projected KL nudging.
# ---------------------------------------------------------------------------


@dataclass
class FormulationCParams:
    """Parameters for Formulation C (Maxwellian-projected KL).

    H[f] = KL(Pi_M f || pi_obs); first variation Phi_M^(c) = A + B v + (1/2) C v^2
    with Bregman coefficients (1D, d=1):

        A = log(rho_f̃/rho_obs) + 1 - (1/2) log(T_f̃/T_obs)
            + (u_obs^2)/(2 T_obs) - (u_f̃^2)/(2 T_f̃)
        B = u_f̃/T_f̃ - u_obs/T_obs
        C = 1/T_obs - 1/T_f̃

    where (rho_f̃, u_f̃, T_f̃) are derived from K_h-smoothed conserved
    moments. Particle drift (matches A's structure):

        dx_extra = -lam * inv_metric * [grad_x A + V * grad_x B
                                         + (1/2) V^2 * grad_x C] dt
        dv_extra = -lam * [B + C * V] dt

    `inv_metric` = 1 / (1 + (V - u_obs)^2 / V_star^2) when
    `use_weighted_metric=True`, else 1.

    `lam` is the nudging strength. `rho_floor`, `T_floor` are required
    (Bregman coefficients have log/reciprocal of rho_f̃ and T_f̃).
    """
    lam: float = 1.0
    use_weighted_metric: bool = True
    V_star: float = 1.0
    lowpass_k_cut_frac: float = 0.25
    lowpass_sharpness: float = 16.0
    rho_floor: float = RHO_FLOOR_DEFAULT
    T_floor: float = T_FLOOR_DEFAULT


def _smoothed_primitive_moments(
    x: np.ndarray, v: np.ndarray, w: np.ndarray, L: float, Nx: int,
    lowpass_k_cut_frac: float, lowpass_sharpness: float,
    rho_floor: float, T_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (rho_f̃, u_f̃, T_f̃) on the grid via K_h-smoothed conserved
    moments. Floors keep them strictly positive so log/reciprocal in
    the Bregman coefficients are well defined.

    For 1D: T = 2 * E_f / rho_f - u_f^2 (where E_f = (1/2)<v^2 f>).
    """
    rho_raw, j_raw, E_raw = _model_moments(x, v, w, L, Nx)
    rho_t = _filter_or_passthrough(rho_raw, L, lowpass_k_cut_frac, lowpass_sharpness)
    j_t = _filter_or_passthrough(j_raw, L, lowpass_k_cut_frac, lowpass_sharpness)
    E_t = _filter_or_passthrough(E_raw, L, lowpass_k_cut_frac, lowpass_sharpness)

    rho_safe = np.maximum(rho_t, rho_floor)
    u_t = j_t / rho_safe
    T_t_raw = 2.0 * (E_t - 0.5 * rho_t * u_t * u_t) / rho_safe
    T_t = np.maximum(T_t_raw, T_floor)
    return rho_safe, u_t, T_t


def _bregman_coefficients(
    rho_f: np.ndarray, u_f: np.ndarray, T_f: np.ndarray,
    rho_obs: np.ndarray, u_obs: np.ndarray, T_obs: np.ndarray,
    rho_floor: float, T_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (A, B, C) on the grid from primitive moments (1D)."""
    rho_obs_safe = np.maximum(rho_obs, rho_floor)
    T_obs_safe = np.maximum(T_obs, T_floor)
    A = (np.log(rho_f / rho_obs_safe) + 1.0
         - 0.5 * np.log(T_f / T_obs_safe)
         + 0.5 * u_obs * u_obs / T_obs_safe
         - 0.5 * u_f * u_f / T_f)
    B = u_f / T_f - u_obs / T_obs_safe
    C = 1.0 / T_obs_safe - 1.0 / T_f
    return A, B, C


def apply_formulation_C(
    x: np.ndarray, v: np.ndarray, w: np.ndarray, L: float, Nx: int,
    rho_obs: np.ndarray, u_obs: np.ndarray, T_obs: np.ndarray,
    params: FormulationCParams, dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Particle update from Formulation C (Maxwellian-projected KL).

    Coefficients A, B, C are evaluated on the K_h-smoothed model
    moments. Drift has the same A + B v + (1/2) C v^2 phase-space
    structure as Formulation A but with Bregman coefficients in place
    of A's linear-in-residual coefficients.
    """
    rho_f, u_f, T_f = _smoothed_primitive_moments(
        x, v, w, L, Nx,
        params.lowpass_k_cut_frac, params.lowpass_sharpness,
        params.rho_floor, params.T_floor,
    )
    A, B, C = _bregman_coefficients(
        rho_f, u_f, T_f, rho_obs, u_obs, T_obs,
        params.rho_floor, params.T_floor,
    )
    grad_A = grad_1d(A, L)
    grad_B = grad_1d(B, L)
    grad_C = grad_1d(C, L)

    # Interpolate to particles.
    B_p = cic_interpolate(B, x, L)
    C_p = cic_interpolate(C, x, L)
    grad_A_p = cic_interpolate(grad_A, x, L)
    grad_B_p = cic_interpolate(grad_B, x, L)
    grad_C_p = cic_interpolate(grad_C, x, L)

    # Velocity drift: -lam * (B + C*V) dt.
    v_new = v - params.lam * (B_p + C_p * v) * dt

    # Position drift: -lam * inv_metric * [grad_A + V grad_B + (1/2) V^2 grad_C] dt.
    dx_drift = grad_A_p + v * grad_B_p + 0.5 * v * v * grad_C_p
    if params.use_weighted_metric:
        u_obs_p = cic_interpolate(u_obs, x, L)
        inv_metric = 1.0 / (1.0 + ((v - u_obs_p) ** 2) / (params.V_star ** 2))
        dx_drift = inv_metric * dx_drift
    x_new = np.mod(x - params.lam * dx_drift * dt, L)
    return x_new, v_new


# ---------------------------------------------------------------------------
# Baseline — classical constant-gain Azouani-Olson-Titi (AOT) moment nudging.
# ---------------------------------------------------------------------------


@dataclass
class FormulationAOTParams:
    """Constant-gain Luenberger / AOT feedback on the primitive moments.

    This is the *classical* continuous-data-assimilation baseline (the
    feedback that Theorem 6.1 generalizes): a constant-gain correction
    that drives the assim primitive moments (rho_f, u_f, T_f) toward the
    observed (rho_obs, u_obs, T_obs), with NO variational / information-
    geometric structure. Unlike Formulation C it has no canonical rate
    ratio: the bulk-velocity and temperature relaxation rates are the two
    independent gains mu_u, mu_T (so in the homogeneous benchmark it gives
    a *tunable* velocity:temperature ratio mu_u : mu_T, in contrast to C's
    fixed 1:2 from the Fisher-Rao geometry).

        bulk-velocity:  v += -mu_u * (u_f(x_p) - u_obs(x_p)) dt
        temperature:    p  = v - u_f;  p += -0.5 * mu_T * (T_f-T_obs)/T_f * p dt
                        (contracts/expands the peculiar velocity so
                         dT/dt = -mu_T (T_f - T_obs), shape-preserving affine)
        density:        x += -mu_rho * d/dx (K_h*(rho_f - rho_obs))(x_p) dt

    All three corrections vanish when the moments match, so AOT is
    residual-driven like A/B/C. Deterministic (no RNG).
    """
    mu_rho: float = 1.0
    mu_u: float = 1.0
    mu_T: float = 1.0
    lowpass_k_cut_frac: float = 0.25
    lowpass_sharpness: float = 16.0
    rho_floor: float = RHO_FLOOR_DEFAULT
    T_floor: float = T_FLOOR_DEFAULT


def apply_aot(
    x: np.ndarray, v: np.ndarray, w: np.ndarray, L: float, Nx: int,
    rho_obs: np.ndarray, u_obs: np.ndarray, T_obs: np.ndarray,
    params: FormulationAOTParams, dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Classical constant-gain AOT nudging on primitive moments over one dt."""
    rho_f, u_f, T_f = _smoothed_primitive_moments(
        x, v, w, L, Nx,
        params.lowpass_k_cut_frac, params.lowpass_sharpness,
        params.rho_floor, params.T_floor,
    )
    # Density-residual gradient for the position channel.
    r0 = rho_f - rho_obs
    Kr0 = _filter_or_passthrough(
        r0, L, params.lowpass_k_cut_frac, params.lowpass_sharpness)
    g0_p = cic_interpolate(grad_1d(Kr0, L), x, L)

    # Primitive moment fields at particles.
    u_f_p = cic_interpolate(u_f, x, L)
    u_obs_p = cic_interpolate(u_obs, x, L)
    T_f_p = cic_interpolate(T_f, x, L)
    T_obs_p = cic_interpolate(np.maximum(T_obs, params.T_floor), x, L)

    # Velocity channel: bulk-velocity nudge + temperature (peculiar-velocity) nudge.
    bulk = params.mu_u * (u_f_p - u_obs_p)
    temp = 0.5 * params.mu_T * ((T_f_p - T_obs_p) / T_f_p) * (v - u_f_p)
    v_new = v - (bulk + temp) * dt

    # Position channel: density nudge down the residual gradient.
    x_new = np.mod(x - params.mu_rho * g0_p * dt, L)
    return x_new, v_new


# ---------------------------------------------------------------------------
# Baseline — naive KL(f || pi_obs) relaxation toward the OBSERVED Maxwellian.
# ---------------------------------------------------------------------------


@dataclass
class FormulationNaiveKLParams:
    """Naive relaxation toward the observed Maxwellian pi_obs = M(u_obs, T_obs).

    This is the paper's *excluded* baseline (Table 1 caption): a feedback
    whose zero set is the single observed Maxwellian pi_obs, NOT the moment
    manifold. Because it relaxes f toward a Maxwellian, it drives a correct
    NON-equilibrium truth (e.g. a two-stream state) AWAY from the truth even
    though the moments already match. Implemented as a Monte-Carlo BGK-style
    relaxation toward pi_obs at rate `lam` (the discrete analogue of the
    W2-gradient flow of KL(f || pi_obs), which has pi_obs as its fixed point).
    Stochastic: requires an RNG.
    """
    lam: float = 1.0
    rho_floor: float = RHO_FLOOR_DEFAULT
    T_floor: float = T_FLOOR_DEFAULT


def apply_naive_kl(
    x: np.ndarray, v: np.ndarray, w: np.ndarray, L: float, Nx: int,
    rho_obs: np.ndarray, u_obs: np.ndarray, T_obs: np.ndarray,
    params: FormulationNaiveKLParams, dt: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Naive KL relaxation toward pi_obs = M(u_obs, T_obs) over one dt.

    With probability p = 1 - exp(-lam*dt) each particle's velocity is replaced
    by a fresh draw from N(u_obs(x_p), T_obs(x_p)). Position is unchanged.
    The fixed point is the observed Maxwellian, so this Maxwellianizes any
    non-Maxwellian shape regardless of whether the moments already agree.
    """
    u_obs_p = cic_interpolate(u_obs, x, L)
    T_obs_p = np.maximum(cic_interpolate(T_obs, x, L), params.T_floor)
    p_replace = 1.0 - np.exp(-params.lam * dt)
    draw = rng.uniform(size=x.shape[0])
    mask = draw < p_replace
    v_new = v.copy()
    if mask.any():
        sigma = np.sqrt(T_obs_p[mask])
        v_new[mask] = u_obs_p[mask] + sigma * rng.standard_normal(int(mask.sum()))
    return x.copy(), v_new


