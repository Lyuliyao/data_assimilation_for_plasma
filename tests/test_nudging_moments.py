"""Verify A/B/C: zero residual ⇒ zero deterministic update; smoke for shapes."""
from __future__ import annotations

import numpy as np

from mfda.collisions import grid_moments_1d1v
from mfda.backend_reference import ReferenceState, field_solve, normalize_weights
from mfda.nudging_moments import (
    FormulationAParams,
    FormulationBParams,
    FormulationCParams,
    apply_formulation_A,
    apply_formulation_B,
    apply_formulation_C,
    _bregman_coefficients,
)


def _make_state(x, v, w, L, Nx) -> ReferenceState:
    w = normalize_weights(w.copy(), L)
    s = ReferenceState(x=x.copy(), v=v.copy(), w=w, L=L, Nx=Nx, dt=0.1)
    field_solve(s)
    return s


def _state_with_assim_obs_match():
    """Build assim particles, then read its own moments back as the
    'observation'. Then any A/B/C residual is identically zero."""
    rng = np.random.default_rng(0)
    L = 2.0 * np.pi
    Nx = 64
    Np = 50_000
    u_star, T_star = 0.2, 1.0
    x = rng.uniform(0.0, L, Np)
    v = u_star + np.sqrt(T_star) * rng.standard_normal(Np)
    w = np.ones(Np)
    state = _make_state(x, v, w, L, Nx)
    rho_obs, u_obs, T_obs = grid_moments_1d1v(state)
    return state, rho_obs, u_obs, T_obs, L, Nx


def test_formulation_A_zero_residual_no_update() -> None:
    state, rho_o, u_o, T_o, L, Nx = _state_with_assim_obs_match()
    params = FormulationAParams()
    x_new, v_new = apply_formulation_A(
        state.x, state.v, state.w, L, Nx,
        rho_o, u_o, T_o, params, dt=0.1,
    )
    # Residuals are zero up to deposit/round-off → updates are zero up to
    # numerical noise (filtering preserves zero).
    assert np.allclose(x_new, state.x, atol=1e-10)
    assert np.allclose(v_new, state.v, atol=1e-10)


def test_formulation_B_zero_residual_no_update() -> None:
    """Revised B (deterministic): residuals zero ⇒ both x and v unchanged."""
    state, rho_o, u_o, T_o, L, Nx = _state_with_assim_obs_match()
    params = FormulationBParams()
    x_new, v_new = apply_formulation_B(
        state.x, state.v, state.w, L, Nx,
        rho_o, u_o, T_o, params, dt=0.1,
    )
    assert np.allclose(x_new, state.x, atol=1e-10)
    assert np.allclose(v_new, state.v, atol=1e-10)


def test_formulation_C_zero_residual_no_update() -> None:
    """Maxwellian-projected C: when smoothed (rho_f̃, u_f̃, T_f̃) match
    (rho_obs, u_obs, T_obs), the Bregman coefficients (A=1, B=0, C=0)
    are constant in x ⇒ ∇A=∇B=∇C=0 and B=C=0 at every particle ⇒
    both x and v updates are identically zero. To avoid model-side K_h
    smoothing changing the moments relative to the (unsmoothed) obs from
    grid_moments_1d1v, we disable the lowpass for this test."""
    state, rho_o, u_o, T_o, L, Nx = _state_with_assim_obs_match()
    params = FormulationCParams(lam=1.0, lowpass_k_cut_frac=1.0)
    x_new, v_new = apply_formulation_C(
        state.x, state.v, state.w, L, Nx,
        rho_o, u_o, T_o, params, dt=0.05,
    )
    assert np.allclose(x_new, state.x, atol=1e-8)
    assert np.allclose(v_new, state.v, atol=1e-8)


def test_bregman_coefficients_vanish_at_match() -> None:
    """A=1 (constant), B=0, C=0 when (rho_f, u_f, T_f) = (rho_obs, u_obs, T_obs)."""
    Nx = 32
    rho = np.ones(Nx)
    u = 0.1 * np.sin(np.arange(Nx) * 2 * np.pi / Nx)
    T = np.ones(Nx)
    A, B, C = _bregman_coefficients(rho, u, T, rho, u, T,
                                     rho_floor=1e-3, T_floor=1e-3)
    assert np.allclose(A, 1.0)
    assert np.allclose(B, 0.0, atol=1e-12)
    assert np.allclose(C, 0.0, atol=1e-12)


def test_all_three_formulations_smoke_small_Np() -> None:
    """Tiny call to confirm shapes and no NaN propagation across all three."""
    rng = np.random.default_rng(0)
    L = 2.0 * np.pi
    Nx = 16
    Np = 200
    x = rng.uniform(0.0, L, Np)
    v = rng.standard_normal(Np)
    w = normalize_weights(np.ones(Np), L)
    rho_obs = np.ones(Nx) + 0.1 * np.cos(np.arange(Nx) * 2 * np.pi / Nx)
    u_obs = 0.05 * np.sin(np.arange(Nx) * 2 * np.pi / Nx)
    T_obs = np.ones(Nx)

    xa, va = apply_formulation_A(x, v, w, L, Nx, rho_obs, u_obs, T_obs,
                                  FormulationAParams(), dt=0.05)
    assert np.all(np.isfinite(xa)) and np.all(np.isfinite(va))

    xb, vb = apply_formulation_B(x, v, w, L, Nx, rho_obs, u_obs, T_obs,
                                  FormulationBParams(), dt=0.05)
    assert np.all(np.isfinite(xb)) and np.all(np.isfinite(vb))

    xc, vc = apply_formulation_C(x, v, w, L, Nx,
                                  rho_obs, u_obs, T_obs,
                                  FormulationCParams(), dt=0.05)
    assert np.all(np.isfinite(xc)) and np.all(np.isfinite(vc))


def test_formulation_A_sign_pushes_particles_away_from_density_peak() -> None:
    """With assim density above observed at x_0, A's gradient flow should
    push nearby particles *away* from x_0 (decreasing the assim density
    there). This pins the sign convention of d_x Phi_M in apply_formulation_A.
    """
    L = 2.0 * np.pi
    Nx = 64
    rng = np.random.default_rng(0)
    Np = 50_000
    # Uniform assim particles + normalised weights so rho_f mean = 1.
    x = rng.uniform(0.0, L, Np)
    v = rng.standard_normal(Np)
    w = normalize_weights(np.ones(Np), L)
    # Make rho_obs a dip at x_0 = L/2: r_0 = rho_f - rho_obs has a positive
    # bump there. Smooth gauss so the gradient is well-defined.
    x_grid = np.arange(Nx) * (L / Nx)
    x0 = 0.5 * L
    width = 0.5
    amp = 0.3
    rho_obs = 1.0 - amp * np.exp(-((x_grid - x0) ** 2) / (2 * width ** 2))
    u_obs = np.zeros(Nx)
    T_obs = np.ones(Nx)
    # gamma_2 = gamma_3 = 0 isolates the rho-residual contribution.
    # Small dt so no particle wraps periodically.
    params = FormulationAParams(gamma_1=1.0, gamma_2=0.0, gamma_3=0.0,
                                lowpass_k_cut_frac=0.5)
    x_new, _ = apply_formulation_A(x, v, w, L, Nx,
                                    rho_obs, u_obs, T_obs, params, dt=0.05)
    # Unwrap displacement to (-L/2, L/2] to avoid periodic-wrap artefacts.
    dx = x_new - x
    dx = np.where(dx > L / 2, dx - L, dx)
    dx = np.where(dx < -L / 2, dx + L, dx)
    near_right = (x > x0 + 0.1) & (x < x0 + 0.6)
    near_left = (x < x0 - 0.1) & (x > x0 - 0.6)
    dx_right = float(np.mean(dx[near_right]))
    dx_left = float(np.mean(dx[near_left]))
    assert dx_right > 0, (
        f"expected positive mean dx to the right of peak, got {dx_right}"
    )
    assert dx_left < 0, (
        f"expected negative mean dx to the left of peak, got {dx_left}"
    )


def test_formulation_C_drift_pulls_velocity_toward_u_obs() -> None:
    """Maxwellian-projected C: assim at u_star=0.3, pi_obs at u=0, both T=1.
    Expected: B = u_f/T_f - u_obs/T_obs ≈ 0.3, C ≈ 0,
    so dv = -lam*(B + C*v)*dt = -lam*0.3*dt drags v toward 0."""
    rng = np.random.default_rng(0)
    L = 2.0 * np.pi
    Nx = 64
    Np = 100_000
    u_assim = 0.3
    x = rng.uniform(0.0, L, Np)
    v = u_assim + rng.standard_normal(Np)
    w = normalize_weights(np.ones(Np), L)
    rho_obs = np.ones(Nx)
    u_obs = np.zeros(Nx)
    T_obs = np.ones(Nx)
    params = FormulationCParams(lam=5.0, use_weighted_metric=False)
    _, v_new = apply_formulation_C(
        x, v, w, L, Nx, rho_obs, u_obs, T_obs, params, dt=0.05,
    )
    mean_v_before = float(np.mean(v))
    mean_v_after = float(np.mean(v_new))
    # Expected shift ≈ -5 * 0.3 * 0.05 = -0.075 → mean_v_after ≈ 0.225
    assert mean_v_after < mean_v_before - 0.02, (
        f"expected C to pull mean_v toward 0; before={mean_v_before:.3f}, "
        f"after={mean_v_after:.3f}"
    )
    assert mean_v_after > -0.1, (
        f"C pulled mean_v all the way past 0: after={mean_v_after:.3f}"
    )
