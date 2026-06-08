"""Verify BGK substep: (a) mass conservation, (b) homogeneous-Maxwellian fixed point."""
from __future__ import annotations

import numpy as np

from mfda.backend_reference import ReferenceState, field_solve, normalize_weights
from mfda.collisions import bgk_substep, grid_moments_1d1v


def _make_state(x, v, w, L, Nx, dt) -> ReferenceState:
    w = normalize_weights(w.copy(), L)
    s = ReferenceState(x=x.copy(), v=v.copy(), w=w, L=L, Nx=Nx, dt=dt)
    field_solve(s)
    return s


def test_bgk_conserves_total_mass_exactly() -> None:
    """BGK only resamples velocities; weights and positions don't change,
    so sum(w) is bit-exact preserved."""
    rng = np.random.default_rng(0)
    L = 2.0 * np.pi
    Nx = 64
    Np = 5000
    dt = 0.1
    nu = 1.0
    x = rng.uniform(0.0, L, Np)
    v = rng.standard_normal(Np)
    w = np.ones(Np)
    state = _make_state(x, v, w, L, Nx, dt)
    m0 = float(state.w.sum())
    bgk_rng = np.random.default_rng(42)
    for _ in range(20):
        bgk_substep(state, nu, bgk_rng)
    assert state.w.sum() == m0


def test_bgk_homogeneous_maxwellian_is_fixed_point() -> None:
    """Homogeneous Maxwellian sampled at (u_star, T_star) should remain a
    Maxwellian with empirical (u, T) close to (u_star, T_star) after many
    BGK substeps — the local Maxwellian target equals the empirical
    distribution in the homogeneous case, so BGK is a no-op in expectation.
    """
    rng = np.random.default_rng(0)
    L = 2.0 * np.pi
    Nx = 32
    Np = 200_000
    dt = 0.1
    nu = 1.0
    u_star, T_star = 0.3, 1.4
    n_steps = int(np.ceil(5.0 / (nu * dt)))  # t_phys = 5 / nu

    x = rng.uniform(0.0, L, Np)
    v = u_star + np.sqrt(T_star) * rng.standard_normal(Np)
    w = np.ones(Np)
    state = _make_state(x, v, w, L, Nx, dt)
    bgk_rng = np.random.default_rng(7)
    for _ in range(n_steps):
        bgk_substep(state, nu, bgk_rng)

    # Compare global empirical (u, T) against (u_star, T_star).
    rho_g, u_g, T_g = grid_moments_1d1v(state)
    u_bulk = float(np.sum(rho_g * u_g) / np.sum(rho_g))
    T_bulk = float(np.sum(rho_g * T_g) / np.sum(rho_g))
    # 2% tolerance — generous for finite-Np noise; key is that nothing
    # drifts away from the target.
    assert abs(u_bulk - u_star) < 0.02 * abs(u_star) + 5e-3
    assert abs(T_bulk - T_star) / T_star < 0.02
