"""Verify LB substep: (a) mass conservation, (b) homogeneous-Maxwellian fixed point,
(c) zero nu is a no-op, (d) cold start relaxes toward local Maxwellian variance."""
from __future__ import annotations

import numpy as np

from mfda.backend_reference import ReferenceState, field_solve, normalize_weights
from mfda.collisions import grid_moments_1d1v, lb_substep


def _make_state(x, v, w, L, Nx, dt) -> ReferenceState:
    w = normalize_weights(w.copy(), L)
    s = ReferenceState(x=x.copy(), v=v.copy(), w=w, L=L, Nx=Nx, dt=dt)
    field_solve(s)
    return s


def test_lb_conserves_total_mass_exactly() -> None:
    """LB only updates velocities; weights and positions don't change."""
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
    lb_rng = np.random.default_rng(42)
    for _ in range(20):
        lb_substep(state, nu, lb_rng)
    assert state.w.sum() == m0


def test_lb_zero_nu_is_noop() -> None:
    rng = np.random.default_rng(0)
    L = 2.0 * np.pi
    Nx = 32
    Np = 1000
    x = rng.uniform(0.0, L, Np)
    v = rng.standard_normal(Np)
    w = np.ones(Np)
    state = _make_state(x, v, w, L, Nx, dt=0.1)
    v_before = state.v.copy()
    lb_substep(state, nu=0.0, rng=np.random.default_rng(1))
    assert np.array_equal(state.v, v_before)


def test_lb_homogeneous_maxwellian_is_fixed_point() -> None:
    """A homogeneous Maxwellian at (u_star, T_star) should remain so on average
    under LB: the OU equilibrium IS the local Maxwellian target."""
    rng = np.random.default_rng(0)
    L = 2.0 * np.pi
    Nx = 32
    Np = 200_000
    dt = 0.1
    nu = 1.0
    u_star, T_star = 0.3, 1.4
    n_steps = int(np.ceil(5.0 / (nu * dt)))

    x = rng.uniform(0.0, L, Np)
    v = u_star + np.sqrt(T_star) * rng.standard_normal(Np)
    w = np.ones(Np)
    state = _make_state(x, v, w, L, Nx, dt)
    lb_rng = np.random.default_rng(7)
    for _ in range(n_steps):
        lb_substep(state, nu, lb_rng)

    rho_g, u_g, T_g = grid_moments_1d1v(state)
    u_bulk = float(np.sum(rho_g * u_g) / np.sum(rho_g))
    T_bulk = float(np.sum(rho_g * T_g) / np.sum(rho_g))
    assert abs(u_bulk - u_star) < 0.02 * abs(u_star) + 5e-3
    assert abs(T_bulk - T_star) / T_star < 0.02


def test_lb_cold_start_heats_toward_local_T() -> None:
    """Start with v ~ delta(0) but spread positions uniformly. The empirical
    local T = 0; with nu*dt small, OU diffusion alone *adds* variance, so T
    grows toward the empirical-self-consistent equilibrium (which is some
    finite value). Precise fixed point is messy because T is updated each
    substep; we just check T grows monotonically away from 0 in the early
    transient."""
    rng = np.random.default_rng(0)
    L = 2.0 * np.pi
    Nx = 32
    Np = 50_000
    dt = 0.05
    nu = 0.5
    x = rng.uniform(0.0, L, Np)
    # Tiny initial spread so empirical T_local is not literally zero (which
    # would zero out the diffusion).
    v = 0.01 * rng.standard_normal(Np)
    w = np.ones(Np)
    state = _make_state(x, v, w, L, Nx, dt)
    lb_rng = np.random.default_rng(13)

    Ts = []
    for _ in range(20):
        lb_substep(state, nu, lb_rng)
        _, _, T_g = grid_moments_1d1v(state)
        Ts.append(float(T_g.mean()))
    # T should grow over time (not strictly monotone with finite-Np noise,
    # but later average must dominate earlier).
    assert np.mean(Ts[10:]) > np.mean(Ts[:5])