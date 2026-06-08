"""Verify observe_moments recovers analytical (rho, u, T) from a known ensemble."""
from __future__ import annotations

import numpy as np

from mfda.backend_reference import ReferenceState, field_solve, normalize_weights
from mfda.observation_moments import MomentObservationSpec, observe_moments


def _make_state(x: np.ndarray, v: np.ndarray, w: np.ndarray,
                L: float, Nx: int) -> ReferenceState:
    w = normalize_weights(w.copy(), L)
    s = ReferenceState(x=x.copy(), v=v.copy(), w=w, L=L, Nx=Nx, dt=0.01)
    field_solve(s)
    return s


def test_uniform_density_constant_drift_and_temperature() -> None:
    rng = np.random.default_rng(0)
    L = 2.0 * np.pi
    Nx = 64
    Np = 200_000
    u_star, T_star = 0.4, 1.2
    x = rng.uniform(0.0, L, Np)
    v = u_star + np.sqrt(T_star) * rng.standard_normal(Np)
    w = np.ones(Np)
    state = _make_state(x, v, w, L, Nx)

    spec = MomentObservationSpec(kind="full")
    rho_o, u_o, T_o = observe_moments(state, spec)

    # Density: mean ~ 1 from the normalised weights.
    assert abs(rho_o.mean() - 1.0) < 5e-3
    # u and T: density-weighted mean across cells.
    rho_safe = np.maximum(rho_o, 1e-3)
    u_bulk = float(np.sum(rho_o * u_o) / np.sum(rho_o))
    T_bulk = float(np.sum(rho_o * T_o) / np.sum(rho_o))
    assert abs(u_bulk - u_star) < 5e-3
    assert abs(T_bulk - T_star) < 1e-2


def test_cosine_density_recovered() -> None:
    """rho(x) = 1 + alpha cos(kx) sampled by importance weights."""
    rng = np.random.default_rng(1)
    L = 2.0 * np.pi
    k = 1.0
    Nx = 128
    Np = 200_000
    alpha = 0.1
    u_star, T_star = 0.0, 1.0

    x = rng.uniform(0.0, L, Np)
    w = 1.0 + alpha * np.cos(k * x)
    v = u_star + np.sqrt(T_star) * rng.standard_normal(Np)
    state = _make_state(x, v, w, L, Nx)

    spec = MomentObservationSpec(kind="full")
    rho_o, _, _ = observe_moments(state, spec)

    # Project onto cos(kx) and sin(kx); sin coefficient should be ~ 0,
    # cos coefficient should be ~ alpha.
    x_grid = np.arange(Nx) * (L / Nx)
    a_cos = (2.0 / Nx) * float(np.sum(rho_o * np.cos(k * x_grid)))
    a_sin = (2.0 / Nx) * float(np.sum(rho_o * np.sin(k * x_grid)))
    assert abs(a_cos - alpha) < 0.02
    assert abs(a_sin) < 0.02
