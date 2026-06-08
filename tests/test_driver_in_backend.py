"""Verify E_ext_func is added to the self-consistent E inside half-kicks."""
from __future__ import annotations

import numpy as np

from mfda.backend_reference import ReferenceState, push_leapfrog_half


def test_constant_E_ext_kicks_resting_particle() -> None:
    """A particle at rest with state.E = 0 and constant E_ext = E0 should
    gain v = 0.5 * dt * E0 from one half-kick."""
    L = 2.0 * np.pi
    Nx = 16
    Np = 1
    dt = 0.1
    E0 = 1.5

    def E_ext(x_grid: np.ndarray, t: float) -> np.ndarray:
        return np.full_like(x_grid, E0)

    state = ReferenceState(
        x=np.array([0.5 * L]),
        v=np.array([0.0]),
        w=np.array([1.0]),
        L=L, Nx=Nx, dt=dt,
        E=np.zeros(Nx), phi=np.zeros(Nx),
        E_ext_func=E_ext,
    )
    push_leapfrog_half(state, 0.5)
    assert np.isclose(state.v[0], 0.5 * dt * E0, atol=1e-12)


def test_no_E_ext_yields_no_kick_when_E_zero() -> None:
    """Sanity: same setup without E_ext gives v unchanged."""
    L = 2.0 * np.pi
    Nx = 16
    dt = 0.1
    state = ReferenceState(
        x=np.array([0.5 * L]),
        v=np.array([0.0]),
        w=np.array([1.0]),
        L=L, Nx=Nx, dt=dt,
        E=np.zeros(Nx), phi=np.zeros(Nx),
    )
    push_leapfrog_half(state, 0.5)
    assert state.v[0] == 0.0
