"""Smoke tests for the reference backend."""
from __future__ import annotations

import numpy as np

from mfda.backend_reference import (
    cic_deposit,
    cic_interpolate,
    field_solve,
    make_state,
    push_leapfrog_drift,
    push_leapfrog_half,
)
from mfda.initial_conditions import perturbed_maxwellian


def test_cic_deposit_is_mean_preserving() -> None:
    L = 2.0 * np.pi
    Nx = 64
    rng = np.random.default_rng(0)
    Np = 5000
    x = rng.uniform(0, L, Np)
    w = np.ones(Np) * (L / Np)  # total weight = L => mean(rho) = 1
    rho = cic_deposit(x, w, L, Nx)
    assert abs(rho.mean() - 1.0) < 5e-2


def test_cic_interp_roundtrip() -> None:
    L = 2.0 * np.pi
    Nx = 128
    x = np.linspace(0.0, L, Nx, endpoint=False)
    field = np.sin(x)
    # Sample the field at the grid points themselves.
    samp = cic_interpolate(field, x, L)
    assert np.allclose(samp, field, atol=1e-12)


def test_short_run_reference_backend_stable() -> None:
    """20-step run of linear Landau should stay finite and conserve mass."""
    L = 2.0 * np.pi / 0.5
    Nx = 64
    Np = 20000
    dt = 1e-2
    rng = np.random.default_rng(0)
    x, v, w = perturbed_maxwellian(Np, L, k=0.5, alpha=1e-3, rng=rng)
    state = make_state(x, v, w, L, Nx, dt)
    m0 = state.w.sum()
    for _ in range(20):
        push_leapfrog_half(state, 0.5)
        push_leapfrog_drift(state)
        field_solve(state)
        push_leapfrog_half(state, 0.5)
    assert np.isfinite(state.phi).all()
    assert np.isfinite(state.E).all()
    assert abs(state.w.sum() - m0) < 1e-12
