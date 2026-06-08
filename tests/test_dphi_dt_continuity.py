"""Test that the continuity-based d phi / dt reconstruction matches analytic.

From the doc, eq. 14:  -Delta (d phi / dt) = -d j / dx,
so given j(x) on the grid, solve_poisson_from_div(j, L) returns d phi / dt
directly.

For j(x) = sin(k x), the analytic answer is d phi / dt = -cos(k x) / k.
This is the unit test prescribed in docs/combined_nudge_amendment.md §3.4.

(It is essentially the same property already exercised in
test_poisson_from_div.py; we restate it here under the dphi_dt name to make
the continuity-link explicit, and add a multi-mode check.)
"""
from __future__ import annotations

import numpy as np

from mfda.poisson import solve_poisson_from_div


def test_dphi_dt_recovers_analytic_for_sin_current() -> None:
    Nx = 256
    L = 2.0 * np.pi
    x = np.linspace(0.0, L, Nx, endpoint=False)
    for n_mode in (1, 2, 4, 7):
        k = 2.0 * np.pi * n_mode / L
        j = np.sin(k * x)
        dphi_dt = solve_poisson_from_div(j, L)
        analytic = -np.cos(k * x) / k
        analytic = analytic - analytic.mean()  # zero-mean gauge
        assert np.allclose(dphi_dt, analytic, atol=1e-12), f"n_mode={n_mode}"


def test_dphi_dt_zero_mean() -> None:
    rng = np.random.default_rng(42)
    j = rng.standard_normal(128)
    dphi_dt = solve_poisson_from_div(j, 3.5)
    assert abs(dphi_dt.mean()) < 1e-12


def test_dphi_dt_linearity() -> None:
    Nx = 64
    L = 4.0
    x = np.linspace(0.0, L, Nx, endpoint=False)
    j1 = np.sin(2 * np.pi * x / L)
    j2 = np.cos(2 * np.pi * x / L * 3)
    a, b = 0.7, -1.3
    out_combined = solve_poisson_from_div(a * j1 + b * j2, L)
    out_separate = a * solve_poisson_from_div(j1, L) + b * solve_poisson_from_div(j2, L)
    assert np.allclose(out_combined, out_separate, atol=1e-13)
