"""Unit tests for the nudging kernels."""
from __future__ import annotations

import numpy as np

from mfda.nudging import (
    apply_nudging,
    apply_position_nudging,
    apply_velocity_nudging,
)


def test_velocity_nudging_only_changes_v() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 2 * np.pi, 100)
    v = rng.standard_normal(100)
    g = rng.standard_normal(100)
    x1, v1 = apply_velocity_nudging(x, v, g, gamma=0.5, dt=0.1)
    assert np.allclose(x1, x)
    assert np.allclose(v1, v - 0.5 * g * 0.1)


def test_position_nudging_only_changes_x() -> None:
    L = 2 * np.pi
    rng = np.random.default_rng(0)
    x = rng.uniform(0, L, 100)
    v = rng.standard_normal(100)
    g = rng.standard_normal(100)
    x1, v1 = apply_position_nudging(x, v, g, gamma=0.5, dt=0.1, L=L)
    assert np.allclose(v1, v)
    assert np.all(x1 >= 0.0) and np.all(x1 < L)


def test_position_nudging_wraps() -> None:
    L = 2 * np.pi
    x = np.array([0.1])
    v = np.array([0.0])
    g = np.array([10.0])   # large positive grad pushes x negative of 0.1
    x1, _ = apply_position_nudging(x, v, g, gamma=1.0, dt=1.0, L=L)
    # Should wrap into [0, L).
    assert 0.0 <= x1[0] < L


def test_dispatch() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 10)
    v = rng.standard_normal(10)
    g = rng.standard_normal(10)
    x_n, v_n = apply_nudging("none", x, v, g, gamma=1.0, dt=0.1)
    assert np.allclose(x_n, x)
    assert np.allclose(v_n, v)

    x_v, v_v = apply_nudging("velocity", x, v, g, gamma=1.0, dt=0.1)
    assert np.allclose(x_v, x)
    assert np.allclose(v_v, v - g * 0.1)

    x_p, v_p = apply_nudging("position", x, v, g, gamma=1.0, dt=0.1, L=1.0)
    assert np.allclose(v_p, v)
