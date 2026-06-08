"""Regression test: alpha=0 must reduce to snapshot-only nudging bit-exactly.

This is the merge gate. If apply_velocity_nudging or apply_nudging changes
output for the existing call signature when alpha=0, every committed result
in results/test0/ and results/test1/ becomes invalid.
"""
from __future__ import annotations

import numpy as np

from mfda.nudging import apply_nudging, apply_velocity_nudging


def _setup(seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 2.0 * np.pi, 1024)
    v = rng.standard_normal(1024)
    grad_psi0 = rng.standard_normal(1024)
    return x, v, grad_psi0


def test_apply_velocity_nudging_alpha_zero_bitexact() -> None:
    x, v, grad_psi0 = _setup(0)
    arbitrary = np.random.default_rng(99).standard_normal(x.shape)
    x1, v1 = apply_velocity_nudging(x, v, grad_psi0, gamma=1.3, dt=0.01)
    x2, v2 = apply_velocity_nudging(
        x, v, grad_psi0, gamma=1.3, dt=0.01,
        grad_psi1_at_particles=arbitrary,
        alpha=0.0,
    )
    assert np.array_equal(x1, x2)
    assert np.array_equal(v1, v2)


def test_apply_velocity_nudging_grad_psi1_none_bitexact() -> None:
    x, v, grad_psi0 = _setup(1)
    x1, v1 = apply_velocity_nudging(x, v, grad_psi0, gamma=0.7, dt=0.05)
    x2, v2 = apply_velocity_nudging(
        x, v, grad_psi0, gamma=0.7, dt=0.05,
        grad_psi1_at_particles=None,
        alpha=42.0,  # alpha is irrelevant when grad_psi1 is None
    )
    assert np.array_equal(x1, x2)
    assert np.array_equal(v1, v2)


def test_apply_nudging_dispatch_alpha_zero_bitexact() -> None:
    x, v, grad_psi0 = _setup(2)
    arbitrary = np.random.default_rng(11).standard_normal(x.shape)
    x1, v1 = apply_nudging("velocity", x, v, grad_psi0, gamma=2.0, dt=0.01)
    x2, v2 = apply_nudging(
        "velocity", x, v, grad_psi0, gamma=2.0, dt=0.01,
        grad_psi1_at_particles=arbitrary, alpha=0.0,
    )
    assert np.array_equal(x1, x2)
    assert np.array_equal(v1, v2)


def test_alpha_nonzero_actually_does_something() -> None:
    """Sanity: with alpha != 0 and grad_psi1 != None the velocity DOES change."""
    x, v, grad_psi0 = _setup(3)
    grad_psi1 = np.ones_like(v)  # uniform grad psi1 = 1
    x1, v1 = apply_velocity_nudging(x, v, grad_psi0, gamma=1.0, dt=0.1)
    x2, v2 = apply_velocity_nudging(
        x, v, grad_psi0, gamma=1.0, dt=0.1,
        grad_psi1_at_particles=grad_psi1,
        alpha=0.5,
    )
    # v2 should equal v1 - 0.05 (= -gamma * alpha * 1 * dt) per particle.
    assert np.allclose(v2 - v1, -0.05)
