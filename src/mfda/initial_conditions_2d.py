"""2D2V initial-condition samplers.

Conventions:
  - Particles uniformly distributed in (x, y); spatial perturbations enter
    via per-particle weights w (importance-sampling), matching the 1D
    convention.
  - Velocity (vx, vy) sampled from a 2D isotropic Maxwellian centred at
    (ux*, uy*) with variance T* per component.
  - All samplers return (x, y, vx, vy, w) of shape (Np,).
"""
from __future__ import annotations

import numpy as np


def perturbed_maxwellian_2d(
    Np: int, Lx: float, Ly: float, kx: float, ky: float,
    alpha: float = 0.01, sigma: float = 1.0,
    rng: np.random.Generator | None = None,
):
    """rho_0(x, y) = 1 + alpha cos(kx x + ky y), isotropic Maxwellian in v."""
    rng = rng or np.random.default_rng(0)
    x = rng.uniform(0.0, Lx, Np)
    y = rng.uniform(0.0, Ly, Np)
    vx = sigma * rng.standard_normal(Np)
    vy = sigma * rng.standard_normal(Np)
    w = 1.0 + alpha * np.cos(kx * x + ky * y)
    return x, y, vx, vy, w


def ic_phase_error_2d(
    Np: int, Lx: float, Ly: float, kx: float, ky: float,
    alpha: float = 0.01, theta0: float = 0.0,
    sigma: float = 1.0,
    rng: np.random.Generator | None = None,
):
    """rho_0(x, y) = 1 + alpha cos(kx x + ky y + theta0). Same v marginal."""
    rng = rng or np.random.default_rng(0)
    x = rng.uniform(0.0, Lx, Np)
    y = rng.uniform(0.0, Ly, Np)
    vx = sigma * rng.standard_normal(Np)
    vy = sigma * rng.standard_normal(Np)
    w = 1.0 + alpha * np.cos(kx * x + ky * y + theta0)
    return x, y, vx, vy, w


def ic_blob_2d(
    Np: int, Lx: float, Ly: float, x0: float, y0: float,
    epsilon: float = 0.3, sigma_blob: float = 0.5,
    sigma: float = 1.0,
    rng: np.random.Generator | None = None,
):
    """rho_0(x, y) = 1 + epsilon * exp(-((x-x0)^2 + (y-y0)^2)/(2 sigma_blob^2)).
    Centre (x0, y0). Maxwellian v with variance sigma^2 per component."""
    rng = rng or np.random.default_rng(0)
    x = rng.uniform(0.0, Lx, Np)
    y = rng.uniform(0.0, Ly, Np)
    vx = sigma * rng.standard_normal(Np)
    vy = sigma * rng.standard_normal(Np)
    # Periodic blob: take min over the 4 nearest periodic images.
    dx = x - x0
    dy = y - y0
    dx = dx - Lx * np.round(dx / Lx)
    dy = dy - Ly * np.round(dy / Ly)
    r2 = dx * dx + dy * dy
    w = 1.0 + epsilon * np.exp(-0.5 * r2 / (sigma_blob * sigma_blob))
    return x, y, vx, vy, w


def wrong_maxwellian_2d(
    Np: int, Lx: float, Ly: float, kx: float, ky: float,
    rho_amp: float = 0.0, theta0: float = 0.0,
    ux_star: float = 0.0, uy_star: float = 0.0, T_star: float = 1.0,
    rng: np.random.Generator | None = None,
):
    """Spatially-uniform (or mildly modulated) Maxwellian at (ux*, uy*, T*).
    Wrong-prior used to initialise the assim run."""
    rng = rng or np.random.default_rng(0)
    x = rng.uniform(0.0, Lx, Np)
    y = rng.uniform(0.0, Ly, Np)
    sigma = float(np.sqrt(T_star))
    vx = ux_star + sigma * rng.standard_normal(Np)
    vy = uy_star + sigma * rng.standard_normal(Np)
    if rho_amp != 0.0:
        w = 1.0 + rho_amp * np.cos(kx * x + ky * y + theta0)
    else:
        w = np.ones(Np)
    return x, y, vx, vy, w
