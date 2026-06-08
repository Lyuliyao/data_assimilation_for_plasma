"""External F_ext drivers for 2D2V runs.

Each driver returns a callable E_ext(x_grid, y_grid, t) -> (Ex, Ey) on the
2D grid. Four flavours:

  oblique_wave                 : U = A cos(kx x + ky y - omega t + theta),
                                 F = -grad U  =>  F = A |k| sin(.) k_hat * |k|
                                 (i.e. (Ex, Ey) = A (kx, ky) sin(.)).
  oblique_wavepacket           : same shape, with Gaussian-in-t envelope.
  checkerboard_standing        : U = A cos(kx x) cos(ky y) cos(omega t).
  oblique_traveling_continuous : F = A g(t) sin(kx x + ky y - omega t + theta)
                                 * (kx, ky) / |k|,
                                 g(t) = 0.5 (1 - cos(pi t / t_ramp)) for t<t_ramp,
                                 g(t) = 1 otherwise (smooth ramp-on, then
                                 persistent traveling wave).  Note: A here is
                                 the *force* amplitude (units E), so the
                                 potential equivalent is U_eq = -A cos(.)/|k|.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

EExtFunc2D = Callable[[np.ndarray, np.ndarray, float], tuple[np.ndarray, np.ndarray]]


def make_oblique_wave(
    A: float, kx: float, ky: float, omega: float, theta: float = 0.0,
) -> EExtFunc2D:
    """U = A cos(kx x + ky y - omega t + theta), F = -grad U."""
    def E_ext(x_grid: np.ndarray, y_grid: np.ndarray, t: float):
        XG, YG = np.meshgrid(x_grid, y_grid, indexing="ij")
        phase = kx * XG + ky * YG - omega * t + theta
        s = np.sin(phase)
        return A * kx * s, A * ky * s

    return E_ext


def make_oblique_wavepacket(
    A: float, kx: float, ky: float, omega: float,
    t0: float, sigma_t: float, theta: float = 0.0,
) -> EExtFunc2D:
    """U = A exp(-(t-t0)^2/(2 sigma_t^2)) cos(kx x + ky y - omega t + theta).
    Localised oblique traveling wave packet."""
    def E_ext(x_grid: np.ndarray, y_grid: np.ndarray, t: float):
        env = float(np.exp(-((t - t0) ** 2) / (2.0 * sigma_t * sigma_t)))
        XG, YG = np.meshgrid(x_grid, y_grid, indexing="ij")
        phase = kx * XG + ky * YG - omega * t + theta
        s = np.sin(phase)
        return A * env * kx * s, A * env * ky * s

    return E_ext


def make_oblique_traveling_continuous(
    A: float, kx: float, ky: float, omega: float,
    ramp_time: float = 1.0, theta: float = 0.0,
) -> EExtFunc2D:
    """F = A g(t) sin(kx x + ky y - omega t + theta) (kx, ky) / |k|.

    g(t) = 0.5 (1 - cos(pi t / t_ramp)) for t < t_ramp, else 1. A smooth
    half-cosine ramp-on with no envelope decay -- the wave persists for all
    t >= t_ramp.

    Note A is the *force* amplitude (not a potential amplitude). The
    equivalent potential is U_eq(x, y, t) = -A g(t) cos(.) / |k|.
    """
    k_mag = float(np.sqrt(kx * kx + ky * ky))
    if k_mag == 0.0:
        raise ValueError("oblique_traveling_continuous requires non-zero (kx, ky).")
    nx = kx / k_mag
    ny = ky / k_mag
    t_ramp = max(float(ramp_time), 1e-12)

    def E_ext(x_grid: np.ndarray, y_grid: np.ndarray, t: float):
        if t < t_ramp:
            g = 0.5 * (1.0 - float(np.cos(np.pi * t / t_ramp)))
        else:
            g = 1.0
        XG, YG = np.meshgrid(x_grid, y_grid, indexing="ij")
        phase = kx * XG + ky * YG - omega * t + theta
        s = np.sin(phase)
        amp = A * g
        return amp * nx * s, amp * ny * s

    return E_ext


def make_checkerboard_standing(
    A: float, kx: float, ky: float, omega: float,
) -> EExtFunc2D:
    """U = A cos(kx x) cos(ky y) cos(omega t).
    Standing-wave checkerboard pattern."""
    def E_ext(x_grid: np.ndarray, y_grid: np.ndarray, t: float):
        XG, YG = np.meshgrid(x_grid, y_grid, indexing="ij")
        cwt = float(np.cos(omega * t))
        Ex = A * kx * np.sin(kx * XG) * np.cos(ky * YG) * cwt
        Ey = A * ky * np.cos(kx * XG) * np.sin(ky * YG) * cwt
        return Ex, Ey

    return E_ext
