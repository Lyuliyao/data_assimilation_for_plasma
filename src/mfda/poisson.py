"""Periodic 1D Poisson solver via rFFT.

Solves
    -d^2 phi / dx^2 = s(x),   phi periodic on [0, L],   mean(phi) = 0.

Also exports a derivative helper that returns -d phi / dx (i.e. the electric
field or, for the adjoint problem, -grad psi).

The same solver is used for two problems in this project:
    1. Forward Poisson:   -Delta phi = rho - 1
    2. Adjoint Poisson:   -Delta psi = phi_f - y

In both cases the source must have zero mean on a periodic domain for the
problem to be well-posed; the solver enforces this by zeroing the k=0 mode.
"""
from __future__ import annotations

import numpy as np


def solve_poisson_1d(source: np.ndarray, L: float) -> np.ndarray:
    """Solve -d^2 u / dx^2 = source on [0, L] with periodic BC and zero mean.

    Parameters
    ----------
    source : (Nx,) array
        The right-hand side, sampled on a uniform grid x_j = j * dx, dx = L/Nx.
    L : float
        Period.

    Returns
    -------
    u : (Nx,) array
        The periodic, zero-mean solution.
    """
    Nx = source.shape[0]
    k = 2.0 * np.pi * np.fft.rfftfreq(Nx, d=L / Nx)  # physical wavenumbers
    s_hat = np.fft.rfft(source)
    u_hat = np.zeros_like(s_hat)
    # Skip k=0 mode to enforce zero mean.
    k2 = k * k
    nonzero = k2 > 0
    u_hat[nonzero] = s_hat[nonzero] / k2[nonzero]
    return np.fft.irfft(u_hat, n=Nx)


def grad_1d(u: np.ndarray, L: float) -> np.ndarray:
    """Spectral derivative du/dx on a periodic grid of length L.

    Returns the mean-free derivative (the k=0 component is zero by construction).
    """
    Nx = u.shape[0]
    k = 2.0 * np.pi * np.fft.rfftfreq(Nx, d=L / Nx)
    u_hat = np.fft.rfft(u)
    du_hat = 1j * k * u_hat
    # Nyquist mode should be set to zero for real-valued derivatives when Nx is even.
    if Nx % 2 == 0:
        du_hat[-1] = 0.0
    return np.fft.irfft(du_hat, n=Nx)


def electric_field_from_density(rho: np.ndarray, L: float, background: float = 1.0) -> np.ndarray:
    """Convenience: return E(x) = -d phi / dx where -Delta phi = rho - background.

    Zero-mean enforcement on phi is automatic via solve_poisson_1d.
    """
    phi = solve_poisson_1d(rho - background, L)
    return -grad_1d(phi, L)


def potential_from_density(rho: np.ndarray, L: float, background: float = 1.0) -> np.ndarray:
    """Return phi with -Delta phi = rho - background, zero mean, periodic."""
    return solve_poisson_1d(rho - background, L)


def adjoint_potential(phi_f: np.ndarray, y: np.ndarray, L: float) -> np.ndarray:
    """Return psi with -Delta psi = phi_f - y, zero mean, periodic.

    The residual (phi_f - y) should already be mean-free in theory because both
    phi_f and y are zero-mean solutions of periodic Poisson. We nevertheless
    subtract the mean defensively to avoid accidental bias from discretisation
    / noise.
    """
    resid = phi_f - y
    resid = resid - resid.mean()
    return solve_poisson_1d(resid, L)


def solve_poisson_from_d2(
    M: np.ndarray, rho: np.ndarray, E: np.ndarray, L: float,
) -> np.ndarray:
    """Return d^2 phi / dt^2 via the continuity chain.

    Derivation: from -Delta phi = rho - 1 and the Vlasov equation,
        d_t j = -d M / dx + E * rho      (1D, q=m=1, E sign convention as in backend)
        d^2 rho / dt^2 = -d/dx (d_t j) = d^2 M / dx^2 - d/dx (rho * E)

    So
        -Delta (d^2 phi / dt^2) = d^2 M / dx^2 - d/dx (rho * E)

    and d^2 phi / dt^2 = solve_poisson_1d(d^2 M / dx^2 - d/dx(rho*E), L),
    with zero-mean enforced. M, rho, E are all defined on the same Nx grid.
    """
    d2_M = grad_1d(grad_1d(M, L), L)
    d_rhoE = grad_1d(rho * E, L)
    source = d2_M - d_rhoE
    source = source - source.mean()
    return solve_poisson_1d(source, L)


def solve_chi(rho: np.ndarray, grad_psi2: np.ndarray, L: float) -> np.ndarray:
    """Return chi with -Delta chi = grad . (rho grad psi_2), zero-mean.

    1D form: -Delta chi = d/dx (rho * d psi_2 / dx). Caller passes the
    pre-computed grad_psi2 = d psi_2 / dx on the same Nx grid as rho.
    """
    interior = rho * grad_psi2
    source = grad_1d(interior, L)
    source = source - source.mean()
    return solve_poisson_1d(source, L)


def solve_poisson_from_div(j: np.ndarray, L: float) -> np.ndarray:
    """Return u with -Delta u = -d j / dx on [0, L], periodic, zero mean.

    Spectrally: u_hat(k) = (-i k * j_hat(k)) / k^2 = -i j_hat(k) / k for k != 0,
    u_hat(0) = 0. The k=0 mode is dropped to enforce the zero-mean gauge.

    Used for the continuity-based reconstruction of d phi / dt: from
    d rho / dt = -d j / dx and -Delta phi = rho - 1, we get
    -Delta (d phi / dt) = -d j / dx, so this returns d phi / dt directly
    given the current density j.
    """
    Nx = j.shape[0]
    k = 2.0 * np.pi * np.fft.rfftfreq(Nx, d=L / Nx)
    j_hat = np.fft.rfft(j)
    u_hat = np.zeros_like(j_hat)
    k2 = k * k
    nonzero = k2 > 0
    u_hat[nonzero] = (-1j * k[nonzero] * j_hat[nonzero]) / k2[nonzero]
    if Nx % 2 == 0:
        u_hat[-1] = 0.0
    return np.fft.irfft(u_hat, n=Nx)
