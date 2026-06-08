"""Periodic 2D Poisson solver via 2D rFFT.

Solves
    -Laplacian u = source on [0, Lx] x [0, Ly], periodic BC, mean(u) = 0.

Same conventions as poisson.py (1D): zero-mean enforced by zeroing the
(kx, ky) = (0, 0) Fourier mode. Spectral gradient zeros Nyquist on each
axis when the corresponding grid count is even.
"""
from __future__ import annotations

import numpy as np


def _wavenumbers_2d(Nx: int, Ny: int, Lx: float, Ly: float):
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
    ky = 2.0 * np.pi * np.fft.rfftfreq(Ny, d=Ly / Ny)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    return KX, KY


def solve_poisson_2d(source: np.ndarray, Lx: float, Ly: float) -> np.ndarray:
    """Solve -Laplacian u = source, periodic on [0, Lx] x [0, Ly], zero mean."""
    Nx, Ny = source.shape
    KX, KY = _wavenumbers_2d(Nx, Ny, Lx, Ly)
    K2 = KX * KX + KY * KY
    s_hat = np.fft.rfft2(source)
    u_hat = np.zeros_like(s_hat)
    nonzero = K2 > 0
    u_hat[nonzero] = s_hat[nonzero] / K2[nonzero]
    return np.fft.irfft2(u_hat, s=(Nx, Ny))


def grad_2d(u: np.ndarray, Lx: float, Ly: float) -> tuple[np.ndarray, np.ndarray]:
    """Spectral gradient (du/dx, du/dy) on a periodic 2D grid."""
    Nx, Ny = u.shape
    KX, KY = _wavenumbers_2d(Nx, Ny, Lx, Ly)
    u_hat = np.fft.rfft2(u)
    dudx_hat = 1j * KX * u_hat
    dudy_hat = 1j * KY * u_hat
    if Nx % 2 == 0:
        dudx_hat[Nx // 2, :] = 0.0
    if Ny % 2 == 0:
        dudy_hat[:, -1] = 0.0
    dudx = np.fft.irfft2(dudx_hat, s=(Nx, Ny))
    dudy = np.fft.irfft2(dudy_hat, s=(Nx, Ny))
    return dudx, dudy


def electric_field_from_density_2d(
    rho: np.ndarray, Lx: float, Ly: float, background: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (Ex, Ey) = -grad phi where -Laplacian phi = rho - background."""
    phi = solve_poisson_2d(rho - background, Lx, Ly)
    dphidx, dphidy = grad_2d(phi, Lx, Ly)
    return -dphidx, -dphidy


def potential_from_density_2d(
    rho: np.ndarray, Lx: float, Ly: float, background: float = 1.0,
) -> np.ndarray:
    return solve_poisson_2d(rho - background, Lx, Ly)
