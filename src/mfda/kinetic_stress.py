"""1D kinetic stress (second velocity moment) deposit.

The continuity-based reconstruction of d^2 phi / dt^2 (eq. 14 of the
amendment doc) needs the kinetic stress tensor

    M(x, t) := integral v^2 f(x, v, t) dv

deposited on the spatial grid, similar to rho but weighted by v^2.

This is the same CIC shape function as backend_reference.cic_deposit but
with per-particle weight v_p^2 * w_p instead of w_p.
"""
from __future__ import annotations

import numpy as np


def cic_deposit_kinetic_stress(
    x: np.ndarray, v: np.ndarray, w: np.ndarray, L: float, Nx: int,
) -> np.ndarray:
    """Deposit M(x) = sum_p v_p^2 w_p S(x - x_p) on a periodic grid (CIC).

    Returns M on the same Nx-cell grid as the density. Mean of M equals
    integral v^2 f dx dv / L = sum_p v_p^2 w_p / L.
    """
    dx = L / Nx
    xi = x / dx
    i0 = np.floor(xi).astype(np.int64)
    frac = xi - i0
    i0 = np.mod(i0, Nx)
    i1 = np.mod(i0 + 1, Nx)
    v2w = v * v * w
    M = (np.bincount(i0, weights=v2w * (1.0 - frac), minlength=Nx)
         + np.bincount(i1, weights=v2w * frac, minlength=Nx))
    M /= dx
    return M
