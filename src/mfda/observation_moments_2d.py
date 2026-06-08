"""2D moment observation: extract (rho, ux, uy, T) from a truth state."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backend_reference_2d import ReferenceState2D
from .diagnostics_2d import (
    RHO_FLOOR_DEFAULT,
    T_FLOOR_DEFAULT,
    grid_moments_2d2v,
)


@dataclass
class MomentObservationSpec2D:
    kind: str = "full"          # "full" or "noisy"
    sigma_rho: float = 0.0
    sigma_u: float = 0.0
    sigma_T: float = 0.0
    rho_floor: float = RHO_FLOOR_DEFAULT
    T_floor: float = T_FLOOR_DEFAULT
    every_q: int = 1
    rng_seed: int = 0


def observe_moments_2d(
    truth: ReferenceState2D,
    spec: MomentObservationSpec2D,
    rng: np.random.Generator | None = None,
):
    """Read (rho, ux, uy, T) from the truth state, optionally add Gaussian noise."""
    rho, ux, uy, T = grid_moments_2d2v(truth, spec.rho_floor, spec.T_floor)
    if spec.kind == "noisy":
        rng = rng or np.random.default_rng(spec.rng_seed)
        if spec.sigma_rho > 0:
            rho = rho + spec.sigma_rho * rng.standard_normal(rho.shape)
            rho = np.maximum(rho, spec.rho_floor)
        if spec.sigma_u > 0:
            ux = ux + spec.sigma_u * rng.standard_normal(ux.shape)
            uy = uy + spec.sigma_u * rng.standard_normal(uy.shape)
        if spec.sigma_T > 0:
            T = T + spec.sigma_T * rng.standard_normal(T.shape)
            T = np.maximum(T, spec.T_floor)
    return rho, ux, uy, T
