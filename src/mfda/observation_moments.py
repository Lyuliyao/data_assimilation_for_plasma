"""Hydrodynamic-moment observation operators (note v3 §3).

Whereas `observation.py` returns potential-only data y(x) = phi_true(x),
this module returns the full hydrodynamic moment triple

    rho_obs(x), u_obs(x), T_obs(x)

on the Poisson grid. These are the inputs consumed by formulations A, B,
and C in `nudging_moments.py`.

In notation of the note (eqs. 14): with B_0 f = rho_f, B_1 f = j_f =
rho_f * u_f, and B_2 f = E_f = (rho_f * |u_f|^2)/2 + (d/2) rho_f * T_f,
the moment triple uniquely determines the local Maxwellian. We return
the *primitive* moments (rho, u, T) rather than (rho, j, E) because
(a) A/B/C all use them in primitive form, (b) the Maxwellian-equivalence
class is parametrised in primitive form.

Conventions
-----------
- 1D1V (the reference backend setting), so j and u are scalars.
- T is computed from the second velocity moment with the bulk-velocity
  contribution removed: T = max(<v^2> - u^2, T_floor).
- Both rho and T are floored to keep downstream computation well-behaved
  in low-density cells. This is consistent with the BGK substep's
  conventions in collisions.py.

Optional noise
--------------
For the "noisy" observation kind, independent Gaussian noise is added to
each of rho_obs, u_obs, T_obs, with separate sigmas. Defaults are zero
(noiseless).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backend_reference import (
    ReferenceState,
    cic_deposit,
    cic_deposit_current,
)
from .collisions import RHO_FLOOR_DEFAULT, T_FLOOR_DEFAULT, grid_moments_1d1v


@dataclass
class MomentObservationSpec:
    kind: str = "full"  # "full" | "noisy"
    sigma_rho: float = 0.0
    sigma_u: float = 0.0
    sigma_T: float = 0.0
    rho_floor: float = RHO_FLOOR_DEFAULT
    T_floor: float = T_FLOOR_DEFAULT
    every_q: int = 1   # assimilate moment obs every q time steps
    rng_seed: int = 0


def observe_moments(
    state: ReferenceState,
    spec: MomentObservationSpec,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rho_obs, u_obs, T_obs) on the Poisson grid from a particle state.

    For "full": returns the empirical grid moments unchanged.
    For "noisy": adds independent Gaussian noise with the spec's sigmas.

    The empirical (rho, u, T) computation is shared with the BGK substep
    via collisions.grid_moments_1d1v.
    """
    rng = rng or np.random.default_rng(spec.rng_seed)
    rho_g, u_g, T_g = grid_moments_1d1v(state, spec.rho_floor, spec.T_floor)
    if spec.kind == "full":
        return rho_g.copy(), u_g.copy(), T_g.copy()
    if spec.kind == "noisy":
        rho_o = rho_g + spec.sigma_rho * rng.standard_normal(rho_g.shape)
        u_o = u_g + spec.sigma_u * rng.standard_normal(u_g.shape)
        T_o = T_g + spec.sigma_T * rng.standard_normal(T_g.shape)
        # Re-floor noisy moments.
        rho_o = np.maximum(rho_o, spec.rho_floor)
        T_o = np.maximum(T_o, spec.T_floor)
        return rho_o, u_o, T_o
    raise ValueError(f"Unknown moment-observation kind: {spec.kind!r}")


def derived_moment_obs(
    rho_obs: np.ndarray, u_obs: np.ndarray, T_obs: np.ndarray, d: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert primitive (rho, u, T) observations to (j_obs, E_obs).

    j_obs = rho_obs * u_obs
    E_obs = (1/2) rho_obs * u_obs^2 + (d/2) rho_obs * T_obs

    Used by formulations A and B (which build residuals against the
    conserved moments j and E rather than the primitives directly).
    """
    j_obs = rho_obs * u_obs
    E_obs = 0.5 * rho_obs * u_obs * u_obs + 0.5 * d * rho_obs * T_obs
    return j_obs, E_obs
