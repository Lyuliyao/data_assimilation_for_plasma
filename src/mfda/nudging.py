"""Two nudging variants: velocity (force-type) and position (phase-space grad).

Both share the same adjoint potential psi (see poisson.adjoint_potential). They
differ only in where the correction enters the particle equations.

These are pure, backend-agnostic kernels. A backend computes grad_psi on its
Poisson grid, interpolates it to particle positions, and calls the appropriate
apply_* function.

Equations from the note:

  Velocity (force-type, eq. 10-11):
      dY/dt = U
      dU/dt = E(Y,t) - gamma * grad_psi(Y,t)

  Position (phase-space gradient, eq. 8-9):
      dY/dt = U - gamma * grad_psi(Y,t)
      dU/dt = E(Y,t)

Here we implement the *nudging increment only*; the backend's own particle
pusher handles the (E, U) terms. So each function returns the increment that
must be added on top of the unnudged step.

Conventions
-----------
- x, v are (Np,) arrays of particle positions and velocities.
- grad_psi_at_particles is (Np,) values of d psi / dx evaluated at the particle
  positions (interpolated from the Poisson grid by the backend).
- gamma is the nudging strength (> 0).
- dt is the time step; the increments are Euler-like over one dt.
"""
from __future__ import annotations

import numpy as np


def apply_velocity_nudging(
    x: np.ndarray,
    v: np.ndarray,
    grad_psi_at_particles: np.ndarray,
    gamma: float,
    dt: float,
    grad_psi1_at_particles: np.ndarray | None = None,
    alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Force-type nudging: add -gamma * grad_psi0 (- gamma * alpha * grad_psi1) to velocity.

    The optional second term implements the time-derivative-observation
    correction (eq. 16-18 of v12 of the note, force-type variant). Per
    pitfall #5 of docs/time_derivative_observation_plan.md, the velocity-
    channel update is `dU = -gamma*alpha*grad_psi1 dt`, NOT
    `-gamma*alpha*v*grad_psi1 dt`. The `v` factor lives in the variational
    derivative delta J / delta f, not in the equations of motion.

    When alpha == 0 (default), the function returns the same value as the
    snapshot-only call - this is enforced by tests.
    """
    v_new = v - gamma * grad_psi_at_particles * dt
    if alpha != 0.0 and grad_psi1_at_particles is not None:
        v_new = v_new - gamma * alpha * grad_psi1_at_particles * dt
    return x, v_new


def apply_position_nudging(
    x: np.ndarray,
    v: np.ndarray,
    grad_psi_at_particles: np.ndarray,
    gamma: float,
    dt: float,
    L: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase-space-gradient nudging: add -gamma * grad_psi to position.

    If `L` is given, positions are wrapped modulo L to maintain the periodic
    domain invariant.
    """
    x_new = x - gamma * grad_psi_at_particles * dt
    if L is not None:
        x_new = np.mod(x_new, L)
    return x_new, v


NUDGING_VARIANTS = {
    "velocity": apply_velocity_nudging,
    "position": apply_position_nudging,
    "none": lambda x, v, *_args, **_kw: (x, v),
}


def apply_nudging(
    variant: str,
    x: np.ndarray,
    v: np.ndarray,
    grad_psi_at_particles: np.ndarray,
    gamma: float,
    dt: float,
    L: float | None = None,
    grad_psi1_at_particles: np.ndarray | None = None,
    alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch on the string name of the variant.

    variant in {"velocity", "position", "none"}.

    The time-derivative term (alpha, grad_psi1_at_particles) is only
    consumed by the "velocity" variant in Phase A. The "position" variant
    ignores those kwargs (Phase B will add the bilinear position-channel
    handling).
    """
    if variant == "position":
        return apply_position_nudging(x, v, grad_psi_at_particles, gamma, dt, L=L)
    if variant == "velocity":
        return apply_velocity_nudging(
            x, v, grad_psi_at_particles, gamma, dt,
            grad_psi1_at_particles=grad_psi1_at_particles,
            alpha=alpha,
        )
    if variant == "none":
        return x, v
    raise ValueError(f"Unknown nudging variant: {variant!r}. "
                     f"Valid: {list(NUDGING_VARIANTS)}")
