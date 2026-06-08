"""WarpX / PICMI backend.

This is a stub that Claude Code should flesh out once a WarpX install is
available (locally or on Bridges-2). The design intent is documented below.

Why WarpX
---------
- Actively maintained, well-supported on DOE HPC, GPU-ready via AMReX.
- 1D/2D/3D ES and EM modes already exist, so the same codebase scales up.
- PICMI Python front-end + pyAMReX for in-Python array access between steps.

Design
------
A `WarpXBackend` instance wraps a PICMI simulation configured for 1D ES.
Each call to `step()` advances one PIC time step. Between steps, the
assimilation loop can:

  1. Read the current density / potential via pyAMReX (ParticleContainer,
     MultiFab). Specifically we read rho on the Poisson grid to compute
     phi_f (or read phi_f directly if WarpX's ES solver has exposed it).
  2. Solve the adjoint Poisson problem for psi externally (in
     mfda.poisson.adjoint_potential) using numpy/FFT — this keeps the
     adjoint independent of WarpX's internal solver.
  3. Compute grad_psi on the same grid.
  4. Apply nudging.
        velocity variant:
          Register grad_psi as a user-defined time-dependent external
          E-field on the next PIC step. PICMI supports this via
          `picmi.AnalyticAppliedField` or equivalent callback. The field
          is updated between steps from Python.
        position variant:
          Use pyAMReX to access particle position arrays and apply
          `x_p -= gamma * grad_psi(x_p) * dt`, wrapping modulo L for the
          periodic domain, BEFORE the next step.

Notes for the implementer
-------------------------
- Installing WarpX on Bridges-2:
    * Try `module avail warpx` first.
    * If unavailable, build with Spack targeting the GPU toolchain
      (see WarpX docs: https://warpx.readthedocs.io/en/latest/install).
    * pyAMReX wheels from conda-forge may work on the login node; GPU
      runs require a matching build.
- Reproducibility:
    * Match Nx, domain, dt, Np to the YAML config.
    * Use a fixed particle initialisation RNG (PICMI supports seeding
      per-species loaders; if not, sample x, v, w in Python and pass
      them to PICMI as a particle list).
- Checkpointing:
    * Use WarpX plotfiles for truth trajectories. Write the potential
      history separately as hdf5 keyed by time step so the assimilation
      loop can consume it without parsing plotfiles.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WarpXConfig:
    L: float
    Nx: int
    Np: int
    v_min: float
    v_max: float
    dt: float
    n_steps: int
    particle_shape: int = 1  # CIC
    use_gpu: bool = False


class WarpXBackend:
    """Stub. Replace with a real PICMI-driven simulation object."""

    def __init__(self, cfg: WarpXConfig):
        self.cfg = cfg
        self._sim = None
        self.t = 0.0
        # Last known grid fields; filled in by step() once implemented.
        self.rho: np.ndarray | None = None
        self.phi: np.ndarray | None = None
        self.E: np.ndarray | None = None
        raise NotImplementedError(
            "backend_warpx.WarpXBackend is a stub. Fill in the PICMI + pyAMReX "
            "calls once a WarpX install is available. See the module docstring "
            "for the intended design."
        )

    # Required interface — mirrors backend_reference.ReferenceState / step.
    def load_particles(self, x: np.ndarray, v: np.ndarray, w: np.ndarray) -> None:
        raise NotImplementedError

    def field_solve(self) -> None:
        raise NotImplementedError

    def step(self) -> None:
        raise NotImplementedError

    # Accessors the assim loop uses.
    def get_density(self) -> np.ndarray:
        raise NotImplementedError

    def get_potential(self) -> np.ndarray:
        raise NotImplementedError

    def get_particles(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def set_external_E(self, E_ext_on_grid: np.ndarray) -> None:
        """For velocity-nudging: inject -gamma * grad_psi as an external field."""
        raise NotImplementedError

    def shift_particle_positions(self, dx: np.ndarray) -> None:
        """For position-nudging: apply x_p += dx(x_p) via pyAMReX."""
        raise NotImplementedError
