"""YAML config loading for 2D2V moment-observation runs."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class Domain2DCfg(BaseModel):
    kx: float = 1.0
    ky: float = 1.0
    Lx: float | None = None    # default 2 pi / kx
    Ly: float | None = None    # default 2 pi / ky


class ICParams2D(BaseModel):
    kind: Literal[
        "perturbed_maxwellian_2d", "ic_phase_error_2d",
        "ic_blob_2d", "wrong_maxwellian_2d",
    ]
    alpha: float = 0.01
    sigma: float = 1.0
    theta0: float = 0.0
    # blob
    x0: float = 0.0
    y0: float = 0.0
    epsilon: float = 0.3
    sigma_blob: float = 0.5
    # wrong_maxwellian
    rho_amp: float = 0.0
    ux_star: float = 0.0
    uy_star: float = 0.0
    T_star: float = 1.0


class PIC2DCfg(BaseModel):
    Nx: int = 64
    Ny: int = 64
    Np: int = 200_000
    dt: float = 0.05
    n_steps: int = 1000
    backend: Literal["reference"] = "reference"


class Collision2DCfg(BaseModel):
    kind: Literal["none", "bgk", "lb"] = "none"
    nu: float = 0.0
    rho_floor: float = 1.0e-3
    T_floor: float = 1.0e-3


class Driver2DCfg(BaseModel):
    """2D external F_ext.

    kind="oblique_wave":                 U = A cos(kx x + ky y - omega t + theta)
    kind="oblique_wavepacket":           U = A exp(-(t-t0)^2/(2 sigma_t^2)) cos(.)
    kind="checkerboard_standing":        U = A cos(kx_d x) cos(ky_d y) cos(omega t)
    kind="oblique_traveling_continuous": F = A g(t) sin(.) (kx, ky)/|k|, g=ramp-on
    """
    kind: Literal[
        "none", "oblique_wave", "oblique_wavepacket",
        "checkerboard_standing", "oblique_traveling_continuous",
    ] = "none"
    A: float = 0.0
    kx_d: float = 1.0
    ky_d: float = 1.0
    omega: float = 1.0
    theta: float = 0.0
    t0: float = 0.0
    sigma_t: float = 1.0
    ramp_time: float = 1.0


class MomentObs2DCfg(BaseModel):
    kind: Literal["full", "noisy"] = "full"
    sigma_rho: float = 0.0
    sigma_u: float = 0.0
    sigma_T: float = 0.0
    rho_floor: float = 1.0e-3
    T_floor: float = 1.0e-3
    every_q: int = 1
    rng_seed: int = 0


class FormulationA2DCfg(BaseModel):
    """Weighted-W2 A (eq. 16): same gammas as A_var plus the metric V_star."""
    gamma_1: float = 1.0
    gamma_2: float = 1.0
    gamma_3: float = 1.0
    V_star: float = 1.0
    lowpass_k_cut_frac: float = 0.25
    lowpass_sharpness: float = 16.0


class FormulationAVariant2DCfg(BaseModel):
    gamma_1: float = 1.0
    gamma_2: float = 1.0
    gamma_3: float = 1.0
    lowpass_k_cut_frac: float = 0.25
    lowpass_sharpness: float = 16.0
    rho_floor: float = 1.0e-3
    T_floor: float = 1.0e-3


class FormulationB2DCfg(BaseModel):
    gamma: float = 1.0
    gamma_1: float = 1.0
    gamma_2: float = 1.0
    gamma_3: float = 1.0
    lowpass_k_cut_frac: float = 0.25
    lowpass_sharpness: float = 16.0
    rho_floor: float = 1.0e-3
    T_floor: float = 1.0e-3


class FormulationC2DCfg(BaseModel):
    """Maxwellian-projected KL nudging in 2D (d=2 Bregman coefficients)."""
    lam: float = 1.0
    use_weighted_metric: bool = True
    V_star: float = 1.0
    lowpass_k_cut_frac: float = 0.25
    lowpass_sharpness: float = 16.0
    rho_floor: float = 1.0e-3
    T_floor: float = 1.0e-3


class FormulationAOT2DCfg(BaseModel):
    """Constant-gain AOT / Luenberger baseline in 2D (not a paper formulation)."""
    mu_rho: float = 1.0
    mu_u: float = 1.0
    mu_T: float = 1.0
    lowpass_k_cut_frac: float = 0.25
    lowpass_sharpness: float = 16.0
    rho_floor: float = 1.0e-3
    T_floor: float = 1.0e-3


class MomentNudge2DCfg(BaseModel):
    formulation: Literal["none", "A", "A_var", "B", "C", "aot"] = "A"
    A: FormulationA2DCfg = Field(default_factory=FormulationA2DCfg)
    A_var: FormulationAVariant2DCfg = Field(default_factory=FormulationAVariant2DCfg)
    B: FormulationB2DCfg = Field(default_factory=FormulationB2DCfg)
    C: FormulationC2DCfg = Field(default_factory=FormulationC2DCfg)
    aot: FormulationAOT2DCfg = Field(default_factory=FormulationAOT2DCfg)


class Diag2DCfg(BaseModel):
    snapshot_steps: list[int] = Field(default_factory=list)
    every_diag_steps: int = 10


class MomentRun2DCfg(BaseModel):
    name: str
    seed: int = 0
    domain: Domain2DCfg
    truth_ic: ICParams2D
    assim_ic: ICParams2D
    pic: PIC2DCfg
    collision: Collision2DCfg = Field(default_factory=Collision2DCfg)
    driver: Driver2DCfg = Field(default_factory=Driver2DCfg)
    moment_observation: MomentObs2DCfg = Field(default_factory=MomentObs2DCfg)
    moment_nudge: MomentNudge2DCfg = Field(default_factory=MomentNudge2DCfg)
    diagnostics: Diag2DCfg = Field(default_factory=Diag2DCfg)
    outputs_dir: str = "../results"

    @property
    def Lx(self) -> float:
        return self.domain.Lx if self.domain.Lx is not None else 2.0 * 3.141592653589793 / self.domain.kx

    @property
    def Ly(self) -> float:
        return self.domain.Ly if self.domain.Ly is not None else 2.0 * 3.141592653589793 / self.domain.ky


def load_moment_2d(path: str | Path) -> MomentRun2DCfg:
    raw = yaml.safe_load(Path(path).read_text())
    return MomentRun2DCfg(**raw)
