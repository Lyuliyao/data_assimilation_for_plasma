"""mfda: mean-field data assimilation for Vlasov-Poisson.

Organizing module. See README.md and CLAUDE.md for the map.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .assimilation_moments import MomentAssimilationOutput, run_moments
from .collisions import bgk_substep, grid_moments_1d1v, lb_substep
from .config import (
    CollisionCfg,
    DriverCfg,
    FormulationACfg,
    FormulationAOTCfg,
    FormulationBCfg,
    FormulationCCfg,
    FormulationNaiveKLCfg,
    MomentNudgeCfg,
    MomentObsCfg,
    MomentRunCfg,
    load_moment,
)
from .nudging_moments import (
    FormulationAParams,
    FormulationAOTParams,
    FormulationBParams,
    FormulationCParams,
    FormulationNaiveKLParams,
    _bregman_coefficients,
    apply_aot,
    apply_formulation_A,
    apply_formulation_B,
    apply_formulation_C,
    apply_naive_kl,
)
from .observation_moments import MomentObservationSpec, observe_moments

__all__ = [
    "MomentAssimilationOutput",
    "run_moments",
    "bgk_substep",
    "lb_substep",
    "grid_moments_1d1v",
    "CollisionCfg",
    "DriverCfg",
    "FormulationACfg",
    "FormulationAOTCfg",
    "FormulationBCfg",
    "FormulationCCfg",
    "MomentNudgeCfg",
    "MomentObsCfg",
    "MomentRunCfg",
    "load_moment",
    "FormulationAParams",
    "FormulationAOTParams",
    "FormulationBParams",
    "FormulationCParams",
    "FormulationNaiveKLParams",
    "FormulationNaiveKLCfg",
    "apply_aot",
    "apply_formulation_A",
    "apply_formulation_B",
    "apply_formulation_C",
    "apply_naive_kl",
    "MomentObservationSpec",
    "observe_moments",
]
