"""End-to-end smoke test of the assimilation loop on the reference backend.

Keep the problem tiny so this runs in well under a minute.
"""
from __future__ import annotations

import numpy as np

from mfda.assimilation import run
from mfda.config import (
    DiagCfg,
    DomainCfg,
    ICParams,
    NudgeCfg,
    ObsCfg,
    PICCfg,
    RunCfg,
    TimeDerivCfg,
)


def _small_cfg(variant: str, *, time_derivative: TimeDerivCfg | None = None) -> RunCfg:
    obs = ObsCfg(kind="full")
    if time_derivative is not None:
        obs = ObsCfg(kind="full", time_derivative=time_derivative)
    return RunCfg(
        name="smoke",
        seed=0,
        domain=DomainCfg(k=0.5),
        truth_ic=ICParams(kind="perturbed_maxwellian", alpha=1e-3, sigma=1.0),
        assim_ic=ICParams(kind="ic_phase_error", alpha=1e-3, theta0=0.3),
        pic=PICCfg(Nx=64, Np=10_000, dt=1e-2, n_steps=50, backend="reference"),
        observation=obs,
        nudge=NudgeCfg(variant=variant, gamma=0.5),
        diagnostics=DiagCfg(every_diag_steps=10, phase_space_Nx=32, phase_space_Nv=32),
        outputs_dir="results",
    )


def test_smoke_velocity() -> None:
    out = run(_small_cfg("velocity"))
    assert len(out.t) > 0
    assert np.isfinite(out.assim_log["e_phi"]).all()


def test_smoke_position() -> None:
    out = run(_small_cfg("position"))
    assert len(out.t) > 0
    assert np.isfinite(out.assim_log["e_phi"]).all()


def test_smoke_none_baseline() -> None:
    out = run(_small_cfg("none"))
    assert len(out.t) > 0
    assert np.isfinite(out.assim_log["e_phi"]).all()


def test_smoke_time_derivative() -> None:
    """Velocity nudge with time-derivative observation enabled (alpha != 0).

    Just checks the run completes and produces finite e_phi/e_j. The
    acceptance gate on whether alpha actually reduces e_f is in
    scripts/sweep_alpha.py + the Phase A campaign, not here.
    """
    td = TimeDerivCfg(enabled=True, alpha=1.0,
                     lowpass_k_cut_frac=0.25, j_source="continuity")
    out = run(_small_cfg("velocity", time_derivative=td))
    assert len(out.t) > 0
    assert np.isfinite(out.assim_log["e_phi"]).all()
    extras = out.assim_log["extras"]
    assert "e_j" in extras
    assert np.isfinite(extras["e_j"]).all()
