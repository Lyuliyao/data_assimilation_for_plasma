"""Backward-compat regression for the combined-nudge schema refactor.

Confirms that:
  - YAML with the legacy `nudge.variant` + `nudge.gamma` keys is shimmed
    into the new ChannelCfg flags.
  - A 50-step end-to-end run with the shim-translated config produces
    finite, sane diagnostics (full bit-equivalence is not required because
    the assim loop's order of operations was restructured; see the kernel
    bit-exact test in test_nudging_dt_alpha_zero.py for the strict gate).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np

from mfda.assimilation import run
from mfda.config import _legacy_nudge_shim, load


def _legacy_yaml(variant: str, gamma: float = 1.0) -> str:
    return textwrap.dedent(f"""
        name: shim_test
        seed: 0
        domain: {{k: 0.5}}
        truth_ic: {{kind: perturbed_maxwellian, alpha: 1.0e-3, sigma: 1.0}}
        assim_ic: {{kind: ic_phase_error, alpha: 1.0e-3, theta0: 0.3}}
        pic: {{Nx: 64, Np: 5000, dt: 1.0e-2, n_steps: 50}}
        observation: {{kind: full}}
        nudge:
          variant: {variant}
          gamma: {gamma}
        diagnostics: {{every_diag_steps: 10, phase_space_Nx: 32, phase_space_Nv: 32}}
    """)


def test_shim_translates_velocity() -> None:
    import yaml as _yaml
    raw = _yaml.safe_load(_legacy_yaml("velocity", gamma=0.7))
    out = _legacy_nudge_shim(raw)
    assert "variant" not in out["nudge"]
    assert out["nudge"]["velocity_snapshot"] == {"enabled": True, "gamma": 0.7}
    assert "position_snapshot" not in out["nudge"]


def test_shim_translates_position() -> None:
    import yaml as _yaml
    raw = _yaml.safe_load(_legacy_yaml("position", gamma=0.5))
    out = _legacy_nudge_shim(raw)
    assert out["nudge"]["position_snapshot"] == {"enabled": True, "gamma": 0.5}


def test_shim_translates_none() -> None:
    import yaml as _yaml
    raw = _yaml.safe_load(_legacy_yaml("none"))
    out = _legacy_nudge_shim(raw)
    assert "variant" not in out["nudge"]
    # No channel should be enabled.
    assert "velocity_snapshot" not in out["nudge"]
    assert "position_snapshot" not in out["nudge"]


def test_shim_with_time_derivative_enables_velocity_dtobs(tmp_path: Path) -> None:
    """If time_derivative.enabled is set with the legacy variant: velocity,
    the shim should also enable velocity_dtobs. This preserves the Phase A/B
    behaviour for older configs that combined the legacy variant with
    augmented observation."""
    import yaml as _yaml
    raw = _yaml.safe_load(textwrap.dedent("""
        name: shim_test
        seed: 0
        domain: {k: 0.5}
        truth_ic: {kind: perturbed_maxwellian, alpha: 1.0e-3, sigma: 1.0}
        assim_ic: {kind: ic_phase_error, alpha: 1.0e-3, theta0: 0.3}
        pic: {Nx: 64, Np: 5000, dt: 1.0e-2, n_steps: 50}
        observation:
          kind: full
          time_derivative:
            enabled: true
            alpha: 2.5
        nudge:
          variant: velocity
          gamma: 1.5
        diagnostics: {every_diag_steps: 10}
    """))
    out = _legacy_nudge_shim(raw)
    assert out["nudge"]["velocity_snapshot"] == {"enabled": True, "gamma": 1.5}
    assert out["nudge"]["velocity_dtobs"] == {
        "enabled": True, "gamma": 1.5, "alpha": 2.5,
    }


def test_legacy_velocity_runs_end_to_end(tmp_path: Path) -> None:
    cfg_path = tmp_path / "shim.yaml"
    cfg_path.write_text(_legacy_yaml("velocity", gamma=0.5))
    cfg = load(cfg_path)
    assert cfg.nudge.velocity_snapshot.enabled is True
    assert cfg.nudge.velocity_snapshot.gamma == 0.5
    out = run(cfg)
    assert len(out.t) > 0
    assert np.isfinite(out.assim_log["e_phi"]).all()
