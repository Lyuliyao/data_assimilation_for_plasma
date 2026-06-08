"""YAML config loading + validation for mfda runs."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class DomainCfg(BaseModel):
    k: float
    L: float | None = None   # if None, set to 2*pi/k
    v_min: float = -6.0
    v_max: float = 6.0


class ICParams(BaseModel):
    """Parameters for the initial-condition sampler."""
    kind: Literal[
        "perturbed_maxwellian", "drifted_maxwellian", "spatial_temperature",
        "two_peak_velocity", "asymmetric_two_peak",
        "two_stream", "bump_on_tail",
        "ic_phase_error", "ic_temperature_error", "ic_hidden_velocity",
        "wrong_maxwellian", "wrong_two_stream",
    ]
    alpha: float = 1e-3
    sigma: float = 1.0
    u0: float = 0.0
    theta0: float = 0.0
    sigma0: float = 1.0
    # asymmetric_two_peak only:
    u_pos: float = 0.0
    u_neg: float = 0.0
    w_pos: float = 0.5
    # spatial_temperature only:
    theta_bg: float = 1.0
    eta: float = 0.0
    # wrong_maxwellian only (note v3 §3 ABC recovery experiments):
    rho_amp: float = 0.0
    u_star: float = 0.0
    T_star: float = 1.0


class PICCfg(BaseModel):
    Nx: int = 128
    Np: int = 1_000_000
    dt: float = 1e-2
    n_steps: int = 2000
    backend: Literal["reference", "warpx"] = "reference"
    particle_shape: int = 1  # 1 = CIC


class TimeDerivCfg(BaseModel):
    """Augmented observation: J += (alpha/2) * |d phi_f / dt - z|^2.

    See docs/time_derivative_observation_plan.md.
    """
    enabled: bool = False
    alpha: float = 0.0
    lowpass_k_cut_frac: float = 0.25
    lowpass_sharpness: float = 16.0
    j_source: Literal["continuity", "finite_diff"] = "continuity"


class TimeSecondDerivCfg(BaseModel):
    """Augmented observation: J += (beta/2) * |d^2 phi_f / dt^2 - w|^2.

    The second time derivative needs a tighter lowpass than z because
    noise variance amplifies as 1/dt^4 (~ 1e8 at dt=1e-2). Default
    k_cut_frac is 0.15, lower than the dtobs default of 0.25. See
    docs/second_derivative_observation_plan.md pitfall §4.1.
    """
    enabled: bool = False
    lowpass_k_cut_frac: float = 0.15
    lowpass_sharpness: float = 16.0


class ObsCfg(BaseModel):
    kind: Literal["full", "noisy", "coarse"] = "full"
    sigma: float = 0.0
    every_m: int = 1
    reconstruction: Literal["fourier", "linear"] = "fourier"
    every_q: int = 1  # assimilate every q time steps
    rng_seed: int = 0
    time_derivative: TimeDerivCfg = Field(default_factory=TimeDerivCfg)
    time_second_derivative: TimeSecondDerivCfg = Field(
        default_factory=TimeSecondDerivCfg
    )


class ChannelCfg(BaseModel):
    """A single nudging channel — one of position_snapshot,
    velocity_snapshot, velocity_dtobs.

    `gamma` is the nudging strength. `alpha` is the time-derivative weight,
    only consumed by *_dtobs channels (no-op on snapshot channels).
    """
    enabled: bool = False
    gamma: float = 1.0
    alpha: float = 0.0


class NudgeCfg(BaseModel):
    """Channel-flag nudging config (replaces the legacy single-variant enum).

    Three channels can be combined freely; see docs/combined_nudge_amendment.md
    eqs. 17-18 for the underlying ODEs. The position-channel time-derivative
    correction (the bilinear term) is deferred per pitfall #5 of
    docs/time_derivative_observation_plan.md.
    """
    position_snapshot: ChannelCfg = Field(default_factory=ChannelCfg)
    velocity_snapshot: ChannelCfg = Field(default_factory=ChannelCfg)
    position_dtobs: ChannelCfg = Field(default_factory=ChannelCfg)
    velocity_dtobs: ChannelCfg = Field(default_factory=ChannelCfg)
    # Second-time-derivative channels — the alpha field of *_d2tobs is
    # interpreted as beta (the coefficient on the new (1/2)|d^2 phi - w|^2 term).
    # position_d2tobs is the bilinear-quadratic position update (U^2 H_3(psi_2)
    # + d_x(E H_1(psi_2)) + H_1(chi)); deferred per pitfall §4.5 of the doc —
    # enable raises NotImplementedError in the assim loop until D3.
    position_d2tobs: ChannelCfg = Field(default_factory=ChannelCfg)
    velocity_d2tobs: ChannelCfg = Field(default_factory=ChannelCfg)
    lowpass_k_cut_frac: float = 0.25
    lowpass_sharpness: float = 16.0

    def set_legacy_variant(self, variant: str, gamma: float = 1.0) -> None:
        """In-place setter that mirrors the old `variant` + `gamma` API.

        Used by sweep scripts that mutate the config per-iteration.
        """
        self.position_snapshot = ChannelCfg()
        self.velocity_snapshot = ChannelCfg()
        if variant == "velocity":
            self.velocity_snapshot = ChannelCfg(enabled=True, gamma=gamma)
        elif variant == "position":
            self.position_snapshot = ChannelCfg(enabled=True, gamma=gamma)
        elif variant != "none":
            raise ValueError(f"Unknown legacy variant: {variant!r}")


class DiagCfg(BaseModel):
    phase_space_Nx: int = 64
    phase_space_Nv: int = 64
    fourier_n_modes: int = 6
    every_diag_steps: int = 10


# ---------------------------------------------------------------------------
# Moment-observation / ABC config blocks (note v3 §3).
# These are optional and only consumed by assimilation_moments.run_moments.
# Existing potential-only runs are unaffected.
# ---------------------------------------------------------------------------


class CollisionCfg(BaseModel):
    """Collision-operator config.

    kind="bgk": Q_BGK(f) = nu (M[f] - f), MC replacement at the local Maxwellian.
    kind="lb":  Q_LB(f)  = nu d_v[(v - u) f + T d_v f], realised as an exact-OU
                Langevin substep. Same equilibrium as BGK; smoother per-step
                update so kinetic instabilities (two-stream etc.) survive better.
    nu : collision frequency in plasma units.
    """
    kind: Literal["none", "bgk", "lb"] = "none"
    nu: float = 0.0
    rho_floor: float = 1.0e-3
    T_floor: float = 1.0e-3
    # Imperfect-model DA: if set, the ASSIMILATING run uses a different
    # collision frequency / operator than the truth (model error). None ->
    # the assim shares the truth's kind/nu (perfect-model, the default).
    assim_nu: float | None = None
    assim_kind: Literal["none", "bgk", "lb"] | None = None


class DriverCfg(BaseModel):
    """Time-dependent external E-field config.

    For kind="off_resonance_wave":
        E_ext(x, t) = E0 * sin(k_d * x - omega_d * t)
    For kind="chirped_wave":
        E_ext(x, t) = E0 * sin(k_d * x - omega_0 * t - 0.5 * beta * t^2 + theta)
        Instantaneous frequency: omega(t) = omega_0 + beta * t.
    For kind="wavepacket":
        E_ext(x, t) = E0 * exp(-(t - t0)^2 / (2 sigma_t^2)) sin(k_d x - omega_d t + theta)
        Localised traveling wave packet centred at t0 with width sigma_t.
    """
    kind: Literal["none", "off_resonance_wave", "chirped_wave", "wavepacket"] = "none"
    E0: float = 0.0
    k_d: float = 1.0
    omega_d: float = 1.3
    omega_0: float = 0.75
    beta: float = 0.005
    theta: float = 0.0
    t0: float = 0.0
    sigma_t: float = 1.0


class MomentObsCfg(BaseModel):
    """Hydrodynamic-moment observation config (replaces potential ObsCfg
    when running the ABC path).
    """
    kind: Literal["full", "noisy"] = "full"
    sigma_rho: float = 0.0
    sigma_u: float = 0.0
    sigma_T: float = 0.0
    rho_floor: float = 1.0e-3
    T_floor: float = 1.0e-3
    every_q: int = 1
    rng_seed: int = 0


class FormulationACfg(BaseModel):
    gamma_1: float = 1.0
    gamma_2: float = 1.0
    gamma_3: float = 1.0
    V_star: float = 1.0
    lowpass_k_cut_frac: float = 0.25
    lowpass_sharpness: float = 16.0


class FormulationBCfg(BaseModel):
    """Direction-split SDE (arxiv v2, eqs. 17-18).

    `gamma_x`, `gamma_v` are the outer multipliers on the position- and
    velocity-channel drifts; `gamma_1, gamma_2, gamma_3` are the per-residual
    weights inside the split losses J_M^rho and J_M^{j,E}.
    """
    gamma_x: float = 1.0
    gamma_v: float = 1.0
    gamma_1: float = 1.0
    gamma_2: float = 1.0
    gamma_3: float = 1.0
    lowpass_k_cut_frac: float = 0.25
    lowpass_sharpness: float = 16.0


class FormulationCCfg(BaseModel):
    """Maxwellian-projected KL formulation (Bregman coefficients)."""
    lam: float = 1.0
    use_weighted_metric: bool = True
    V_star: float = 1.0
    lowpass_k_cut_frac: float = 0.25
    lowpass_sharpness: float = 16.0
    rho_floor: float = 1.0e-3
    T_floor: float = 1.0e-3


class FormulationAOTCfg(BaseModel):
    """Classical constant-gain AOT / Luenberger baseline (not a paper formulation).

    Drives primitive moments (rho_f, u_f, T_f) -> (rho_obs, u_obs, T_obs)
    with independent constant gains. Used as the external comparison
    baseline; it is the feedback Theorem 6.1 generalizes.
    """
    mu_rho: float = 1.0
    mu_u: float = 1.0
    mu_T: float = 1.0
    lowpass_k_cut_frac: float = 0.25
    lowpass_sharpness: float = 16.0
    rho_floor: float = 1.0e-3
    T_floor: float = 1.0e-3


class FormulationNaiveKLCfg(BaseModel):
    """Naive KL relaxation toward pi_obs (paper's excluded baseline, Table 1).

    Maxwellianizes f toward M(u_obs, T_obs); zero set is the single observed
    Maxwellian, not the moment manifold. Used to demonstrate the obstruction
    (it drives a non-equilibrium truth away). Stochastic (BGK-toward-pi_obs).
    """
    lam: float = 1.0
    rho_floor: float = 1.0e-3
    T_floor: float = 1.0e-3


class MomentNudgeCfg(BaseModel):
    """Choice of formulation A/B/C (+ AOT / naive-KL baselines) plus params.

    Only the block matching `formulation` is consumed; the others are
    ignored. `formulation: none` runs the assim particles with no nudging
    (useful baseline for the recover-check). `formulation: aot` runs the
    classical constant-gain AOT baseline; `formulation: naive_kl` runs the
    excluded KL-toward-pi_obs baseline (the obstruction demonstration).
    """
    formulation: Literal["none", "A", "B", "C", "aot", "naive_kl"] = "A"
    A: FormulationACfg = Field(default_factory=FormulationACfg)
    B: FormulationBCfg = Field(default_factory=FormulationBCfg)
    C: FormulationCCfg = Field(default_factory=FormulationCCfg)
    aot: FormulationAOTCfg = Field(default_factory=FormulationAOTCfg)
    naive_kl: FormulationNaiveKLCfg = Field(default_factory=FormulationNaiveKLCfg)


class MomentRunCfg(BaseModel):
    """Top-level config for the note v3 §3 ABC recovery campaign.

    Distinct from RunCfg (which is potential-only / channel-based)
    so the existing runs and tests remain bit-exact.
    """
    name: str
    seed: int = 0
    domain: DomainCfg
    truth_ic: ICParams
    assim_ic: ICParams
    pic: PICCfg
    collision: CollisionCfg = Field(default_factory=CollisionCfg)
    driver: DriverCfg = Field(default_factory=DriverCfg)
    moment_observation: MomentObsCfg = Field(default_factory=MomentObsCfg)
    moment_nudge: MomentNudgeCfg = Field(default_factory=MomentNudgeCfg)
    diagnostics: DiagCfg = Field(default_factory=DiagCfg)
    outputs_dir: str = "results"

    @field_validator("domain")
    @classmethod
    def _fill_L_moment(cls, v: DomainCfg) -> DomainCfg:
        if v.L is None:
            import math
            v.L = 2.0 * math.pi / v.k
        return v


class RunCfg(BaseModel):
    name: str
    seed: int = 0
    domain: DomainCfg
    truth_ic: ICParams
    assim_ic: ICParams
    pic: PICCfg
    observation: ObsCfg = Field(default_factory=ObsCfg)
    nudge: NudgeCfg = Field(default_factory=NudgeCfg)
    diagnostics: DiagCfg = Field(default_factory=DiagCfg)
    outputs_dir: str = "results"

    @field_validator("domain")
    @classmethod
    def _fill_L(cls, v: DomainCfg) -> DomainCfg:
        if v.L is None:
            import math
            v.L = 2.0 * math.pi / v.k
        return v


def _legacy_nudge_shim(data: dict) -> dict:
    """In-place: translate the legacy `nudge.variant` + `nudge.gamma` keys to
    the new ChannelCfg flags so existing configs (and existing scripts that
    set `cfg.nudge.variant = ...`) keep working without modification.

    Behavior matrix (legacy -> new):
      variant=velocity, gamma=g          -> velocity_snapshot{enabled, gamma=g}
      variant=position, gamma=g          -> position_snapshot{enabled, gamma=g}
      variant=none                       -> all channels disabled

    If observation.time_derivative.enabled is true and the legacy variant was
    "velocity", also auto-enable velocity_dtobs at the same gamma with the
    time_derivative.alpha. This preserves bit-exactness of all Phase A/B runs
    that combined `variant: velocity` with `time_derivative.enabled: true`.
    """
    raw = data.get("nudge")
    if not raw or "variant" not in raw:
        return data
    v = raw.pop("variant")
    g = raw.pop("gamma", 1.0)
    if v == "velocity":
        raw["velocity_snapshot"] = {"enabled": True, "gamma": g}
        td = data.get("observation", {}).get("time_derivative", {})
        if td.get("enabled"):
            raw["velocity_dtobs"] = {
                "enabled": True, "gamma": g, "alpha": td.get("alpha", 0.0),
            }
    elif v == "position":
        raw["position_snapshot"] = {"enabled": True, "gamma": g}
    # v == "none" -> all channels stay disabled (their default).
    data["nudge"] = raw
    return data


def load(path: str | Path) -> RunCfg:
    path = Path(path)
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    data = _legacy_nudge_shim(data)
    cfg = RunCfg.model_validate(data)
    od = Path(cfg.outputs_dir)
    if not od.is_absolute():
        cfg.outputs_dir = str((path.resolve().parent / od).resolve())
    return cfg


def load_moment(path: str | Path) -> MomentRunCfg:
    """Load a moment-observation (note v3 §3 ABC) config."""
    path = Path(path)
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    cfg = MomentRunCfg.model_validate(data)
    od = Path(cfg.outputs_dir)
    if not od.is_absolute():
        cfg.outputs_dir = str((path.resolve().parent / od).resolve())
    return cfg
