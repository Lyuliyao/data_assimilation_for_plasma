"""Assimilation loop for hydrodynamic-moment observations (note v3 §3).

Runs two synchronised simulations with BGK collisions and a time-dependent
external driver:

  - TRUTH       : no nudging, generates (rho_obs, u_obs, T_obs) snapshots
                  at the configured cadence.
  - ASSIMILATED : independent IC (typically a wrong-Maxwellian prior),
                  nudged with formulation A, B, or C between drift and
                  the second half-kick.

Distinct from `assimilation.run` (which is potential-only / channel-based).

Per-step structure
------------------
    half-kick   (truth & assim,  includes E_ext if set)
    drift       (both)
    field_solve (both)
    [observation step, if (n+1) % every_q == 0:]
        (rho_obs, u_obs, T_obs) <- observe_moments(truth)
        apply chosen formulation A/B/C to assim particles
        if formulation moved x_a, re-field-solve assim
    BGK substep (both, independent RNG draws)
    half-kick   (both, post-collision)
    diagnostics

The BGK substep is applied to truth and assim independently; the truth
is not "the right answer" for any given particle, only for the moments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from . import initial_conditions as ic_mod
from .backend_reference import (
    EExtFunc,
    ReferenceState,
    cic_deposit,
    field_solve,
    make_state,
    push_leapfrog_drift,
    push_leapfrog_half,
)
from .collisions import bgk_substep, grid_moments_1d1v, lb_substep
from .config import (
    DriverCfg,
    ICParams,
    MomentRunCfg,
)
from .diagnostics import (
    DiagnosticsLog,
    density_error,
    electric_energy,
    low_k_modes,
    phase_space_error,
    potential_error,
    temperature_error,
    velocity_moment_error,
)
from .nudging_moments import (
    FormulationAParams,
    FormulationBParams,
    FormulationCParams,
    apply_formulation_A,
    apply_formulation_B,
    apply_formulation_C,
    apply_aot,
    FormulationAOTParams,
    apply_naive_kl,
    FormulationNaiveKLParams,
)
from .observation_moments import MomentObservationSpec, observe_moments


# ---------------------------------------------------------------------------
# IC sampler dispatch (subset of assimilation.IC_SAMPLERS plus wrong_maxwellian).
# ---------------------------------------------------------------------------

IC_SAMPLERS: dict[str, Callable[..., Any]] = {
    "perturbed_maxwellian": ic_mod.perturbed_maxwellian,
    "two_peak_velocity": ic_mod.two_peak_velocity,
    "asymmetric_two_peak": ic_mod.asymmetric_two_peak,
    "two_stream": ic_mod.two_stream,
    "bump_on_tail": ic_mod.bump_on_tail,
    "ic_phase_error": ic_mod.ic_phase_error,
    "ic_temperature_error": ic_mod.ic_temperature_error,
    "ic_hidden_velocity": ic_mod.ic_hidden_velocity,
    "wrong_maxwellian": ic_mod.wrong_maxwellian,
    "wrong_two_stream": ic_mod.wrong_two_stream,
}


def _sample(ic: ICParams, Np: int, L: float, k: float, rng: np.random.Generator):
    sampler = IC_SAMPLERS[ic.kind]
    kw: dict[str, Any] = dict(Np=Np, L=L, k=k, alpha=ic.alpha, rng=rng)
    if ic.kind == "perturbed_maxwellian":
        kw["sigma"] = ic.sigma
    if ic.kind in {"two_peak_velocity", "ic_hidden_velocity"}:
        kw["u0"] = ic.u0
    if ic.kind == "asymmetric_two_peak":
        kw["u_pos"] = ic.u_pos
        kw["u_neg"] = ic.u_neg
        kw["w_pos"] = ic.w_pos
    if ic.kind == "ic_phase_error":
        kw["theta0"] = ic.theta0
    if ic.kind == "ic_temperature_error":
        kw["sigma0"] = ic.sigma0
    if ic.kind == "wrong_maxwellian":
        kw["rho_amp"] = ic.rho_amp
        kw["u_star"] = ic.u_star
        kw["T_star"] = ic.T_star
    if ic.kind == "wrong_two_stream":
        kw["u_star"] = ic.u_star
        kw["T_star"] = ic.T_star
    return sampler(**kw)


# ---------------------------------------------------------------------------
# Driver builder.
# ---------------------------------------------------------------------------


def _build_driver(cfg: DriverCfg) -> EExtFunc | None:
    if cfg.kind == "none" or cfg.E0 == 0.0:
        return None
    if cfg.kind == "off_resonance_wave":
        E0 = float(cfg.E0)
        kd = float(cfg.k_d)
        wd = float(cfg.omega_d)

        def E_ext(x: np.ndarray, t: float) -> np.ndarray:
            return E0 * np.sin(kd * x - wd * t)

        return E_ext
    if cfg.kind == "chirped_wave":
        # F_ext = E0 sin(k_d x - omega_0 t - 0.5 beta t^2 + theta).
        # Instantaneous frequency omega(t) = omega_0 + beta t.
        E0 = float(cfg.E0)
        kd = float(cfg.k_d)
        w0 = float(cfg.omega_0)
        beta = float(cfg.beta)
        theta = float(cfg.theta)

        def E_ext(x: np.ndarray, t: float) -> np.ndarray:
            phase = kd * x - w0 * t - 0.5 * beta * t * t + theta
            return E0 * np.sin(phase)

        return E_ext
    if cfg.kind == "wavepacket":
        # F_ext = E0 * exp(-(t-t0)^2 / (2 sigma_t^2)) sin(k_d x - omega_d t + theta).
        # Localised traveling wave packet centred at t0 with width sigma_t.
        E0 = float(cfg.E0)
        kd = float(cfg.k_d)
        wd = float(cfg.omega_d)
        t0 = float(cfg.t0)
        sigma_t = float(cfg.sigma_t)
        theta = float(cfg.theta)

        def E_ext(x: np.ndarray, t: float) -> np.ndarray:
            env = float(np.exp(-((t - t0) ** 2) / (2.0 * sigma_t * sigma_t)))
            return E0 * env * np.sin(kd * x - wd * t + theta)

        return E_ext
    raise ValueError(f"Unknown driver kind: {cfg.kind!r}")


# ---------------------------------------------------------------------------
# Output container.
# ---------------------------------------------------------------------------


@dataclass
class MomentAssimilationOutput:
    t: np.ndarray
    truth_log: dict[str, Any]
    assim_log: dict[str, Any]
    final_truth: dict[str, np.ndarray]
    final_assim: dict[str, np.ndarray]
    config_snapshot: dict[str, Any]
    snapshots_truth: dict[int, dict[str, np.ndarray]] | None = None
    snapshots_assim: dict[int, dict[str, np.ndarray]] | None = None


# ---------------------------------------------------------------------------
# Main loop.
# ---------------------------------------------------------------------------


def run_moments(
    cfg: MomentRunCfg,
    snapshot_steps: list[int] | None = None,
    nudge_until_step: int | None = None,
    truth_np: int | None = None,
) -> MomentAssimilationOutput:
    """End-to-end truth + assimilated moment-observation run.

    See module docstring for per-step structure. Both truth and assim
    receive the same E_ext driver (model is perfect) and run their own
    independent BGK substep (since BGK depends on the local state).

    nudge_until_step : if set, the chosen formulation is applied only for
        steps n < nudge_until_step. After that the assim particles
        evolve under Vlasov + driver + BGK alone (free run). Used by
        the "nudge then free" experiment to test whether the assimilated
        state is self-sustaining or relapses without continued nudging.

    truth_np : if set, the truth ensemble is sampled with this many particles
        instead of cfg.pic.Np, on the same grid. Used to test an
        independently and more finely sampled (higher-resolution) truth so
        the recovery is not an artifact of equal truth/assim particle counts.
    """
    rng = np.random.default_rng(cfg.seed)
    obs_rng = np.random.default_rng(cfg.moment_observation.rng_seed)
    bgk_rng_truth = np.random.default_rng(cfg.seed + 100)
    bgk_rng_assim = np.random.default_rng(cfg.seed + 200)

    L = cfg.domain.L
    k = cfg.domain.k
    Nx = cfg.pic.Nx
    Np = cfg.pic.Np
    dt = cfg.pic.dt
    n_steps = cfg.pic.n_steps

    # ---- Build driver (shared by truth + assim) ----
    E_ext_func = _build_driver(cfg.driver)

    # ---- Build initial states ----
    # When truth_np is set (higher-resolution-truth / density campaign) we sample the
    # truth and assim ensembles from INDEPENDENT RNG streams spawned from cfg.seed, so
    # that changing truth_np alters only the truth sampling noise and leaves the assim
    # ensemble fixed at a given seed (clean attribution for the truth-resolution and
    # assim-Np sweeps). When truth_np is None we keep the legacy shared-rng path so the
    # previously generated results remain bit-for-bit reproducible.
    Np_truth = truth_np if truth_np is not None else Np
    if truth_np is not None:
        _child_t, _child_a = np.random.SeedSequence(cfg.seed).spawn(2)
        rng_truth = np.random.default_rng(_child_t)
        rng_assim = np.random.default_rng(_child_a)
        x_t, v_t, w_t = _sample(cfg.truth_ic, Np_truth, L, k, rng_truth)
        x_a, v_a, w_a = _sample(cfg.assim_ic, Np, L, k, rng_assim)
    else:
        x_t, v_t, w_t = _sample(cfg.truth_ic, Np_truth, L, k, rng)
        x_a, v_a, w_a = _sample(cfg.assim_ic, Np, L, k, rng)
    truth = make_state(x_t, v_t, w_t, L, Nx, dt, E_ext_func=E_ext_func)
    assim = make_state(x_a, v_a, w_a, L, Nx, dt, E_ext_func=E_ext_func)

    # ---- Observation spec ----
    obs_spec = MomentObservationSpec(
        kind=cfg.moment_observation.kind,
        sigma_rho=cfg.moment_observation.sigma_rho,
        sigma_u=cfg.moment_observation.sigma_u,
        sigma_T=cfg.moment_observation.sigma_T,
        rho_floor=cfg.moment_observation.rho_floor,
        T_floor=cfg.moment_observation.T_floor,
        every_q=cfg.moment_observation.every_q,
        rng_seed=cfg.moment_observation.rng_seed,
    )

    # ---- Diagnostics accumulators ----
    truth_log = DiagnosticsLog()
    assim_log = DiagnosticsLog()
    diag_cfg = cfg.diagnostics

    # ---- Snapshot policy (mirrors assimilation.py) ----
    if snapshot_steps is None:
        n_default = 7
        snapshot_steps = sorted({
            int(round(i * n_steps / (n_default - 1)))
            for i in range(n_default)
        })
    snap_steps_set: set[int] = set(snapshot_steps)
    snaps_truth: dict[int, dict[str, np.ndarray]] = {}
    snaps_assim: dict[int, dict[str, np.ndarray]] = {}
    if 0 in snap_steps_set:
        snaps_truth[0] = _snapshot(truth)
        snaps_assim[0] = _snapshot(assim)

    # ---- Pre-build formulation params from cfg (immutable per-run) ----
    formulation = cfg.moment_nudge.formulation
    A_params = FormulationAParams(
        gamma_1=cfg.moment_nudge.A.gamma_1,
        gamma_2=cfg.moment_nudge.A.gamma_2,
        gamma_3=cfg.moment_nudge.A.gamma_3,
        V_star=cfg.moment_nudge.A.V_star,
        lowpass_k_cut_frac=cfg.moment_nudge.A.lowpass_k_cut_frac,
        lowpass_sharpness=cfg.moment_nudge.A.lowpass_sharpness,
    )
    B_params = FormulationBParams(
        gamma_x=cfg.moment_nudge.B.gamma_x,
        gamma_v=cfg.moment_nudge.B.gamma_v,
        gamma_1=cfg.moment_nudge.B.gamma_1,
        gamma_2=cfg.moment_nudge.B.gamma_2,
        gamma_3=cfg.moment_nudge.B.gamma_3,
        lowpass_k_cut_frac=cfg.moment_nudge.B.lowpass_k_cut_frac,
        lowpass_sharpness=cfg.moment_nudge.B.lowpass_sharpness,
    )
    C_params = FormulationCParams(
        lam=cfg.moment_nudge.C.lam,
        use_weighted_metric=cfg.moment_nudge.C.use_weighted_metric,
        V_star=cfg.moment_nudge.C.V_star,
        lowpass_k_cut_frac=cfg.moment_nudge.C.lowpass_k_cut_frac,
        lowpass_sharpness=cfg.moment_nudge.C.lowpass_sharpness,
        rho_floor=cfg.moment_nudge.C.rho_floor,
        T_floor=cfg.moment_nudge.C.T_floor,
    )
    AOT_params = FormulationAOTParams(
        mu_rho=cfg.moment_nudge.aot.mu_rho,
        mu_u=cfg.moment_nudge.aot.mu_u,
        mu_T=cfg.moment_nudge.aot.mu_T,
        lowpass_k_cut_frac=cfg.moment_nudge.aot.lowpass_k_cut_frac,
        lowpass_sharpness=cfg.moment_nudge.aot.lowpass_sharpness,
        rho_floor=cfg.moment_nudge.aot.rho_floor,
        T_floor=cfg.moment_nudge.aot.T_floor,
    )
    NAIVE_KL_params = FormulationNaiveKLParams(
        lam=cfg.moment_nudge.naive_kl.lam,
        rho_floor=cfg.moment_nudge.naive_kl.rho_floor,
        T_floor=cfg.moment_nudge.naive_kl.T_floor,
    )
    collision_kind = cfg.collision.kind
    collision_nu = cfg.collision.nu if collision_kind in ("bgk", "lb") else 0.0
    # Imperfect-model DA: the assim run may use a different collision
    # operator/frequency than the truth. Defaults to the truth's (perfect model).
    assim_collision_kind = cfg.collision.assim_kind or collision_kind
    assim_collision_nu = (
        cfg.collision.assim_nu if cfg.collision.assim_nu is not None
        else cfg.collision.nu)
    if assim_collision_kind not in ("bgk", "lb"):
        assim_collision_nu = 0.0

    for n in range(n_steps):
        # 1) Half-kick (uses E_ext if set).
        push_leapfrog_half(truth, 0.5)
        push_leapfrog_half(assim, 0.5)

        # 2) Drift.
        push_leapfrog_drift(truth)
        push_leapfrog_drift(assim)

        # 3) Forward field solve.
        field_solve(truth)
        field_solve(assim)

        # 4) Observe + nudge (potentially sparse-in-time).
        nudging_active = (
            formulation != "none"
            and (nudge_until_step is None or n < nudge_until_step)
        )
        do_obs = (n % obs_spec.every_q) == 0 and nudging_active
        if do_obs:
            rho_obs, u_obs, T_obs = observe_moments(truth, obs_spec, rng=obs_rng)
            x_was = assim.x.copy()
            if formulation == "A":
                assim.x, assim.v = apply_formulation_A(
                    assim.x, assim.v, assim.w, L, Nx,
                    rho_obs, u_obs, T_obs, A_params, dt,
                )
            elif formulation == "B":
                assim.x, assim.v = apply_formulation_B(
                    assim.x, assim.v, assim.w, L, Nx,
                    rho_obs, u_obs, T_obs, B_params, dt,
                )
            elif formulation == "C":
                assim.x, assim.v = apply_formulation_C(
                    assim.x, assim.v, assim.w, L, Nx,
                    rho_obs, u_obs, T_obs, C_params, dt,
                )
            elif formulation == "aot":
                assim.x, assim.v = apply_aot(
                    assim.x, assim.v, assim.w, L, Nx,
                    rho_obs, u_obs, T_obs, AOT_params, dt,
                )
            elif formulation == "naive_kl":
                assim.x, assim.v = apply_naive_kl(
                    assim.x, assim.v, assim.w, L, Nx,
                    rho_obs, u_obs, T_obs, NAIVE_KL_params, dt,
                    bgk_rng_assim,
                )
            # If x moved, re-solve so the next half-kick uses the corrected E.
            if not np.array_equal(x_was, assim.x):
                field_solve(assim)

        # 5) Collision substep (both runs, independent RNGs; assim may use a
        #    different operator/frequency for imperfect-model experiments).
        if collision_nu > 0.0:
            substep = bgk_substep if collision_kind == "bgk" else lb_substep
            substep(
                truth, collision_nu, bgk_rng_truth,
                rho_floor=cfg.collision.rho_floor,
                T_floor=cfg.collision.T_floor,
            )
        if assim_collision_nu > 0.0:
            substep_a = bgk_substep if assim_collision_kind == "bgk" else lb_substep
            substep_a(
                assim, assim_collision_nu, bgk_rng_assim,
                rho_floor=cfg.collision.rho_floor,
                T_floor=cfg.collision.T_floor,
            )

        # 6) Second half-kick.
        push_leapfrog_half(truth, 0.5)
        push_leapfrog_half(assim, 0.5)
        truth.t += dt
        assim.t += dt

        # 7) Diagnostics.
        if (n % diag_cfg.every_diag_steps) == 0 or n == n_steps - 1:
            t_now = (n + 1) * dt
            # phi error (still informative even though we observe moments).
            e_phi = potential_error(assim.phi, truth.phi, L)
            rho_t, u_t, T_t = grid_moments_1d1v(truth)
            rho_a, u_a, T_a = grid_moments_1d1v(assim)
            e_rho = density_error(rho_a, rho_t, L)
            e_u = velocity_moment_error(u_a, u_t, rho_t, L,
                                        rho_floor=cfg.collision.rho_floor)
            e_T = temperature_error(T_a, T_t, rho_t, L,
                                    rho_floor=cfg.collision.rho_floor)
            e_f = phase_space_error(
                assim.x, assim.v, assim.w,
                truth.x, truth.v, truth.w,
                L=L, v_min=cfg.domain.v_min, v_max=cfg.domain.v_max,
                Nx=diag_cfg.phase_space_Nx, Nv=diag_cfg.phase_space_Nv,
            )
            en_t = electric_energy(truth.E, L)
            en_a = electric_energy(assim.E, L)
            modes_t = low_k_modes(truth.E, diag_cfg.fourier_n_modes)
            modes_a = low_k_modes(assim.E, diag_cfg.fourier_n_modes)
            truth_log.push(t_now, energy=en_t, modes=modes_t)
            assim_log.push(
                t_now,
                e_phi=e_phi, e_rho=e_rho, e_f=e_f,
                e_u=e_u, e_T=e_T,
                energy=en_a, modes=modes_a,
            )

        # Snapshot save.
        step = n + 1
        if step in snap_steps_set:
            snaps_truth[step] = _snapshot(truth)
            snaps_assim[step] = _snapshot(assim)

    return MomentAssimilationOutput(
        t=np.array(assim_log.t),
        truth_log=truth_log.to_dict(),
        assim_log=assim_log.to_dict(),
        final_truth={"x": truth.x, "v": truth.v, "w": truth.w,
                     "phi": truth.phi, "E": truth.E},
        final_assim={"x": assim.x, "v": assim.v, "w": assim.w,
                     "phi": assim.phi, "E": assim.E},
        config_snapshot=cfg.model_dump(),
        snapshots_truth=(snaps_truth if snap_steps_set else None),
        snapshots_assim=(snaps_assim if snap_steps_set else None),
    )


def _snapshot(state: ReferenceState) -> dict[str, np.ndarray]:
    return {
        "x": state.x.copy(),
        "v": state.v.copy(),
        "w": state.w.copy(),
        "phi": (state.phi.copy() if state.phi is not None else np.zeros(0)),
        "t": state.t,
    }
