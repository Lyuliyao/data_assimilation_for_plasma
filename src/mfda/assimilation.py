"""Assimilation loop.

Runs two synchronized simulations:

  - TRUTH       : no nudging, generates y_n at observation times.
  - ASSIMILATED : with nudging (velocity- or position-type), driven by the
                  residual phi_f - y.

Both simulations use the same backend (currently: reference). WarpX support
can be plugged in by replacing the calls in `advance_one_step`.

The loop is written for the reference backend's half-kick / drift / half-kick
leapfrog. Nudging is applied between the drift and the second half-kick, i.e.
after the forward Poisson solve that yields phi_f.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from . import initial_conditions as ic_mod
from . import observation as obs_mod
from .backend_reference import (
    ReferenceState,
    cic_deposit,
    cic_deposit_current,
    cic_interpolate,
    field_solve,
    make_state,
    push_leapfrog_drift,
    push_leapfrog_half,
)
from .config import ICParams, RunCfg
from .diagnostics import (
    DiagnosticsLog,
    current_error,
    density_error,
    electric_energy,
    kinetic_stress_error,
    low_k_modes,
    phase_space_error,
    potential_error,
    vmarginal_variance_error,
)
from .filtering import lowpass_filter
from .kinetic_stress import cic_deposit_kinetic_stress
from .observation_time import (
    time_derivative_observation,
    time_second_derivative_observation,
)
from .poisson import (
    grad_1d,
    solve_chi,
    solve_poisson_1d,
    solve_poisson_from_d2,
    solve_poisson_from_div,
)


IC_SAMPLERS: dict[str, Callable[..., Any]] = {
    "perturbed_maxwellian": ic_mod.perturbed_maxwellian,
    "drifted_maxwellian": ic_mod.drifted_maxwellian,
    "spatial_temperature": ic_mod.spatial_temperature,
    "two_peak_velocity": ic_mod.two_peak_velocity,
    "asymmetric_two_peak": ic_mod.asymmetric_two_peak,
    "two_stream": ic_mod.two_stream,
    "bump_on_tail": ic_mod.bump_on_tail,
    "ic_phase_error": ic_mod.ic_phase_error,
    "ic_temperature_error": ic_mod.ic_temperature_error,
    "ic_hidden_velocity": ic_mod.ic_hidden_velocity,
}


def _sample(ic: ICParams, Np: int, L: float, k: float, rng: np.random.Generator):
    sampler = IC_SAMPLERS[ic.kind]
    kw: dict[str, Any] = dict(Np=Np, L=L, k=k, alpha=ic.alpha, rng=rng)
    if ic.kind == "perturbed_maxwellian":
        kw["sigma"] = ic.sigma
    if ic.kind == "drifted_maxwellian":
        kw["sigma"] = ic.sigma
        kw["u0"] = ic.u0
    if ic.kind == "spatial_temperature":
        kw["theta_bg"] = ic.theta_bg
        kw["eta"] = ic.eta
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
    return sampler(**kw)


@dataclass
class AssimilationOutput:
    t: np.ndarray
    truth_log: dict[str, Any]
    assim_log: dict[str, Any]
    final_truth: dict[str, np.ndarray]
    final_assim: dict[str, np.ndarray]
    config_snapshot: dict[str, Any]
    # If snapshot_steps was passed to run(), these dicts are keyed by step
    # index and each value is {"x", "v", "w", "phi", "t"}. Otherwise empty.
    snapshots_truth: dict[int, dict[str, np.ndarray]] | None = None
    snapshots_assim: dict[int, dict[str, np.ndarray]] | None = None


def run(
    cfg: RunCfg,
    snapshot_steps: list[int] | None = None,
) -> AssimilationOutput:
    """End-to-end truth + assimilated run on the reference backend.

    Snapshot policy
    ---------------
    By default (snapshot_steps=None), 7 evenly-spaced snapshots of
    (x, v, w, phi, t) are saved into output.snapshots_{truth,assim}
    at steps {0, n/6, n/3, n/2, 2n/3, 5n/6, n} where n = cfg.pic.n_steps.
    This makes the visualisation pipeline always available without the
    caller needing to think about it. The cost is 7 copies of the
    particle arrays (~7 * 3 * Np * 8 bytes ~ 170 MB at Np=1e6),
    transient.

    To explicitly disable snapshot saving (e.g. for sweep cells that
    only care about scalar metrics), pass snapshot_steps=[].

    To request specific steps, pass a list — e.g.
        snapshot_steps=[0, 100, 200, 300, ...]
    """
    rng = np.random.default_rng(cfg.seed)
    obs_rng = np.random.default_rng(cfg.observation.rng_seed)

    L = cfg.domain.L
    k = cfg.domain.k
    Nx = cfg.pic.Nx
    Np = cfg.pic.Np
    dt = cfg.pic.dt
    n_steps = cfg.pic.n_steps

    # ---- Build initial states ----
    x_t, v_t, w_t = _sample(cfg.truth_ic, Np, L, k, rng)
    x_a, v_a, w_a = _sample(cfg.assim_ic, Np, L, k, rng)
    truth = make_state(x_t, v_t, w_t, L, Nx, dt)
    assim = make_state(x_a, v_a, w_a, L, Nx, dt)

    # ---- Diagnostics accumulators ----
    truth_log = DiagnosticsLog()
    assim_log = DiagnosticsLog()

    # Observation spec
    obs_spec = obs_mod.ObservationSpec(
        kind=cfg.observation.kind,
        sigma=cfg.observation.sigma,
        every_m=cfg.observation.every_m,
        reconstruction=cfg.observation.reconstruction,
        every_q=cfg.observation.every_q,
        rng_seed=cfg.observation.rng_seed,
    )

    nudge = cfg.nudge
    diag_cfg = cfg.diagnostics
    tdcfg = cfg.observation.time_derivative
    t2cfg = cfg.observation.time_second_derivative

    # Time-derivative ring buffers. y_prev for d/dt (z), y_prev2 for d^2/dt^2 (w).
    # Each is filled lazily; nudging skips the corresponding term until enough
    # history has accumulated.
    y_prev: np.ndarray | None = None
    y_prev2: np.ndarray | None = None
    t_prev: float = 0.0
    t_prev2: float = 0.0

    # Intermediate-state snapshots for visualisation. Step index 0 snapshots
    # the post-IC state (before the first push). Default: 7 evenly-spaced
    # steps including 0 and n_steps. Pass [] to disable, or a custom list to
    # override.
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
        snaps_truth[0] = {"x": truth.x.copy(), "v": truth.v.copy(),
                          "w": truth.w.copy(),
                          "phi": (truth.phi.copy() if truth.phi is not None else np.zeros(0)),
                          "t": truth.t}
        snaps_assim[0] = {"x": assim.x.copy(), "v": assim.v.copy(),
                          "w": assim.w.copy(),
                          "phi": (assim.phi.copy() if assim.phi is not None else np.zeros(0)),
                          "t": assim.t}

    for n in range(n_steps):
        # 1) Half-kick both simulations.
        push_leapfrog_half(truth, 0.5)
        push_leapfrog_half(assim, 0.5)

        # 2) Drift.
        push_leapfrog_drift(truth)
        push_leapfrog_drift(assim)

        # 3) Forward field solve on both.
        field_solve(truth)
        field_solve(assim)

        # 4) Observe (potentially sparse-in-time) and nudge — channel-based
        #    dispatch. Four channels can be enabled independently:
        #      position_snapshot : -gamma * d psi0 / dx  in dY
        #      velocity_snapshot : -gamma * d psi0 / dx  in dU (force-type psi0)
        #      position_dtobs    : -gamma * alpha * U * d^2 psi1 / dx^2  in dY
        #                          (1D form of the bilinear  -gamma * alpha *
        #                          grad_x (U . grad_x psi1)  from eq. 17)
        #      velocity_dtobs    : -gamma * alpha * d psi1 / dx  in dU
        #    See docs/combined_nudge_amendment.md (eqs. 17-18).
        ps_ch = nudge.position_snapshot
        vs_ch = nudge.velocity_snapshot
        pd_ch = nudge.position_dtobs
        vd_ch = nudge.velocity_dtobs
        p2d_ch = nudge.position_d2tobs
        v2d_ch = nudge.velocity_d2tobs
        need_psi0 = ps_ch.enabled or vs_ch.enabled
        need_psi1_channels = pd_ch.enabled or vd_ch.enabled
        need_psi2_channels = p2d_ch.enabled or v2d_ch.enabled
        any_observation = need_psi0 or need_psi1_channels or need_psi2_channels

        if (n % obs_spec.every_q) == 0 and any_observation:
            t_now_obs = (n + 1) * dt
            y = obs_mod.observe(truth.phi, obs_spec, rng=obs_rng)

            # psi0: snapshot adjoint (only if a snapshot channel is on).
            grad_psi0_grid: np.ndarray | None = None
            if need_psi0:
                resid = assim.phi - y
                if nudge.lowpass_k_cut_frac < 1.0:
                    resid = lowpass_filter(
                        resid, L,
                        k_cut_frac=nudge.lowpass_k_cut_frac,
                        sharpness=nudge.lowpass_sharpness,
                    )
                psi0 = solve_poisson_1d(resid - resid.mean(), L)
                grad_psi0_grid = grad_1d(psi0, L)

            # psi1: time-derivative adjoint (only if a dtobs channel is on,
            # and we have a previous y to difference, and the augmented
            # observation block is enabled for the lowpass / j_source settings).
            grad_psi1_grid: np.ndarray | None = None
            lap_psi1_grid: np.ndarray | None = None
            need_psi1 = (
                need_psi1_channels and tdcfg.enabled and y_prev is not None
            )
            if need_psi1:
                dt_obs = t_now_obs - t_prev
                z = time_derivative_observation(
                    y, y_prev, dt_obs, L,
                    lowpass_k_cut_frac=tdcfg.lowpass_k_cut_frac,
                    lowpass_sharpness=tdcfg.lowpass_sharpness,
                )
                # Continuity: -Delta (d phi / dt) = -d j / dx, so
                # solve_poisson_from_div(j) returns d phi / dt directly.
                j_a = cic_deposit_current(assim.x, assim.v, assim.w, L, Nx)
                dphi_dt = solve_poisson_from_div(j_a, L)
                psi1_resid = dphi_dt - z
                psi1_resid = psi1_resid - psi1_resid.mean()
                psi1 = solve_poisson_1d(psi1_resid, L)
                grad_psi1_grid = grad_1d(psi1, L)
                # h1 = d^2 psi1 / dx^2 only needed for the bilinear position
                # channel; cheap (one extra rFFT round-trip).
                if pd_ch.enabled:
                    lap_psi1_grid = grad_1d(grad_psi1_grid, L)

            # psi2: second-time-derivative adjoint. Needed for both
            # velocity_d2tobs (h2 only) and position_d2tobs (grad_psi2, h3,
            # plus chi from solve_chi). Requires the two-step back buffer
            # (y_prev2). See docs/second_derivative_observation_plan.md.
            lap_psi2_grid: np.ndarray | None = None     # h2 = d^2 psi_2 / dx^2
            grad_psi2_grid: np.ndarray | None = None    # h1 = d psi_2 / dx
            grad3_psi2_grid: np.ndarray | None = None   # h3 = d^3 psi_2 / dx^3
            grad_chi_grid: np.ndarray | None = None     # h1 = d chi / dx
            grad_E_h1psi2_grid: np.ndarray | None = None  # d/dx (E * h1(psi_2))
            need_psi2 = (
                need_psi2_channels
                and t2cfg.enabled
                and y_prev is not None
                and y_prev2 is not None
            )
            if need_psi2:
                # Same observation cadence as the dtobs term — we use
                # (t_now_obs - t_prev) as Delta t. Assumes regular cadence
                # (every_q fixed); the same dt^2 normalisation is applied.
                dt_obs2 = t_now_obs - t_prev
                w_obs = time_second_derivative_observation(
                    y, y_prev, y_prev2, dt_obs2, L,
                    lowpass_k_cut_frac=t2cfg.lowpass_k_cut_frac,
                    lowpass_sharpness=t2cfg.lowpass_sharpness,
                )
                # Continuity: -Delta (d^2 phi / dt^2) = d^2 M / dx^2
                #                                       - d/dx (rho * E)
                M_a = cic_deposit_kinetic_stress(
                    assim.x, assim.v, assim.w, L, Nx,
                )
                rho_a = cic_deposit(assim.x, assim.w, L, Nx)
                d2phi_dt = solve_poisson_from_d2(M_a, rho_a, assim.E, L)
                psi2_resid = d2phi_dt - w_obs
                psi2_resid = psi2_resid - psi2_resid.mean()
                psi2 = solve_poisson_1d(psi2_resid, L)
                grad_psi2_grid = grad_1d(psi2, L)
                lap_psi2_grid = grad_1d(grad_psi2_grid, L)
                if p2d_ch.enabled:
                    # h3 = d^3 psi_2 / dx^3 is the highest spatial
                    # derivative in the augmented system; high-k noise
                    # is amplified by an extra factor of k. Per pitfall
                    # §4.5, may need its own narrower lowpass — defer
                    # tuning until D3 results land.
                    grad3_psi2_grid = grad_1d(lap_psi2_grid, L)
                    # chi solves -Delta chi = grad . (rho * grad psi_2).
                    chi = solve_chi(rho_a, grad_psi2_grid, L)
                    grad_chi_grid = grad_1d(chi, L)
                    # d/dx (E * h1(psi_2)) — the middle of the three
                    # position-d2tobs sub-terms.
                    grad_E_h1psi2_grid = grad_1d(assim.E * grad_psi2_grid, L)

            # Position channel — combine snapshot + bilinear dtobs +
            # bilinear-quadratic d2tobs into a single dx update. All
            # interpolations at the PRE-update assim.x to match eq. 17's
            # Y(t). The position_d2tobs piece adds three sub-terms per
            # docs/second_derivative_observation_plan.md §1.3:
            #   -gamma*beta * U^2 * H_3(psi_2)              (sub-term A)
            #   -gamma*beta * d/dx (E * H_1(psi_2)) at x_p   (sub-term B)
            #   -gamma*beta * H_1(chi) at x_p                (sub-term C)
            any_position = (
                ps_ch.enabled
                or (need_psi1 and pd_ch.enabled)
                or (need_psi2 and p2d_ch.enabled)
            )
            if any_position:
                dx_total = np.zeros_like(assim.x)
                if ps_ch.enabled:
                    grad_psi0_p = cic_interpolate(grad_psi0_grid, assim.x, L)
                    dx_total = dx_total - ps_ch.gamma * grad_psi0_p * dt
                if need_psi1 and pd_ch.enabled:
                    lap_psi1_p = cic_interpolate(lap_psi1_grid, assim.x, L)
                    dx_total = (
                        dx_total
                        - pd_ch.gamma * pd_ch.alpha * assim.v * lap_psi1_p * dt
                    )
                if need_psi2 and p2d_ch.enabled:
                    h3_p = cic_interpolate(grad3_psi2_grid, assim.x, L)
                    grad_E_h1psi2_p = cic_interpolate(
                        grad_E_h1psi2_grid, assim.x, L
                    )
                    grad_chi_p = cic_interpolate(grad_chi_grid, assim.x, L)
                    p2d_term = (
                        assim.v * assim.v * h3_p
                        + grad_E_h1psi2_p
                        + grad_chi_p
                    )
                    dx_total = dx_total - p2d_ch.gamma * p2d_ch.alpha * p2d_term * dt
                assim.x = np.mod(assim.x + dx_total, L)
                # x moved -> redeposit rho and re-solve phi, since the velocity
                # channel below interpolates fields at the new positions.
                field_solve(assim)

            # Velocity channel — snapshot, dtobs, and the new d2tobs
            # contributions all sum into a single dU update.
            #
            #   dU = E dt
            #      - gamma * grad_psi0 dt                           (vs)
            #      - gamma * alpha * grad_psi1 dt                   (vd)
            #      - 2 * gamma * beta * U * d^2 psi_2 / dx^2 dt     (v2d)
            #
            # The d2tobs term is the FIRST v-dependent force in the
            # augmented functional — see §1.2 of the second-derivative
            # plan doc.
            any_velocity = (
                vs_ch.enabled
                or (need_psi1 and vd_ch.enabled)
                or (need_psi2 and v2d_ch.enabled)
            )
            if any_velocity:
                dv_total = np.zeros_like(assim.v)
                if vs_ch.enabled:
                    grad_psi0_p = cic_interpolate(grad_psi0_grid, assim.x, L)
                    dv_total = dv_total + vs_ch.gamma * grad_psi0_p
                if need_psi1 and vd_ch.enabled:
                    grad_psi1_p = cic_interpolate(grad_psi1_grid, assim.x, L)
                    dv_total = dv_total + vd_ch.gamma * vd_ch.alpha * grad_psi1_p
                if need_psi2 and v2d_ch.enabled:
                    lap_psi2_p = cic_interpolate(lap_psi2_grid, assim.x, L)
                    # The factor of 2 is from the symmetry of the v . v
                    # outer product in the variational derivative.
                    dv_total = (
                        dv_total
                        + 2.0 * v2d_ch.gamma * v2d_ch.alpha * assim.v * lap_psi2_p
                    )
                assim.v = assim.v - dv_total * dt

            # Persist the latest observation(s). y_prev2 takes the previous
            # y_prev before y_prev itself is updated to the new y.
            if tdcfg.enabled or t2cfg.enabled:
                if y_prev is not None:
                    y_prev2 = y_prev
                    t_prev2 = t_prev
                y_prev = y.copy()
                t_prev = t_now_obs

        # 5) Second half-kick.
        push_leapfrog_half(truth, 0.5)
        push_leapfrog_half(assim, 0.5)
        truth.t += dt
        assim.t += dt

        # 6) Diagnostics.
        if (n % diag_cfg.every_diag_steps) == 0 or n == n_steps - 1:
            t_now = (n + 1) * dt
            e_phi = potential_error(assim.phi, truth.phi, L)
            rho_a = cic_deposit(assim.x, assim.w, L, Nx)
            rho_t = cic_deposit(truth.x, truth.w, L, Nx)
            e_rho = density_error(rho_a, rho_t, L)
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
            # Globally-integrated kinetic stress, M_total(t) = sum_p w_p v_p^2
            # = integral over (x, v) of v^2 f. Logged for both truth and assim
            # so we can plot the time evolution of the second moment that the
            # d2tobs channels are designed to drive.
            M_total_t = float(np.sum(truth.w * truth.v * truth.v))
            M_total_a = float(np.sum(assim.w * assim.v * assim.v))
            truth_log.push(t_now, energy=en_t, modes=modes_t,
                           M_total=M_total_t)
            assim_kwargs: dict[str, Any] = dict(
                e_phi=e_phi, e_rho=e_rho, e_f=e_f, M_total=M_total_a,
                energy=en_a, modes=modes_a,
            )
            if tdcfg.enabled:
                j_a = cic_deposit_current(assim.x, assim.v, assim.w, L, Nx)
                j_t = cic_deposit_current(truth.x, truth.v, truth.w, L, Nx)
                assim_kwargs["e_j"] = current_error(j_a, j_t, L)
            if t2cfg.enabled:
                M_a = cic_deposit_kinetic_stress(assim.x, assim.v, assim.w, L, Nx)
                M_t = cic_deposit_kinetic_stress(truth.x, truth.v, truth.w, L, Nx)
                assim_kwargs["e_M"] = kinetic_stress_error(M_a, M_t, L)
                assim_kwargs["e_var_v"] = vmarginal_variance_error(
                    assim.v, assim.w, truth.v, truth.w,
                )
            assim_log.push(t_now, **assim_kwargs)

        # Snapshot save (post second half-kick, post diagnostics for the step).
        step = n + 1
        if step in snap_steps_set:
            snaps_truth[step] = {"x": truth.x.copy(), "v": truth.v.copy(),
                                 "w": truth.w.copy(),
                                 "phi": (truth.phi.copy() if truth.phi is not None else np.zeros(0)),
                                 "t": truth.t}
            snaps_assim[step] = {"x": assim.x.copy(), "v": assim.v.copy(),
                                 "w": assim.w.copy(),
                                 "phi": (assim.phi.copy() if assim.phi is not None else np.zeros(0)),
                                 "t": assim.t}

    return AssimilationOutput(
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
