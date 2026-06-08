"""2D2V assimilation main loop -- mirrors assimilation_moments.run_moments.

Two synchronised states: truth (no nudging, generates the moment obs) and
assim (formulation A_var or B applied between drift and second half-kick).
Both share the same external F_ext driver and same collision operator; the
only difference is the IC and the nudging.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .backend_reference_2d import (
    ReferenceState2D,
    cic_interpolate_2d,
    field_solve_2d,
    make_state_2d,
    push_leapfrog_drift_2d,
    push_leapfrog_half_2d,
)
from .collisions_2d import bgk_substep_2d, lb_substep_2d
from .config_2d import MomentRun2DCfg
from .diagnostics_2d import (
    density_error_2d,
    electric_energy_2d,
    grid_moments_2d2v,
    potential_error_2d,
    temperature_error_2d,
    velocity_error_2d,
)
from .driver_2d import (
    EExtFunc2D,
    make_checkerboard_standing,
    make_oblique_traveling_continuous,
    make_oblique_wave,
    make_oblique_wavepacket,
)
from .initial_conditions_2d import (
    ic_blob_2d,
    ic_phase_error_2d,
    perturbed_maxwellian_2d,
    wrong_maxwellian_2d,
)
from .nudging_moments_2d import (
    FormulationAParams2D,
    FormulationAVariantParams2D,
    FormulationBParams2D,
    FormulationCParams2D,
    FormulationAOTParams2D,
    apply_formulation_A_2d,
    apply_formulation_A_variant_2d,
    apply_formulation_B_2d,
    apply_formulation_C_2d,
    apply_aot_2d,
)
from .observation_moments_2d import (
    MomentObservationSpec2D,
    observe_moments_2d,
)


def _sample_ic_2d(ic, Np, Lx, Ly, kx, ky, rng):
    if ic.kind == "perturbed_maxwellian_2d":
        return perturbed_maxwellian_2d(Np, Lx, Ly, kx, ky,
                                        alpha=ic.alpha, sigma=ic.sigma, rng=rng)
    if ic.kind == "ic_phase_error_2d":
        return ic_phase_error_2d(Np, Lx, Ly, kx, ky,
                                  alpha=ic.alpha, theta0=ic.theta0,
                                  sigma=ic.sigma, rng=rng)
    if ic.kind == "ic_blob_2d":
        return ic_blob_2d(Np, Lx, Ly, ic.x0, ic.y0,
                          epsilon=ic.epsilon, sigma_blob=ic.sigma_blob,
                          sigma=ic.sigma, rng=rng)
    if ic.kind == "wrong_maxwellian_2d":
        return wrong_maxwellian_2d(Np, Lx, Ly, kx, ky,
                                    rho_amp=ic.rho_amp, theta0=ic.theta0,
                                    ux_star=ic.ux_star, uy_star=ic.uy_star,
                                    T_star=ic.T_star, rng=rng)
    raise ValueError(f"Unknown IC kind: {ic.kind!r}")


def _build_driver_2d(cfg) -> EExtFunc2D | None:
    if cfg.kind == "none" or cfg.A == 0.0:
        return None
    if cfg.kind == "oblique_wave":
        return make_oblique_wave(cfg.A, cfg.kx_d, cfg.ky_d, cfg.omega, cfg.theta)
    if cfg.kind == "oblique_wavepacket":
        return make_oblique_wavepacket(cfg.A, cfg.kx_d, cfg.ky_d, cfg.omega,
                                        cfg.t0, cfg.sigma_t, cfg.theta)
    if cfg.kind == "checkerboard_standing":
        return make_checkerboard_standing(cfg.A, cfg.kx_d, cfg.ky_d, cfg.omega)
    if cfg.kind == "oblique_traveling_continuous":
        return make_oblique_traveling_continuous(
            cfg.A, cfg.kx_d, cfg.ky_d, cfg.omega,
            ramp_time=cfg.ramp_time, theta=cfg.theta,
        )
    raise ValueError(f"Unknown driver kind: {cfg.kind!r}")


@dataclass
class DiagnosticsLog2D:
    # Cadence: every_diag_steps. ABC error metrics + electric energy.
    e_phi: list[float] = field(default_factory=list)
    e_rho: list[float] = field(default_factory=list)
    e_u: list[float] = field(default_factory=list)
    e_T: list[float] = field(default_factory=list)
    energy: list[float] = field(default_factory=list)
    # Cadence: every_diag_steps. Fourier amplitudes at driver mode (kx_d, ky_d),
    # stored as complex so phase can be recovered.
    phi_hat: list[complex] = field(default_factory=list)
    rho_hat: list[complex] = field(default_factory=list)
    jlong_hat: list[complex] = field(default_factory=list)  # (kx_d jx + ky_d jy)/|k|
    T_hat: list[complex] = field(default_factory=list)
    # Cadence: every_diag_steps. L2 norms of conserved-moment residuals between
    # assim and truth (zero for the truth log).
    r0_norm: list[float] = field(default_factory=list)
    r1_norm: list[float] = field(default_factory=list)
    r2_norm: list[float] = field(default_factory=list)
    # Cadence: every nudge step. RMS over particles of the nudging correction
    # applied at that step (zero-length for none and for truth).
    t_nudge: list[float] = field(default_factory=list)
    bx_rms: list[float] = field(default_factory=list)
    by_rms: list[float] = field(default_factory=list)
    bvx_rms: list[float] = field(default_factory=list)
    bvy_rms: list[float] = field(default_factory=list)


def _mode_hat_2d(field: np.ndarray, mx: int, my: int) -> complex:
    """Normalised complex Fourier coefficient at integer mode (mx, my)."""
    Nx, Ny = field.shape
    return complex(np.fft.fft2(field)[mx % Nx, my % Ny] / (Nx * Ny))


def _resolve_driver_mode_2d(cfg: MomentRun2DCfg) -> tuple[int, int, float, float, float]:
    """Return (mx, my, kx_d, ky_d, |k|) — integer mode indices matching the
    driver wavevector on the (Nx, Ny) grid with domain (Lx, Ly). Falls back to
    (1, 1) for none-driver runs so plots are still informative."""
    Lx, Ly = cfg.Lx, cfg.Ly
    kx_d = cfg.driver.kx_d if cfg.driver.kind != "none" else cfg.domain.kx
    ky_d = cfg.driver.ky_d if cfg.driver.kind != "none" else cfg.domain.ky
    mx_f = kx_d * Lx / (2.0 * np.pi)
    my_f = ky_d * Ly / (2.0 * np.pi)
    mx = int(round(mx_f))
    my = int(round(my_f))
    k_mag = float(np.sqrt(kx_d * kx_d + ky_d * ky_d))
    return mx, my, float(kx_d), float(ky_d), k_mag


@dataclass
class MomentAssimilationOutput2D:
    t: np.ndarray
    truth_log: dict[str, Any]
    assim_log: dict[str, Any]
    final_truth: dict[str, np.ndarray]
    final_assim: dict[str, np.ndarray]


def _snapshot_2d(state: ReferenceState2D) -> dict[str, np.ndarray]:
    return {
        "x": state.x.copy(),
        "y": state.y.copy(),
        "vx": state.vx.copy(),
        "vy": state.vy.copy(),
        "w": state.w.copy(),
    }


def run_moments_2d(
    cfg: MomentRun2DCfg,
    snapshot_steps: list[int] | None = None,
    nudge_until_step: int | None = None,
) -> MomentAssimilationOutput2D:
    Lx = cfg.Lx
    Ly = cfg.Ly
    Nx = cfg.pic.Nx
    Ny = cfg.pic.Ny
    Np = cfg.pic.Np
    dt = cfg.pic.dt
    n_steps = cfg.pic.n_steps

    rng_truth = np.random.default_rng(cfg.seed)
    rng_assim = np.random.default_rng(cfg.seed + 1)

    # ---- Sample ICs ----
    xt, yt, vxt, vyt, wt = _sample_ic_2d(
        cfg.truth_ic, Np, Lx, Ly, cfg.domain.kx, cfg.domain.ky, rng_truth,
    )
    xa, ya, vxa, vya, wa = _sample_ic_2d(
        cfg.assim_ic, Np, Lx, Ly, cfg.domain.kx, cfg.domain.ky, rng_assim,
    )

    E_ext_func = _build_driver_2d(cfg.driver)

    truth = make_state_2d(xt, yt, vxt, vyt, wt, Lx, Ly, Nx, Ny, dt, E_ext_func)
    assim = make_state_2d(xa, ya, vxa, vya, wa, Lx, Ly, Nx, Ny, dt, E_ext_func)

    # ---- Observation spec ----
    obs_spec = MomentObservationSpec2D(
        kind=cfg.moment_observation.kind,
        sigma_rho=cfg.moment_observation.sigma_rho,
        sigma_u=cfg.moment_observation.sigma_u,
        sigma_T=cfg.moment_observation.sigma_T,
        rho_floor=cfg.moment_observation.rho_floor,
        T_floor=cfg.moment_observation.T_floor,
        every_q=cfg.moment_observation.every_q,
        rng_seed=cfg.moment_observation.rng_seed,
    )
    obs_rng = np.random.default_rng(obs_spec.rng_seed)

    # ---- Diagnostics + snapshots ----
    truth_log = DiagnosticsLog2D()
    assim_log = DiagnosticsLog2D()
    diag_cfg = cfg.diagnostics
    snapshot_steps = snapshot_steps if snapshot_steps is not None else diag_cfg.snapshot_steps
    if not snapshot_steps:
        snapshot_steps = sorted({int(round(i * n_steps / 6)) for i in range(7)})
    snap_steps_set = set(snapshot_steps)
    snaps_truth: dict[int, dict[str, np.ndarray]] = {}
    snaps_assim: dict[int, dict[str, np.ndarray]] = {}
    if 0 in snap_steps_set:
        snaps_truth[0] = _snapshot_2d(truth)
        snaps_assim[0] = _snapshot_2d(assim)

    # ---- Pre-build formulation params ----
    formulation = cfg.moment_nudge.formulation
    A_params = FormulationAParams2D(
        gamma_1=cfg.moment_nudge.A.gamma_1,
        gamma_2=cfg.moment_nudge.A.gamma_2,
        gamma_3=cfg.moment_nudge.A.gamma_3,
        V_star=cfg.moment_nudge.A.V_star,
        lowpass_k_cut_frac=cfg.moment_nudge.A.lowpass_k_cut_frac,
        lowpass_sharpness=cfg.moment_nudge.A.lowpass_sharpness,
    )
    A_var_params = FormulationAVariantParams2D(
        gamma_1=cfg.moment_nudge.A_var.gamma_1,
        gamma_2=cfg.moment_nudge.A_var.gamma_2,
        gamma_3=cfg.moment_nudge.A_var.gamma_3,
        lowpass_k_cut_frac=cfg.moment_nudge.A_var.lowpass_k_cut_frac,
        lowpass_sharpness=cfg.moment_nudge.A_var.lowpass_sharpness,
        rho_floor=cfg.moment_nudge.A_var.rho_floor,
        T_floor=cfg.moment_nudge.A_var.T_floor,
    )
    B_params = FormulationBParams2D(
        gamma=cfg.moment_nudge.B.gamma,
        gamma_1=cfg.moment_nudge.B.gamma_1,
        gamma_2=cfg.moment_nudge.B.gamma_2,
        gamma_3=cfg.moment_nudge.B.gamma_3,
        lowpass_k_cut_frac=cfg.moment_nudge.B.lowpass_k_cut_frac,
        lowpass_sharpness=cfg.moment_nudge.B.lowpass_sharpness,
        rho_floor=cfg.moment_nudge.B.rho_floor,
        T_floor=cfg.moment_nudge.B.T_floor,
    )
    C_params = FormulationCParams2D(
        lam=cfg.moment_nudge.C.lam,
        use_weighted_metric=cfg.moment_nudge.C.use_weighted_metric,
        V_star=cfg.moment_nudge.C.V_star,
        lowpass_k_cut_frac=cfg.moment_nudge.C.lowpass_k_cut_frac,
        lowpass_sharpness=cfg.moment_nudge.C.lowpass_sharpness,
        rho_floor=cfg.moment_nudge.C.rho_floor,
        T_floor=cfg.moment_nudge.C.T_floor,
    )
    AOT_params = FormulationAOTParams2D(
        mu_rho=cfg.moment_nudge.aot.mu_rho,
        mu_u=cfg.moment_nudge.aot.mu_u,
        mu_T=cfg.moment_nudge.aot.mu_T,
        lowpass_k_cut_frac=cfg.moment_nudge.aot.lowpass_k_cut_frac,
        lowpass_sharpness=cfg.moment_nudge.aot.lowpass_sharpness,
        rho_floor=cfg.moment_nudge.aot.rho_floor,
        T_floor=cfg.moment_nudge.aot.T_floor,
    )
    collision_kind = cfg.collision.kind
    collision_nu = cfg.collision.nu if collision_kind in ("bgk", "lb") else 0.0
    coll_rng_truth = np.random.default_rng(cfg.seed + 100)
    coll_rng_assim = np.random.default_rng(cfg.seed + 200)

    mx, my, kx_d, ky_d, k_mag = _resolve_driver_mode_2d(cfg)
    inv_k_mag = 1.0 / k_mag if k_mag > 0.0 else 0.0

    times = []

    for n in range(n_steps):
        # 1) Half-kick
        push_leapfrog_half_2d(truth, 0.5)
        push_leapfrog_half_2d(assim, 0.5)
        # 2) Drift
        push_leapfrog_drift_2d(truth)
        push_leapfrog_drift_2d(assim)
        # 3) Field solve
        field_solve_2d(truth)
        field_solve_2d(assim)
        # 4) Observe + nudge
        nudging_active = (
            formulation != "none"
            and (nudge_until_step is None or n < nudge_until_step)
        )
        do_obs = (n % obs_spec.every_q) == 0 and nudging_active
        if do_obs:
            rho_obs, ux_obs, uy_obs, T_obs = observe_moments_2d(truth, obs_spec, obs_rng)
            x_was = assim.x.copy()
            y_was = assim.y.copy()
            vx_was = assim.vx.copy()
            vy_was = assim.vy.copy()
            if formulation == "A":
                assim.x, assim.y, assim.vx, assim.vy = apply_formulation_A_2d(
                    assim.x, assim.y, assim.vx, assim.vy, assim.w,
                    Lx, Ly, Nx, Ny,
                    rho_obs, ux_obs, uy_obs, T_obs, A_params, dt,
                )
            elif formulation == "A_var":
                assim.x, assim.y, assim.vx, assim.vy = apply_formulation_A_variant_2d(
                    assim.x, assim.y, assim.vx, assim.vy, assim.w,
                    Lx, Ly, Nx, Ny,
                    rho_obs, ux_obs, uy_obs, T_obs, A_var_params, dt,
                )
            elif formulation == "B":
                assim.x, assim.y, assim.vx, assim.vy = apply_formulation_B_2d(
                    assim.x, assim.y, assim.vx, assim.vy, assim.w,
                    Lx, Ly, Nx, Ny,
                    rho_obs, ux_obs, uy_obs, T_obs, B_params, dt,
                )
            elif formulation == "C":
                assim.x, assim.y, assim.vx, assim.vy = apply_formulation_C_2d(
                    assim.x, assim.y, assim.vx, assim.vy, assim.w,
                    Lx, Ly, Nx, Ny,
                    rho_obs, ux_obs, uy_obs, T_obs, C_params, dt,
                )
            elif formulation == "aot":
                assim.x, assim.y, assim.vx, assim.vy = apply_aot_2d(
                    assim.x, assim.y, assim.vx, assim.vy, assim.w,
                    Lx, Ly, Nx, Ny,
                    rho_obs, ux_obs, uy_obs, T_obs, AOT_params, dt,
                )
            # RMS nudging-correction norms (unwrap periodic boundary).
            dx_b = assim.x - x_was
            dx_b -= Lx * np.round(dx_b / Lx)
            dy_b = assim.y - y_was
            dy_b -= Ly * np.round(dy_b / Ly)
            assim_log.t_nudge.append((n + 1) * dt)
            assim_log.bx_rms.append(float(np.sqrt(np.mean(dx_b * dx_b))))
            assim_log.by_rms.append(float(np.sqrt(np.mean(dy_b * dy_b))))
            assim_log.bvx_rms.append(float(np.sqrt(np.mean((assim.vx - vx_was) ** 2))))
            assim_log.bvy_rms.append(float(np.sqrt(np.mean((assim.vy - vy_was) ** 2))))
            if not (np.array_equal(x_was, assim.x) and np.array_equal(y_was, assim.y)):
                field_solve_2d(assim)
        # 5) Collision
        if collision_nu > 0.0:
            substep = bgk_substep_2d if collision_kind == "bgk" else lb_substep_2d
            substep(truth, collision_nu, coll_rng_truth,
                    rho_floor=cfg.collision.rho_floor, T_floor=cfg.collision.T_floor)
            substep(assim, collision_nu, coll_rng_assim,
                    rho_floor=cfg.collision.rho_floor, T_floor=cfg.collision.T_floor)
        # 6) Second half-kick
        push_leapfrog_half_2d(truth, 0.5)
        push_leapfrog_half_2d(assim, 0.5)
        truth.t += dt
        assim.t += dt

        # 7) Diagnostics
        if (n % diag_cfg.every_diag_steps) == 0 or n == n_steps - 1:
            t_now = (n + 1) * dt
            times.append(t_now)
            rho_t, ux_t, uy_t, T_t = grid_moments_2d2v(truth)
            rho_a, ux_a, uy_a, T_a = grid_moments_2d2v(assim)
            assim_log.e_phi.append(potential_error_2d(assim.phi, truth.phi))
            assim_log.e_rho.append(density_error_2d(rho_a, rho_t))
            assim_log.e_u.append(velocity_error_2d(rho_a, ux_a, uy_a, rho_t, ux_t, uy_t, T_truth=T_t))
            assim_log.e_T.append(temperature_error_2d(rho_a, T_a, rho_t, T_t))
            assim_log.energy.append(electric_energy_2d(assim.Ex, assim.Ey, Lx, Ly))
            truth_log.energy.append(electric_energy_2d(truth.Ex, truth.Ey, Lx, Ly))

            # Fourier mode at driver wavevector (truth and assim).
            jx_t, jy_t = rho_t * ux_t, rho_t * uy_t
            jx_a, jy_a = rho_a * ux_a, rho_a * uy_a
            truth_log.phi_hat.append(_mode_hat_2d(truth.phi, mx, my))
            truth_log.rho_hat.append(_mode_hat_2d(rho_t, mx, my))
            truth_log.T_hat.append(_mode_hat_2d(T_t, mx, my))
            truth_log.jlong_hat.append(
                (kx_d * _mode_hat_2d(jx_t, mx, my) + ky_d * _mode_hat_2d(jy_t, mx, my)) * inv_k_mag
            )
            assim_log.phi_hat.append(_mode_hat_2d(assim.phi, mx, my))
            assim_log.rho_hat.append(_mode_hat_2d(rho_a, mx, my))
            assim_log.T_hat.append(_mode_hat_2d(T_a, mx, my))
            assim_log.jlong_hat.append(
                (kx_d * _mode_hat_2d(jx_a, mx, my) + ky_d * _mode_hat_2d(jy_a, mx, my)) * inv_k_mag
            )
            # Moment-residual norms: r0=rho, r1=j (vector L2), r2=E.
            r0 = rho_a - rho_t
            r1x = jx_a - jx_t
            r1y = jy_a - jy_t
            E_a = 0.5 * rho_a * (ux_a * ux_a + uy_a * uy_a) + rho_a * T_a
            E_t = 0.5 * rho_t * (ux_t * ux_t + uy_t * uy_t) + rho_t * T_t
            r2 = E_a - E_t
            assim_log.r0_norm.append(float(np.sqrt(np.mean(r0 * r0))))
            assim_log.r1_norm.append(float(np.sqrt(np.mean(r1x * r1x + r1y * r1y))))
            assim_log.r2_norm.append(float(np.sqrt(np.mean(r2 * r2))))

        n_next = n + 1
        if n_next in snap_steps_set:
            snaps_truth[n_next] = _snapshot_2d(truth)
            snaps_assim[n_next] = _snapshot_2d(assim)

    out = MomentAssimilationOutput2D(
        t=np.asarray(times),
        truth_log={
            "energy": np.asarray(truth_log.energy),
            "phi_hat": np.asarray(truth_log.phi_hat, dtype=complex),
            "rho_hat": np.asarray(truth_log.rho_hat, dtype=complex),
            "jlong_hat": np.asarray(truth_log.jlong_hat, dtype=complex),
            "T_hat": np.asarray(truth_log.T_hat, dtype=complex),
            "snapshots": snaps_truth,
            "driver_mode": np.array([mx, my, kx_d, ky_d, k_mag]),
        },
        assim_log={
            "e_phi": np.asarray(assim_log.e_phi),
            "e_rho": np.asarray(assim_log.e_rho),
            "e_u":   np.asarray(assim_log.e_u),
            "e_T":   np.asarray(assim_log.e_T),
            "energy": np.asarray(assim_log.energy),
            "phi_hat": np.asarray(assim_log.phi_hat, dtype=complex),
            "rho_hat": np.asarray(assim_log.rho_hat, dtype=complex),
            "jlong_hat": np.asarray(assim_log.jlong_hat, dtype=complex),
            "T_hat": np.asarray(assim_log.T_hat, dtype=complex),
            "r0_norm": np.asarray(assim_log.r0_norm),
            "r1_norm": np.asarray(assim_log.r1_norm),
            "r2_norm": np.asarray(assim_log.r2_norm),
            "t_nudge": np.asarray(assim_log.t_nudge),
            "bx_rms":  np.asarray(assim_log.bx_rms),
            "by_rms":  np.asarray(assim_log.by_rms),
            "bvx_rms": np.asarray(assim_log.bvx_rms),
            "bvy_rms": np.asarray(assim_log.bvy_rms),
            "snapshots": snaps_assim,
        },
        final_truth=_snapshot_2d(truth),
        final_assim=_snapshot_2d(assim),
    )
    return out
