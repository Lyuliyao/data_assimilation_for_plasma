"""Diagnostics listed in section 3 of the note.

All error norms are discrete L2 norms on the Poisson grid or on a phase-space
histogram. Each function returns a scalar per call; the caller stitches the
time-series.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def l2_on_grid(a: np.ndarray, L: float) -> float:
    """Discrete L2 norm on a periodic uniform grid of length L."""
    dx = L / a.shape[0]
    return float(np.sqrt(dx * np.sum(a * a)))


def l2_on_grid_2d(a: np.ndarray, Lx: float, Lv: float) -> float:
    dx = Lx / a.shape[0]
    dv = Lv / a.shape[1]
    return float(np.sqrt(dx * dv * np.sum(a * a)))


def potential_error(phi: np.ndarray, phi_truth: np.ndarray, L: float) -> float:
    return l2_on_grid(phi - phi_truth, L)


def density_error(rho: np.ndarray, rho_truth: np.ndarray, L: float) -> float:
    return l2_on_grid(rho - rho_truth, L)


def current_error(j: np.ndarray, j_truth: np.ndarray, L: float) -> float:
    """Discrete L2 norm of (j - j_truth) on the periodic grid.

    j and j_truth are first velocity moments of f, deposited via
    backend_reference.cic_deposit_current. Used to verify the new
    time-derivative observation channel is being matched.
    """
    return l2_on_grid(j - j_truth, L)


def kinetic_stress_error(M: np.ndarray, M_truth: np.ndarray, L: float) -> float:
    """Discrete L2 norm of (M - M_truth) on the periodic grid.

    M = integral v^2 f dv (second velocity moment, kinetic stress) deposited
    via mfda.kinetic_stress.cic_deposit_kinetic_stress. Used to verify the
    new second-time-derivative observation channel is being matched.
    """
    return l2_on_grid(M - M_truth, L)


def velocity_moment_error(
    u: np.ndarray, u_truth: np.ndarray, rho_truth: np.ndarray, L: float,
    rho_floor: float = 1.0e-3,
) -> float:
    """Density-weighted L^2 norm of (u - u_truth) on the periodic grid.

    Used for the bulk-velocity error e_u when both fields are derived from
    PIC moments. The density weight is the truth density, restricted to
    cells where rho_truth > rho_floor — this avoids blowing up in nearly-
    empty cells where u is statistically meaningless.

        e_u^2 = (1/L) * integral [rho_truth(x) * (u - u_truth)^2] / mean(rho_truth) dx
              = average over x of (rho_truth/<rho>) * (u - u_truth)^2
    """
    mask = rho_truth > rho_floor
    if not mask.any():
        return 0.0
    dx = L / u.shape[0]
    rho_mean = float(np.mean(rho_truth))
    if rho_mean <= 0.0:
        rho_mean = 1.0
    weight = np.where(mask, rho_truth / rho_mean, 0.0)
    diff_sq = (u - u_truth) ** 2
    return float(np.sqrt(dx / L * np.sum(weight * diff_sq)))


def temperature_error(
    T: np.ndarray, T_truth: np.ndarray, rho_truth: np.ndarray, L: float,
    rho_floor: float = 1.0e-3,
) -> float:
    """Density-weighted L^2 norm of (T - T_truth). Same masking as e_u."""
    mask = rho_truth > rho_floor
    if not mask.any():
        return 0.0
    dx = L / T.shape[0]
    rho_mean = float(np.mean(rho_truth))
    if rho_mean <= 0.0:
        rho_mean = 1.0
    weight = np.where(mask, rho_truth / rho_mean, 0.0)
    diff_sq = (T - T_truth) ** 2
    return float(np.sqrt(dx / L * np.sum(weight * diff_sq)))


def vmarginal_variance_error(
    v: np.ndarray, w: np.ndarray,
    v_truth: np.ndarray, w_truth: np.ndarray,
) -> float:
    """|var(f_v) - var(f_v_truth)| (absolute difference of v-marginal variances).

    Variance is the leading bulk metric of the velocity marginal — sensitive
    to the bimodal-vs-Maxwellian mismatch in a way that the histogram L^2
    (e_f) is not. This is the metric the d2tobs term should most directly
    improve (it is the only term that delivers a v-dependent force, hence
    the only term that can change the v-marginal's variance).
    """
    w_total = float(np.sum(w))
    wt_total = float(np.sum(w_truth))
    mean_a = float(np.sum(w * v)) / w_total
    mean_t = float(np.sum(w_truth * v_truth)) / wt_total
    var_a = float(np.sum(w * (v - mean_a) ** 2)) / w_total
    var_t = float(np.sum(w_truth * (v_truth - mean_t) ** 2)) / wt_total
    return abs(var_a - var_t)


def electric_energy(E: np.ndarray, L: float) -> float:
    """E(t) = 1/2 * integral |E(x,t)|^2 dx."""
    dx = L / E.shape[0]
    return float(0.5 * dx * np.sum(E * E))


def low_k_modes(field: np.ndarray, n_modes: int = 6) -> np.ndarray:
    """Return |FFT(field)|[0..n_modes-1], i.e. magnitudes of the lowest modes."""
    F = np.fft.rfft(field)
    return np.abs(F[:n_modes])


def phase_space_histogram(
    x: np.ndarray, v: np.ndarray, w: np.ndarray,
    L: float, v_min: float, v_max: float, Nx: int, Nv: int,
) -> np.ndarray:
    """2D histogram of weighted particles on [0, L] x [v_min, v_max].

    Returns a (Nx, Nv) array normalized so that sum(hist * dx * dv) ~ total
    weight / domain.
    """
    H, _, _ = np.histogram2d(
        x, v, bins=(Nx, Nv),
        range=((0.0, L), (v_min, v_max)),
        weights=w,
    )
    return H / (L * (v_max - v_min) / (Nx * Nv))


def phase_space_error(
    x: np.ndarray, v: np.ndarray, w: np.ndarray,
    x_truth: np.ndarray, v_truth: np.ndarray, w_truth: np.ndarray,
    L: float, v_min: float, v_max: float, Nx: int = 64, Nv: int = 64,
) -> float:
    H = phase_space_histogram(x, v, w, L, v_min, v_max, Nx, Nv)
    H_t = phase_space_histogram(x_truth, v_truth, w_truth, L, v_min, v_max, Nx, Nv)
    return l2_on_grid_2d(H - H_t, L, v_max - v_min)


def conservation(x: np.ndarray, v: np.ndarray, w: np.ndarray, E: np.ndarray, L: float) -> dict[str, float]:
    """Return mass, momentum, and kinetic+field energy. Drift = now - t0."""
    mass = float(np.sum(w))
    momentum = float(np.sum(w * v))
    ke = 0.5 * float(np.sum(w * v * v))
    pe = electric_energy(E, L)
    return {"mass": mass, "momentum": momentum, "kinetic_energy": ke,
            "field_energy": pe, "total_energy": ke + pe}


@dataclass
class DiagnosticsLog:
    """Time-indexed accumulator for all diagnostics."""
    t: list[float] = field(default_factory=list)
    e_phi: list[float] = field(default_factory=list)
    e_rho: list[float] = field(default_factory=list)
    e_f: list[float] = field(default_factory=list)
    energy: list[float] = field(default_factory=list)
    modes: list[np.ndarray] = field(default_factory=list)
    conservation: list[dict[str, float]] = field(default_factory=list)
    extras: dict[str, list[Any]] = field(default_factory=dict)

    def push(self, t: float, **kwargs: Any) -> None:
        self.t.append(t)
        self.e_phi.append(kwargs.get("e_phi", float("nan")))
        self.e_rho.append(kwargs.get("e_rho", float("nan")))
        self.e_f.append(kwargs.get("e_f", float("nan")))
        self.energy.append(kwargs.get("energy", float("nan")))
        if "modes" in kwargs:
            self.modes.append(kwargs["modes"])
        if "conservation" in kwargs:
            self.conservation.append(kwargs["conservation"])
        for k, v in kwargs.items():
            if k in {"e_phi", "e_rho", "e_f", "energy", "modes", "conservation"}:
                continue
            self.extras.setdefault(k, []).append(v)

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": np.array(self.t),
            "e_phi": np.array(self.e_phi),
            "e_rho": np.array(self.e_rho),
            "e_f": np.array(self.e_f),
            "energy": np.array(self.energy),
            "modes": np.array(self.modes) if self.modes else np.zeros((0,)),
            "conservation": self.conservation,
            "extras": {k: np.array(v) for k, v in self.extras.items()},
        }
