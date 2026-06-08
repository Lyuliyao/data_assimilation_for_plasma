"""Initial-condition samplers for the PIC runs.

Each sampler returns (x, v) arrays of shape (Np,) drawn from the target
distribution, together with per-particle weights w of shape (Np,) so that

    rho(x) ~ sum_p w_p * S(x - x_p),

where S is the deposition shape. For an equilibrium-normalized sampler
(e.g. perturbed Maxwellian on a 1+cos(kx) background), we use the uniform-x
importance scheme: x ~ Uniform(0, L), v ~ target velocity marginal, and
w = (1 + alpha cos(kx)) for the spatial perturbation. This matches standard
Vlasov-PIC practice for linear Landau and is what the note expects.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DomainSpec:
    L: float   # spatial period
    v_min: float
    v_max: float


def _uniform_x(Np: int, L: float, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.0, L, size=Np)


def _gaussian_v(Np: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return sigma * rng.standard_normal(Np)


def perturbed_maxwellian(
    Np: int, L: float, k: float, alpha: float,
    sigma: float = 1.0, rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """f*(x, v) = (2*pi)^{-1/2} exp(-v^2/2) * (1 + alpha cos(kx)), isotropic sigma.

    Equation (14) of the note. Default sigma=1 corresponds to the standard
    weakly perturbed Maxwellian in plasma units.
    """
    rng = rng or np.random.default_rng(0)
    x = _uniform_x(Np, L, rng)
    v = _gaussian_v(Np, sigma, rng)
    w = 1.0 + alpha * np.cos(k * x)
    return x, v, w


def drifted_maxwellian(
    Np: int, L: float, k: float, alpha: float,
    sigma: float = 1.0, u0: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """f(x, v) = (1 + alpha cos(k x)) * M(v - u0; sigma).

    Spatially-perturbed Maxwellian with a uniform velocity drift u0.
    Used for the current-mismatch test: truth runs with +u0 and the
    assim model with -u0, so rho is identical at t=0 but j_truth =
    +u0 * rho(x), j_assim = -u0 * rho(x). Phi-only nudging cannot see
    the sign of u0; the d_t phi (continuity) channel does.
    """
    rng = rng or np.random.default_rng(0)
    x = _uniform_x(Np, L, rng)
    v = sigma * rng.standard_normal(Np) + u0
    w = 1.0 + alpha * np.cos(k * x)
    return x, v, w


def spatial_temperature(
    Np: int, L: float, k: float, alpha: float,
    theta_bg: float = 1.0, eta: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """f(x, v) = (1 + alpha cos kx) * M_{theta(x)}(v),  theta(x) = theta_bg + eta cos kx.

    Position-dependent Maxwellian variance: per-particle v is sampled
    with sigma = sqrt(theta(x_p)).

    Used for the spatial-temperature mismatch test: truth has eta=+eta_0,
    model has eta=-eta_0. The bulk temperature integral
    int theta(x) dx / L = theta_bg matches between runs (cos integrates
    to zero), so M_total is the same — the bulk-theta identifiability
    null seen in temperature_mismatch is REMOVED. The remaining ΔM is
    pure perturbation: 2 eta cos(kx)(1 + alpha cos(kx)), which is NOT
    in the kernel of ∂_x² and so gives the d_t^2 phi channel real signal.
    """
    rng = rng or np.random.default_rng(0)
    x = _uniform_x(Np, L, rng)
    theta_x = theta_bg + eta * np.cos(k * x)
    if (theta_x <= 0).any():
        raise ValueError(
            f"theta(x) must be positive everywhere; min(theta) = "
            f"{float(theta_x.min()):.4f} (theta_bg={theta_bg}, eta={eta})"
        )
    sigma_x = np.sqrt(theta_x)
    v = sigma_x * rng.standard_normal(Np)
    w = 1.0 + alpha * np.cos(k * x)
    return x, v, w


def two_peak_velocity(
    Np: int, L: float, k: float, alpha: float, u0: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """f2(x, v, 0) = rho0(x) * [M(v - u0) + M(v + u0)] / 2, eq. (13).

    Same spatial density as `perturbed_maxwellian` but with a velocity-bimodal
    distribution. Used for Test 0 (identifiability sanity check).
    """
    rng = rng or np.random.default_rng(0)
    x = _uniform_x(Np, L, rng)
    # Pick sign uniformly at random, then shift.
    signs = rng.choice([-1.0, 1.0], size=Np)
    v = rng.standard_normal(Np) + signs * u0
    w = 1.0 + alpha * np.cos(k * x)
    return x, v, w


def asymmetric_two_peak(
    Np: int, L: float, k: float, alpha: float,
    u_pos: float, u_neg: float, w_pos: float = 0.5,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two Maxwellian streams at v=u_pos and v=u_neg with weights (w_pos, 1 - w_pos).

    The net first velocity moment is
        j_truth(x) = rho_0(x) * (w_pos * u_pos + (1 - w_pos) * u_neg)
    which is non-zero whenever the two streams are unbalanced. With the default
    (u_pos=1.5, u_neg=-1.0, w_pos=0.5) we get j_truth = 0.25 * rho_0(x).

    Used for Phase C of the combined-nudge campaign — see
    docs/combined_nudge_amendment.md section 6 — to break the j-symmetry
    that pinned Test 0 / Test 1 hidden-velocity floors.
    """
    rng = rng or np.random.default_rng(0)
    x = _uniform_x(Np, L, rng)
    is_pos = rng.uniform(size=Np) < w_pos
    v = rng.standard_normal(Np)
    v[is_pos] += u_pos
    v[~is_pos] += u_neg
    w = 1.0 + alpha * np.cos(k * x)
    return x, v, w


def two_stream(
    Np: int, L: float, k: float, alpha: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """f*(x, v) = (2*pi)^{-1/2} v^2 exp(-v^2/2) * (1 + alpha cos(kx)), eq. (19).

    Rejection-sampled velocity from v^2 * exp(-v^2/2). Used for Test 2 (not in
    the first milestone).
    """
    rng = rng or np.random.default_rng(0)
    x = _uniform_x(Np, L, rng)
    # Inverse-CDF is cumbersome; use rejection with a Gaussian proposal.
    v = np.empty(Np)
    got = 0
    # proposal scale 1.5 covers the v^2 exp(-v^2/2) target well
    scale = 1.5
    while got < Np:
        prop = rng.normal(scale=scale, size=Np * 2)
        # target up to constant: v^2 exp(-v^2/2); proposal: exp(-v^2/(2 scale^2))
        ratio = (prop * prop) * np.exp(-0.5 * prop * prop + 0.5 * prop * prop / scale**2)
        M = (scale ** 2) * np.exp(-1.0)  # max of v^2 exp(-v^2/2) / proposal up to const
        u = rng.uniform(size=prop.shape[0])
        accept = u < ratio / M
        take = prop[accept]
        n_take = min(Np - got, take.shape[0])
        v[got:got + n_take] = take[:n_take]
        got += n_take
    w = 1.0 + alpha * np.cos(k * x)
    return x, v, w


def bump_on_tail(
    Np: int, L: float, k: float, alpha: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """d0(v) = (2*pi)^{-1/2} (0.9 exp(-v^2/2) + 0.2 exp(-2 (v - 4.5)^2)), eq. (22).

    The normalization in the note gives total weight 0.9 + 0.2 * sqrt(pi/2) / sqrt(pi)
    but we sample from the two components with their stated amplitudes and
    treat the integrated mass as a spatial weight (same convention as the
    other samplers). Used for Test 3 (not in the first milestone).
    """
    rng = rng or np.random.default_rng(0)
    # Component weights (unnormalized in v-space mass).
    # Integral of 0.9 exp(-v^2/2) dv = 0.9 * sqrt(2*pi)
    # Integral of 0.2 exp(-2 (v-4.5)^2) dv = 0.2 * sqrt(pi/2)
    mass_core = 0.9 * np.sqrt(2.0 * np.pi)
    mass_bump = 0.2 * np.sqrt(np.pi / 2.0)
    p_bump = mass_bump / (mass_core + mass_bump)
    u = rng.uniform(size=Np)
    is_bump = u < p_bump
    n_bump = int(is_bump.sum())
    n_core = Np - n_bump
    v = np.empty(Np)
    # Core: N(0, 1).
    v_core = rng.standard_normal(n_core)
    # Bump: N(4.5, 1/2).
    v_bump = 4.5 + 0.5 * rng.standard_normal(n_bump)
    v[~is_bump] = v_core
    v[is_bump] = v_bump
    x = _uniform_x(Np, L, rng)
    w = 1.0 + alpha * np.cos(k * x)
    return x, v, w


# ---- Initial guesses for the assimilated run (eqs. 16, 17, 18 of the note) ----

def ic_phase_error(Np: int, L: float, k: float, alpha: float, theta0: float,
                   rng: np.random.Generator | None = None):
    """Eq. (16): rho0_guess = 1 + alpha cos(k x + theta0), Maxwellian in v."""
    rng = rng or np.random.default_rng(0)
    x = _uniform_x(Np, L, rng)
    v = rng.standard_normal(Np)
    w = 1.0 + alpha * np.cos(k * x + theta0)
    return x, v, w


def ic_temperature_error(Np: int, L: float, k: float, alpha: float, sigma0: float,
                         rng: np.random.Generator | None = None):
    """Eq. (17): Maxwellian with variance sigma0^2 instead of 1."""
    rng = rng or np.random.default_rng(0)
    x = _uniform_x(Np, L, rng)
    v = sigma0 * rng.standard_normal(Np)
    w = 1.0 + alpha * np.cos(k * x)
    return x, v, w


def ic_hidden_velocity(Np: int, L: float, k: float, alpha: float, u0: float,
                       rng: np.random.Generator | None = None):
    """Eq. (18): same ICs as two_peak_velocity. Same rho0 as truth but bimodal v."""
    return two_peak_velocity(Np, L, k, alpha, u0, rng=rng)


def wrong_maxwellian(
    Np: int, L: float, k: float, alpha: float,
    rho_amp: float = 0.0, u_star: float = 0.0, T_star: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spatially-homogeneous Maxwellian with arbitrary (rho-perturbation, u, T).

    Used to initialise the *assimilated* run from a totally wrong prior in the
    note v3 §3 ABC recovery experiments. Independent of the truth IC — there
    is no shared random structure.

    Parameters
    ----------
    rho_amp : amplitude of an optional 1+rho_amp*cos(kx) density modulation
              (set to 0 for a perfectly uniform background, which is the
              "wrong-prior" baseline used in the recover-check).
    u_star  : bulk velocity of the assumed Maxwellian.
    T_star  : temperature of the assumed Maxwellian (variance of v sampled
              as sqrt(T_star) * standard_normal).

    Notes
    -----
    `alpha` and `k` are accepted for samp-er-API uniformity but are unused
    when rho_amp=0; they are kept so the dispatcher in assimilation.py can
    call this with the same kwarg signature as the other samplers.
    """
    rng = rng or np.random.default_rng(0)
    x = _uniform_x(Np, L, rng)
    v = u_star + np.sqrt(T_star) * rng.standard_normal(Np)
    if rho_amp != 0.0:
        w = 1.0 + rho_amp * np.cos(k * x)
    else:
        w = np.ones(Np)
    return x, v, w


def wrong_two_stream(
    Np: int, L: float, k: float, alpha: float,
    u_star: float = 0.0, T_star: float = 3.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-stream SHAPE with imposed (wrong) moments (mean u_star, variance T_star).

    Samples the two-stream velocity profile v^2 exp(-v^2/2) (effective variance
    <v^2> = 3, mean 0) and applies the affine map v -> u_star + sqrt(T_star/3) v.
    The affine map preserves the *bimodal shape* exactly while setting the bulk
    velocity to u_star and the temperature to T_star. Used to initialise the
    assimilated run with the correct non-Maxwellian shape but wrong moments, so
    the obstruction experiment can show that moment-manifold formulations
    (A/B/C/AOT) correct the moments while preserving the shape, whereas the
    naive KL-toward-pi_obs baseline Maxwellianizes it.
    """
    rng = rng or np.random.default_rng(0)
    x, v0, w = two_stream(Np, L, k, alpha, rng=rng)
    v = u_star + np.sqrt(T_star / 3.0) * v0
    return x, v, w
