"""E1 — homogeneous-Maxwellian moment-convergence rate validation.

Validates the linearized rate predictions of the paper (Prop 5.1/5.2, Table 2)
on a homogeneous-Maxwellian target. Runs none/aot/A/B/C on the same config and
fits the exponential decay rates of e_u(t) and e_T(t). The headline check is the
*ratio* rate_T / rate_u:

    Formulation C        -> 2.0   (fixed 1:2, Fisher-Rao geometry, Prop 5.2)
    Formulations A/B/AOT -> ~1.0  (tunable; 1:1 at unit gains, Table 2)

Usage:
    python scripts/run_rate_validation.py --config configs/exp1_homogeneous_rate.yaml
    # smoke:
    python scripts/run_rate_validation.py --config configs/exp1_homogeneous_rate.yaml \
        --np-override 50000 --n-steps-override 250 --name-suffix _smoke
"""
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from plot_style import COLORS, LABELS, apply_style, smooth_log, figsize as figsz, tidy_log_yaxis  # noqa: E402,F401
apply_style()

import numpy as np

from mfda.assimilation_moments import run_moments
from mfda.config import load_moment

FORMULATIONS = ["none", "aot", "A", "B", "C"]
# Predicted rate ratio rate_T / rate_u at the default unit gains.
PREDICTED_RATIO = {"none": float("nan"), "aot": 1.0, "A": 1.0, "B": 1.0, "C": 2.0}
FIT_T_LO = 0.3   # skip the initial discrete-kick transient


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _series(out, name: str) -> np.ndarray:
    extras = out.assim_log.get("extras", {})
    if name in extras:
        return np.asarray(extras[name], dtype=float)
    if name in out.assim_log:
        return np.asarray(out.assim_log[name], dtype=float)
    return np.zeros(0)


def fit_rate(t: np.ndarray, e: np.ndarray) -> tuple[float, float, tuple]:
    """Fit e(t) ~ e0 * exp(-rate t) over the linear-in-log decay window.

    The window is [FIT_T_LO, t_floor] where t_floor is where e first falls to
    3x the estimated noise floor (median of the last 20% of the series).
    Returns (rate, floor, (t_window_lo, t_window_hi, n_points)).
    """
    if t.size < 6 or e.size != t.size:
        return float("nan"), float("nan"), (float("nan"), float("nan"), 0)
    e = np.where(np.isfinite(e), e, np.nan)
    tail = e[int(0.8 * e.size):]
    floor = float(np.nanmedian(tail)) if tail.size else 0.0
    floor = max(floor, 1e-12)
    above = e > 3.0 * floor
    # First contiguous window from FIT_T_LO where e stays above 3*floor.
    idx = np.where((t >= FIT_T_LO) & above & np.isfinite(e) & (e > 0))[0]
    if idx.size < 4:
        return float("nan"), floor, (float("nan"), float("nan"), int(idx.size))
    # Keep the leading contiguous run so we fit clean exponential decay only.
    cut = np.where(np.diff(idx) > 1)[0]
    if cut.size:
        idx = idx[: cut[0] + 1]
    if idx.size < 4:
        return float("nan"), floor, (float("nan"), float("nan"), int(idx.size))
    slope, _ = np.polyfit(t[idx], np.log(e[idx]), 1)
    return float(-slope), floor, (float(t[idx[0]]), float(t[idx[-1]]), int(idx.size))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--np-override", type=int, default=None)
    ap.add_argument("--n-steps-override", type=int, default=None)
    ap.add_argument("--name-suffix", default="")
    ap.add_argument("--strength", type=float, default=None,
                    help="Nudging-strength multiplier: scales C.lam and "
                         "A/B gammas uniformly (parameter-free / sensitivity sweep).")
    args = ap.parse_args()

    cfg = load_moment(args.config)
    overrides = {}
    if args.strength is not None:
        g = args.strength
        cfg.moment_nudge.C.lam = g
        cfg.moment_nudge.A.gamma_1 = cfg.moment_nudge.A.gamma_2 = cfg.moment_nudge.A.gamma_3 = g
        cfg.moment_nudge.B.gamma_x = cfg.moment_nudge.B.gamma_v = g
        cfg.moment_nudge.aot.mu_rho = cfg.moment_nudge.aot.mu_u = cfg.moment_nudge.aot.mu_T = g
        overrides["strength"] = g
    if args.np_override is not None:
        cfg.pic.Np = args.np_override
        overrides["Np"] = args.np_override
    if args.n_steps_override is not None:
        cfg.pic.n_steps = args.n_steps_override
        overrides["n_steps"] = args.n_steps_override

    out_dir = Path(cfg.outputs_dir) / (cfg.name + args.name_suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[rate-val] outputs: {out_dir}")

    series: dict[str, dict] = {}
    rows: list[dict] = []
    for f in FORMULATIONS:
        cfg.moment_nudge.formulation = f  # type: ignore[assignment]
        print(f"[rate-val] running formulation={f} ...", flush=True)
        out = run_moments(cfg)
        t = np.asarray(out.t, dtype=float)
        e_u = _series(out, "e_u")
        e_T = _series(out, "e_T")
        rate_u, floor_u, win_u = fit_rate(t, e_u)
        rate_T, floor_T, win_T = fit_rate(t, e_T)
        ratio = rate_T / rate_u if (np.isfinite(rate_u) and rate_u > 1e-6) else float("nan")
        series[f] = {"t": t, "e_u": e_u, "e_T": e_T,
                     "rate_u": rate_u, "rate_T": rate_T,
                     "floor_u": floor_u, "floor_T": floor_T}
        rows.append({
            "formulation": f,
            "rate_u": f"{rate_u:.4g}", "rate_T": f"{rate_T:.4g}",
            "ratio_T_over_u": f"{ratio:.4g}",
            "predicted_ratio": f"{PREDICTED_RATIO[f]:.3g}",
            "fit_window_u": f"[{win_u[0]:.2f},{win_u[1]:.2f}] n={win_u[2]}",
            "fit_window_T": f"[{win_T[0]:.2f},{win_T[1]:.2f}] n={win_T[2]}",
        })
        print(f"[rate-val]   rate_u={rate_u:.4g} rate_T={rate_T:.4g} "
              f"ratio={ratio:.4g} (pred {PREDICTED_RATIO[f]})", flush=True)

    # CSV.
    csv_path = out_dir / "rate_summary.csv"
    with open(csv_path, "w", newline="") as fp:
        wr = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # Figure: e_u and e_T decay vs t (semilogy), smoothed, with fitted rates.
    fig, axes = plt.subplots(1, 2, figsize=figsz(1.0, 0.40))
    panel_titles = {"e_u": r"$e_u$  (bulk velocity)",
                    "e_T": r"$e_T$  (temperature)"}
    for name, ax in (("e_u", axes[0]), ("e_T", axes[1])):
        for f in FORMULATIONS:
            s = series[f]
            t, e = s["t"], s[name]
            if e.size == 0:
                continue
            rate = s["rate_" + name[-1]]
            lab = LABELS.get(f, f) + ("" if not np.isfinite(rate)
                                      else f" (rate ${rate:.2g}$)")
            e = np.maximum(e, 1e-16)
            ax.semilogy(t, e, color=COLORS[f], alpha=0.15, lw=0.7)
            ax.semilogy(t, smooth_log(e), color=COLORS[f], lw=1.8, label=lab)
        ax.set_title(panel_titles[name])
        ax.set_xlabel("$t$")
        ax.grid(True, which="both", alpha=0.3)
        tidy_log_yaxis(ax)
        ax.legend()
    fig.tight_layout()
    fig_path = out_dir / "rate_curves.png"
    fig.savefig(fig_path)
    plt.close(fig)

    # Save the decay curves so the figure can be restyled without re-running.
    np.savez(
        out_dir / "rate_curves.npz",
        **{f"{f}_t": series[f]["t"] for f in FORMULATIONS},
        **{f"{f}_e_u": series[f]["e_u"] for f in FORMULATIONS},
        **{f"{f}_e_T": series[f]["e_T"] for f in FORMULATIONS},
    )

    # SUMMARY.md
    lines = [f"# {cfg.name + args.name_suffix}", "",
             f"git SHA: `{_git_sha()}`"]
    if overrides:
        lines.append(f"Overrides: `{overrides}`")
    lines += [
        "",
        "## Moment-convergence rate validation (Prop 5.1/5.2, Table 2)",
        "",
        "Fitted exponential decay rates of e_u(t), e_T(t) on a homogeneous",
        "Maxwellian target. Headline check = ratio rate_T / rate_u:",
        "**C should be ~2 (fixed 1:2); A/B/AOT ~1 at unit gains.**",
        "",
        "| formulation | rate_u | rate_T | ratio T/u | predicted | fit window (u) | fit window (T) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append("| " + " | ".join([
            r["formulation"], r["rate_u"], r["rate_T"], r["ratio_T_over_u"],
            r["predicted_ratio"], r["fit_window_u"], r["fit_window_T"]]) + " |")
    lines += ["", "## Rate curves", "", f"![rate curves]({fig_path.name})", ""]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines))
    print(f"[rate-val] wrote {csv_path}, {fig_path}, SUMMARY.md")
    print("[rate-val] done.")


if __name__ == "__main__":
    main()
