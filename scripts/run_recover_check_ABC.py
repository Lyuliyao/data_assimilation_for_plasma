"""Recover-check across formulations none/A/B/C (note v3 §3 ABC).

Runs the same MomentRunCfg four times — once per formulation — and reports
whether each formulation drives the (e_rho, e_u, e_T, e_f) error series
down by >= one decade between [0, 5] and [10, 50] (in dimensionless time).

Usage:
    python scripts/run_recover_check_ABC.py \
        --config configs/test_moment_obs_ABC_driven_BGK.yaml

Smoke-mode overrides:
    python scripts/run_recover_check_ABC.py \
        --config configs/test_moment_obs_ABC_driven_BGK.yaml \
        --np-override 50000 --n-steps-override 200 \
        --name-suffix _smoke

Outputs (in <outputs_dir>/<name><suffix>/):
    {formulation}.npz             one per formulation (assim_log + truth_log)
    error_curves.png              2x2 figure: e_rho, e_u, e_T, e_f vs t
    recover_summary.csv           one row per formulation with judgement
    SUMMARY.md                    human-readable summary with figure embedded
"""
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mfda.assimilation_moments import run_moments
from mfda.config import load_moment


FORMULATIONS = ["none", "aot", "A", "B", "C"]
EARLY_WINDOW = (0.0, 5.0)
LATE_WINDOW = (10.0, 50.0)
DECADE = 10.0
# Formulation C's invariant is pi_obs (not f_truth) — judging it on e_f is
# unfair per §5.1, so the recover decision uses only the moment metrics.
METRICS_FOR_JUDGEMENT = {
    "none": ["e_rho", "e_u", "e_T", "e_f"],
    "aot": ["e_rho", "e_u", "e_T", "e_f"],
    "naive_kl": ["e_rho", "e_u", "e_T", "e_f"],
    "A": ["e_rho", "e_u", "e_T", "e_f"],
    "B": ["e_rho", "e_u", "e_T", "e_f"],
    "C": ["e_rho", "e_u", "e_T"],
}
METRIC_NAMES = ["e_rho", "e_u", "e_T", "e_f"]
COLORS = {"none": "tab:gray", "aot": "tab:red", "naive_kl": "tab:purple",
          "A": "tab:blue", "B": "tab:orange", "C": "tab:green"}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _window_mean(t: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    mask = (t >= lo) & (t <= hi)
    if not mask.any():
        return float("nan")
    return float(np.nanmean(y[mask]))


def _series_for(assim_log: dict, name: str) -> np.ndarray | None:
    if name in assim_log and len(assim_log[name]) > 0:
        return np.asarray(assim_log[name])
    extras = assim_log.get("extras", {})
    if name in extras:
        return np.asarray(extras[name])
    return None


def _judge(t: np.ndarray, assim_log: dict, formulation: str) -> dict:
    row = {"formulation": formulation}
    metrics = METRICS_FOR_JUDGEMENT[formulation]
    all_recover = True
    failing = []
    for name in ["e_rho", "e_u", "e_T", "e_f"]:
        ser = _series_for(assim_log, name)
        if ser is None:
            row[f"{name}_early"] = float("nan")
            row[f"{name}_late"] = float("nan")
            if name in metrics:
                all_recover = False
                failing.append(f"{name}(missing)")
            continue
        early = _window_mean(t, ser, *EARLY_WINDOW)
        late = _window_mean(t, ser, *LATE_WINDOW)
        row[f"{name}_early"] = early
        row[f"{name}_late"] = late
        if name in metrics:
            recovered = (np.isfinite(early) and np.isfinite(late)
                          and early > 0 and late <= early / DECADE)
            if not recovered:
                all_recover = False
                failing.append(name)
    row["judged_metrics"] = "+".join(metrics)
    row["recovered"] = "yes" if all_recover else "no"
    row["failing"] = ",".join(failing) if failing else ""
    return row


def _plot_velocity_kde(
    out_dir: Path, finals: dict[str, dict], v_min: float, v_max: float,
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(v_min, v_max, 121)
    # Truth: pull from any formulation's final_truth (they're identical seeds).
    truth = finals[FORMULATIONS[0]]["final_truth"]
    h, edges = np.histogram(truth["v"], bins=bins, weights=truth["w"], density=True)
    ax.plot(0.5 * (edges[:-1] + edges[1:]), h, color="black",
            linewidth=2, label="truth")
    colors = COLORS
    for f in FORMULATIONS:
        a = finals[f]["final_assim"]
        h, edges = np.histogram(a["v"], bins=bins, weights=a["w"], density=True)
        ax.plot(0.5 * (edges[:-1] + edges[1:]), h, color=colors[f],
                linewidth=1.2, label=f)
    ax.set_xlabel("v")
    ax.set_ylabel("f(v) at final time")
    ax.set_title("Velocity marginal — truth vs formulations")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / "velocity_kde.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_phase_space(
    out_dir: Path, finals: dict[str, dict], L: float,
    v_min: float, v_max: float,
) -> Path:
    panels = ["truth"] + FORMULATIONS
    fig, axes = plt.subplots(
        1, len(panels), figsize=(3.6 * len(panels), 4), sharey=True,
    )
    Nx, Nv = 64, 64
    xb = np.linspace(0, L, Nx + 1)
    vb = np.linspace(v_min, v_max, Nv + 1)
    for ax, label in zip(axes, panels):
        if label == "truth":
            d = finals[FORMULATIONS[0]]["final_truth"]
        else:
            d = finals[label]["final_assim"]
        H, _, _ = np.histogram2d(d["x"], d["v"], bins=[xb, vb], weights=d["w"])
        ax.imshow(H.T, origin="lower", aspect="auto",
                  extent=[0, L, v_min, v_max], cmap="viridis")
        ax.set_title(label)
        ax.set_xlabel("x")
    axes[0].set_ylabel("v")
    fig.suptitle("Phase-space density f(x,v) at final time")
    fig.tight_layout()
    out = out_dir / "phase_space.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot(out_dir: Path, results: dict[str, dict]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True)
    titles = {"e_rho": r"$e_\rho$  (density)",
              "e_u":   r"$e_u$  (bulk velocity)",
              "e_T":   r"$e_T$  (temperature)",
              "e_f":   r"$e_f$  (phase-space density)"}
    panels = [("e_rho", axes[0, 0]), ("e_u", axes[0, 1]),
              ("e_T", axes[1, 0]), ("e_f", axes[1, 1])]
    colors = COLORS
    for name, ax in panels:
        for f, res in results.items():
            t = res["t"]
            ser = _series_for(res["assim_log"], name)
            if ser is None or len(ser) == 0:
                continue
            ax.semilogy(t, np.maximum(ser, 1e-16), label=f, color=colors[f],
                        linewidth=2.0)
        ax.set_title(titles[name], fontsize=14)
        ax.set_xlabel("t", fontsize=12)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=11)
        ax.tick_params(labelsize=11)
    fig.tight_layout()
    out = out_dir / "error_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _write_summary_md(
    out_dir: Path, name: str, sha: str, judgement_rows: list[dict],
    overrides: dict, fig_path: Path,
    kde_path: Path | None = None, ps_path: Path | None = None,
) -> None:
    lines = []
    lines.append(f"# {name}")
    lines.append("")
    lines.append(f"git SHA: `{sha}`")
    if overrides:
        lines.append(f"Overrides: `{overrides}`")
    lines.append("")
    lines.append("## Recover judgement")
    lines.append("")
    lines.append("Each error series is judged on at least one decade drop")
    lines.append(f"between t∈{EARLY_WINDOW} and t∈{LATE_WINDOW}.")
    lines.append("Formulation C is judged on (e_rho, e_u, e_T) only — its")
    lines.append("invariant is pi_obs, not f_truth, so e_f is informational.")
    lines.append("")
    header = ["formulation", "recovered", "failing", "e_rho_early→late",
              "e_u_early→late", "e_T_early→late", "e_f_early→late"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for r in judgement_rows:
        cells = [
            r["formulation"],
            r["recovered"],
            r.get("failing", ""),
            f"{r['e_rho_early']:.3g} → {r['e_rho_late']:.3g}",
            f"{r['e_u_early']:.3g} → {r['e_u_late']:.3g}",
            f"{r['e_T_early']:.3g} → {r['e_T_late']:.3g}",
            f"{r['e_f_early']:.3g} → {r['e_f_late']:.3g}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Improvement vs no-nudging (late window)")
    lines.append("")
    lines.append("Per-channel ratio = e_x(late, formulation) / e_x(late, `none`).")
    lines.append("**< 1 means better than no nudging; e.g. 0.10 = 10x reduction.**")
    lines.append("This replaces the old binary decade flag, which read as uniform")
    lines.append("failure whenever a channel (typically e_rho) sat at the particle-")
    lines.append("noise floor and could not drop a full decade regardless of method.")
    lines.append("")
    by_f = {r["formulation"]: r for r in judgement_rows}
    none_row = by_f.get("none", {})
    ihdr = ["formulation"] + [f"{m} ratio" for m in METRIC_NAMES]
    lines.append("| " + " | ".join(ihdr) + " |")
    lines.append("| " + " | ".join("---" for _ in ihdr) + " |")
    for r in judgement_rows:
        cells = [r["formulation"]]
        for m in METRIC_NAMES:
            late = r.get(f"{m}_late", float("nan"))
            base = none_row.get(f"{m}_late", float("nan"))
            ratio = (late / base) if (base and base == base and base > 0) else float("nan")
            cells.append("1.00 (ref)" if r["formulation"] == "none"
                         else f"{ratio:.2g}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Error curves")
    lines.append("")
    lines.append(f"![error curves]({fig_path.name})")
    lines.append("")
    if kde_path is not None:
        lines.append("## Velocity marginal (final time)")
        lines.append("")
        lines.append(f"![velocity KDE]({kde_path.name})")
        lines.append("")
    if ps_path is not None:
        lines.append("## Phase-space density (final time)")
        lines.append("")
        lines.append(f"![phase space]({ps_path.name})")
        lines.append("")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--np-override", type=int, default=None,
                    help="Override pic.Np (smoke runs).")
    ap.add_argument("--n-steps-override", type=int, default=None,
                    help="Override pic.n_steps (smoke runs).")
    ap.add_argument("--name-suffix", default="",
                    help="Append to outputs subdir (e.g. _smoke).")
    ap.add_argument("--nu-override", type=float, default=None,
                    help="Override collision.nu (sweep runs).")
    ap.add_argument("--E0-override", type=float, default=None,
                    help="Override driver.E0 (sweep runs).")
    ap.add_argument("--omega-d-override", type=float, default=None,
                    help="Override driver.omega_d (sweep runs).")
    ap.add_argument("--formulations", default=None,
                    help="Comma-separated formulation set (e.g. none,A,C,aot,naive_kl).")
    ap.add_argument("--every-q", type=int, default=None,
                    help="Override moment_observation.every_q (sparse-in-time).")
    ap.add_argument("--sigma-u", type=float, default=None,
                    help="Observation noise on u (sets kind=noisy).")
    ap.add_argument("--sigma-T", type=float, default=None,
                    help="Observation noise on T (sets kind=noisy).")
    ap.add_argument("--sigma-rho", type=float, default=None,
                    help="Observation noise on rho (sets kind=noisy).")
    ap.add_argument("--assim-nu", type=float, default=None,
                    help="Imperfect model: assim collision frequency (truth keeps collision.nu).")
    ap.add_argument("--assim-kind", default=None,
                    help="Imperfect model: assim collision operator (bgk/lb).")
    ap.add_argument("--seed", type=int, default=None,
                    help="Override cfg.seed (multi-seed error-bar runs).")
    ap.add_argument("--truth-np-override", type=int, default=None,
                    help="Sample the truth ensemble with this many particles "
                         "(higher-resolution / independently-sampled truth).")
    args = ap.parse_args()

    global FORMULATIONS
    if args.formulations is not None:
        FORMULATIONS = [s.strip() for s in args.formulations.split(",") if s.strip()]

    cfg = load_moment(args.config)
    overrides = {}
    if args.np_override is not None:
        cfg.pic.Np = args.np_override
        overrides["Np"] = args.np_override
    if args.n_steps_override is not None:
        cfg.pic.n_steps = args.n_steps_override
        overrides["n_steps"] = args.n_steps_override
    if args.nu_override is not None:
        cfg.collision.nu = args.nu_override
        overrides["nu"] = args.nu_override
    if args.E0_override is not None:
        cfg.driver.E0 = args.E0_override
        overrides["E0"] = args.E0_override
    if args.omega_d_override is not None:
        cfg.driver.omega_d = args.omega_d_override
        overrides["omega_d"] = args.omega_d_override
    if args.every_q is not None:
        cfg.moment_observation.every_q = args.every_q
        overrides["every_q"] = args.every_q
    if any(s is not None for s in (args.sigma_u, args.sigma_T, args.sigma_rho)):
        cfg.moment_observation.kind = "noisy"
        if args.sigma_u is not None:
            cfg.moment_observation.sigma_u = args.sigma_u
            overrides["sigma_u"] = args.sigma_u
        if args.sigma_T is not None:
            cfg.moment_observation.sigma_T = args.sigma_T
            overrides["sigma_T"] = args.sigma_T
        if args.sigma_rho is not None:
            cfg.moment_observation.sigma_rho = args.sigma_rho
            overrides["sigma_rho"] = args.sigma_rho
    if args.assim_nu is not None:
        cfg.collision.assim_nu = args.assim_nu
        overrides["assim_nu"] = args.assim_nu
    if args.assim_kind is not None:
        cfg.collision.assim_kind = args.assim_kind
        overrides["assim_kind"] = args.assim_kind
    if args.seed is not None:
        cfg.seed = args.seed
        cfg.moment_observation.rng_seed = args.seed
        overrides["seed"] = args.seed
    if args.truth_np_override is not None:
        overrides["truth_np"] = args.truth_np_override

    out_root = Path(cfg.outputs_dir)
    out_dir = out_root / (cfg.name + args.name_suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[recover-check] outputs: {out_dir}")

    results: dict[str, dict] = {}
    for formulation in FORMULATIONS:
        cfg.moment_nudge.formulation = formulation  # type: ignore[assignment]
        print(f"[recover-check] running formulation={formulation} ...", flush=True)
        out = run_moments(cfg, truth_np=args.truth_np_override)
        npz_path = out_dir / f"{formulation}.npz"
        np.savez(
            npz_path,
            t=out.t,
            assim_log_e_phi=out.assim_log["e_phi"],
            assim_log_e_rho=out.assim_log["e_rho"],
            assim_log_e_f=out.assim_log["e_f"],
            assim_log_energy=out.assim_log["energy"],
            assim_log_modes=out.assim_log["modes"],
            assim_log_e_u=out.assim_log.get("extras", {}).get(
                "e_u", np.zeros(0)),
            assim_log_e_T=out.assim_log.get("extras", {}).get(
                "e_T", np.zeros(0)),
            truth_log_energy=out.truth_log["energy"],
            truth_log_modes=out.truth_log["modes"],
            final_assim_x=out.final_assim["x"],
            final_assim_v=out.final_assim["v"],
            final_assim_w=out.final_assim["w"],
            final_truth_x=out.final_truth["x"],
            final_truth_v=out.final_truth["v"],
            final_truth_w=out.final_truth["w"],
        )
        print(f"[recover-check]   wrote {npz_path}", flush=True)
        results[formulation] = {
            "t": out.t,
            "assim_log": out.assim_log,
            "truth_log": out.truth_log,
            "final_assim": out.final_assim,
            "final_truth": out.final_truth,
        }

    # Judgement.
    judgement_rows = [
        _judge(results[f]["t"], results[f]["assim_log"], f)
        for f in FORMULATIONS
    ]
    csv_path = out_dir / "recover_summary.csv"
    with open(csv_path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(judgement_rows[0].keys()))
        writer.writeheader()
        writer.writerows(judgement_rows)
    print(f"[recover-check] wrote {csv_path}")

    fig_path = _plot(out_dir, results)
    print(f"[recover-check] wrote {fig_path}")

    kde_path = _plot_velocity_kde(out_dir, results,
                                   cfg.domain.v_min, cfg.domain.v_max)
    print(f"[recover-check] wrote {kde_path}")
    ps_path = _plot_phase_space(out_dir, results, cfg.domain.L,
                                 cfg.domain.v_min, cfg.domain.v_max)
    print(f"[recover-check] wrote {ps_path}")

    _write_summary_md(out_dir, cfg.name + args.name_suffix, _git_sha(),
                       judgement_rows, overrides, fig_path,
                       kde_path=kde_path, ps_path=ps_path)
    print(f"[recover-check] wrote {out_dir / 'SUMMARY.md'}")
    print("[recover-check] done.")


if __name__ == "__main__":
    main()
