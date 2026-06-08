"""Nudge-then-free comparison: does the assimilated state self-sustain?

For each formulation in {A, B, C} (and the unnudged baseline "none"),
runs two simulations in parallel:

  - "always" : nudging on for the full duration (the standard recover-check)
  - "free"   : nudging on for steps n < nudge_until_step, then off

Compares the post-cutoff error trajectories. The interpretation:

  - free stays near "always"  ⇒  the nudged state captured the
                                  underlying dynamics; Vlasov+BGK
                                  alone preserves the assimilation gain
  - free relapses toward "none" ⇒ the nudging was superficial; without
                                  continued forcing the assim drifts back

Outputs to <outputs_dir>/<name>_nudge_until<frac>/:
    {formulation}_{mode}.npz                         8 files
    error_curves.png                                 4 panels, 7 lines + cutoff line
    velocity_kde.png, phase_space.png                final-time figures
    SUMMARY.md                                        relapse ratios per formulation

Usage:
    python scripts/run_nudge_then_free.py \\
        --config configs/test_moment_obs_ABC_driven_BGK_truly_resonant.yaml \\
        --nudge-fraction 0.5
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


FORMULATIONS = ["none", "A", "B", "C"]
MODES = ["always", "free"]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _series_for(assim_log: dict, name: str) -> np.ndarray | None:
    if name in assim_log and len(assim_log[name]) > 0:
        return np.asarray(assim_log[name])
    return np.asarray(assim_log.get("extras", {}).get(name, []))


def _post_cutoff_mean(t: np.ndarray, y: np.ndarray, t_cut: float,
                      t_end: float) -> float:
    mask = (t >= t_cut) & (t <= t_end)
    if not mask.any():
        return float("nan")
    return float(np.nanmean(y[mask]))


def _plot_errors(out_dir: Path, results: dict, t_cut: float) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    titles = {"e_rho": r"$e_\rho$", "e_u": r"$e_u$",
              "e_T": r"$e_T$", "e_f": r"$e_f$"}
    panels = [("e_rho", axes[0, 0]), ("e_u", axes[0, 1]),
              ("e_T", axes[1, 0]), ("e_f", axes[1, 1])]
    colors = {"none": "tab:gray", "A": "tab:blue",
              "B": "tab:orange", "C": "tab:green"}
    linestyles = {"always": "-", "free": "--"}
    for name, ax in panels:
        for f in FORMULATIONS:
            for m in MODES:
                if f == "none" and m == "free":
                    continue  # 'none' has nothing to free
                key = (f, m)
                if key not in results:
                    continue
                t = results[key]["t"]
                ser = _series_for(results[key]["assim_log"], name)
                if ser is None or len(ser) == 0:
                    continue
                label = f"{f} ({m})" if f != "none" else "none"
                ax.semilogy(t, np.maximum(ser, 1e-16), label=label,
                            color=colors[f], linestyle=linestyles[m],
                            linewidth=1.4)
        ax.axvline(t_cut, color="red", linestyle=":", linewidth=1,
                   alpha=0.7, label=f"nudge off (t={t_cut:.1f})")
        ax.set_title(titles[name])
        ax.set_xlabel("t")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    out = out_dir / "error_curves.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_velocity_kde(out_dir: Path, results: dict,
                       v_min: float, v_max: float) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(v_min, v_max, 121)
    truth = results[("A", "always")]["final_truth"]
    h, edges = np.histogram(truth["v"], bins=bins, weights=truth["w"],
                            density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(centers, h, color="black", linewidth=2.5, label="truth")
    colors = {"none": "tab:gray", "A": "tab:blue",
              "B": "tab:orange", "C": "tab:green"}
    linestyles = {"always": "-", "free": "--"}
    for f in FORMULATIONS:
        for m in MODES:
            if f == "none" and m == "free":
                continue
            key = (f, m)
            if key not in results:
                continue
            a = results[key]["final_assim"]
            h, _ = np.histogram(a["v"], bins=bins, weights=a["w"],
                                density=True)
            label = f"{f} ({m})" if f != "none" else "none"
            ax.plot(centers, h, color=colors[f], linestyle=linestyles[m],
                    linewidth=1.2, label=label)
    ax.set_xlabel("v")
    ax.set_ylabel("f(v) at final time")
    ax.set_title("Velocity marginal — truth vs (always vs nudge-then-free)")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / "velocity_kde.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_phase_space(out_dir: Path, results: dict, L: float,
                      v_min: float, v_max: float) -> Path:
    # 2 rows (modes), 5 cols (truth + 4 formulations).
    # Rows: always, free; truth column shared (just shown twice).
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey=True)
    Nx, Nv = 64, 64
    xb = np.linspace(0, L, Nx + 1)
    vb = np.linspace(v_min, v_max, Nv + 1)
    truth = results[("A", "always")]["final_truth"]
    for row, mode in enumerate(MODES):
        # truth panel
        H, _, _ = np.histogram2d(truth["x"], truth["v"], bins=[xb, vb],
                                  weights=truth["w"])
        axes[row, 0].imshow(H.T, origin="lower", aspect="auto",
                             extent=[0, L, v_min, v_max], cmap="viridis")
        axes[row, 0].set_title(f"truth")
        axes[row, 0].set_ylabel(f"{mode}\n\nv")
        for col, f in enumerate(FORMULATIONS, start=1):
            if f == "none" and mode == "free":
                axes[row, col].set_visible(False)
                continue
            key = (f, mode)
            if key not in results:
                continue
            d = results[key]["final_assim"]
            H, _, _ = np.histogram2d(d["x"], d["v"], bins=[xb, vb],
                                      weights=d["w"])
            axes[row, col].imshow(H.T, origin="lower", aspect="auto",
                                   extent=[0, L, v_min, v_max], cmap="viridis")
            axes[row, col].set_title(f"{f} ({mode})")
            axes[row, col].set_xlabel("x")
    fig.suptitle("Phase-space density f(x,v) at final time")
    fig.tight_layout()
    out = out_dir / "phase_space.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _judge_relapse(results: dict, t_cut: float, t_end: float) -> list[dict]:
    """For each formulation, compute mean error in (t_cut, t_end] for
    always/free/none, plus a 'relapse fraction' = (free - always) / (none - always)
    on each metric. 0 = no relapse, 1 = full relapse to baseline.
    """
    rows = []
    for f in FORMULATIONS:
        if f == "none":
            continue
        row = {"formulation": f}
        for metric in ["e_rho", "e_u", "e_T", "e_f"]:
            t = results[(f, "always")]["t"]
            e_a = _post_cutoff_mean(
                t, _series_for(results[(f, "always")]["assim_log"], metric),
                t_cut, t_end)
            e_free = _post_cutoff_mean(
                t, _series_for(results[(f, "free")]["assim_log"], metric),
                t_cut, t_end)
            t_n = results[("none", "always")]["t"]
            e_none = _post_cutoff_mean(
                t_n, _series_for(results[("none", "always")]["assim_log"], metric),
                t_cut, t_end)
            row[f"{metric}_always"] = e_a
            row[f"{metric}_free"] = e_free
            row[f"{metric}_none"] = e_none
            denom = e_none - e_a
            if denom > 0 and np.isfinite(e_free) and np.isfinite(e_a):
                row[f"{metric}_relapse"] = float((e_free - e_a) / denom)
            else:
                row[f"{metric}_relapse"] = float("nan")
        rows.append(row)
    return rows


def _write_summary_md(out_dir: Path, name: str, sha: str, t_cut: float,
                      t_end: float, judgement_rows: list[dict],
                      overrides: dict) -> None:
    lines = []
    lines.append(f"# {name}")
    lines.append("")
    lines.append(f"git SHA: `{sha}`")
    if overrides:
        lines.append(f"Overrides: `{overrides}`")
    lines.append(f"Nudging off at t = {t_cut:.2f}; post-cutoff window "
                 f"({t_cut:.2f}, {t_end:.2f}].")
    lines.append("")
    lines.append("## Relapse table")
    lines.append("")
    lines.append("Relapse fraction = (free_post_mean - always_post_mean) /")
    lines.append("(none_post_mean - always_post_mean). 0 = nudged state is")
    lines.append("self-sustaining; 1 = full relapse to the no-nudge baseline.")
    lines.append("")
    header = ["formulation", "metric", "always", "free", "none", "relapse"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for r in judgement_rows:
        for metric in ["e_rho", "e_u", "e_T", "e_f"]:
            cells = [
                r["formulation"], metric,
                f"{r[f'{metric}_always']:.3g}",
                f"{r[f'{metric}_free']:.3g}",
                f"{r[f'{metric}_none']:.3g}",
                f"{r[f'{metric}_relapse']:.3g}",
            ]
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Error curves")
    lines.append("")
    lines.append("![error curves](error_curves.png)")
    lines.append("")
    lines.append("## Velocity marginal (final time)")
    lines.append("")
    lines.append("![velocity KDE](velocity_kde.png)")
    lines.append("")
    lines.append("## Phase-space density (final time)")
    lines.append("")
    lines.append("![phase space](phase_space.png)")
    lines.append("")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--nudge-fraction", type=float, default=0.5,
                    help="Fraction of n_steps for which nudging is on.")
    ap.add_argument("--np-override", type=int, default=None)
    ap.add_argument("--n-steps-override", type=int, default=None)
    ap.add_argument("--nu-override", type=float, default=None)
    ap.add_argument("--name-suffix", default="")
    args = ap.parse_args()

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

    nudge_until = int(round(args.nudge_fraction * cfg.pic.n_steps))
    t_cut = nudge_until * cfg.pic.dt
    t_end = cfg.pic.n_steps * cfg.pic.dt

    suffix = (args.name_suffix
              or f"_nudge_until{int(round(args.nudge_fraction * 100))}")
    out_dir = Path(cfg.outputs_dir) / (cfg.name + suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[nudge-then-free] outputs: {out_dir}")
    print(f"[nudge-then-free] cutoff: step {nudge_until} (t={t_cut:.2f}), "
          f"end t={t_end:.2f}")

    results: dict = {}
    for formulation in FORMULATIONS:
        for mode in MODES:
            if formulation == "none" and mode == "free":
                continue  # identical to "always" for none
            cfg.moment_nudge.formulation = formulation  # type: ignore[assignment]
            cutoff = nudge_until if mode == "free" else None
            print(f"[nudge-then-free] running {formulation}/{mode} ...",
                  flush=True)
            out = run_moments(cfg, nudge_until_step=cutoff)
            npz = out_dir / f"{formulation}_{mode}.npz"
            np.savez(
                npz, t=out.t,
                assim_log_e_phi=out.assim_log["e_phi"],
                assim_log_e_rho=out.assim_log["e_rho"],
                assim_log_e_f=out.assim_log["e_f"],
                assim_log_energy=out.assim_log["energy"],
                assim_log_e_u=out.assim_log.get("extras", {}).get(
                    "e_u", np.zeros(0)),
                assim_log_e_T=out.assim_log.get("extras", {}).get(
                    "e_T", np.zeros(0)),
                final_assim_x=out.final_assim["x"],
                final_assim_v=out.final_assim["v"],
                final_assim_w=out.final_assim["w"],
                final_truth_x=out.final_truth["x"],
                final_truth_v=out.final_truth["v"],
                final_truth_w=out.final_truth["w"],
            )
            print(f"[nudge-then-free]   wrote {npz}", flush=True)
            results[(formulation, mode)] = {
                "t": out.t,
                "assim_log": out.assim_log,
                "final_assim": out.final_assim,
                "final_truth": out.final_truth,
            }

    fig_path = _plot_errors(out_dir, results, t_cut)
    print(f"[nudge-then-free] wrote {fig_path}")
    kde_path = _plot_velocity_kde(out_dir, results,
                                   cfg.domain.v_min, cfg.domain.v_max)
    print(f"[nudge-then-free] wrote {kde_path}")
    ps_path = _plot_phase_space(out_dir, results, cfg.domain.L,
                                 cfg.domain.v_min, cfg.domain.v_max)
    print(f"[nudge-then-free] wrote {ps_path}")

    judgement_rows = _judge_relapse(results, t_cut, t_end)
    csv_path = out_dir / "relapse_summary.csv"
    if judgement_rows:
        with open(csv_path, "w", newline="") as fp:
            writer = csv.DictWriter(fp,
                                     fieldnames=list(judgement_rows[0].keys()))
            writer.writeheader()
            writer.writerows(judgement_rows)
        print(f"[nudge-then-free] wrote {csv_path}")

    _write_summary_md(out_dir, cfg.name + suffix, _git_sha(),
                      t_cut, t_end, judgement_rows, overrides)
    print(f"[nudge-then-free] wrote {out_dir / 'SUMMARY.md'}")
    print("[nudge-then-free] done.")


if __name__ == "__main__":
    main()
