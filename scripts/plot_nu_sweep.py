"""Aggregate nu-sweep recover_summary.csv files into a 4-panel error-vs-nu plot.

Reads results/<sweep_name_template>_nu<...>/recover_summary.csv for the
nu values supplied, and writes a single figure with one panel per metric
(e_rho, e_u, e_T, e_f), one curve per formulation (none/A/B/C).

Usage:
    python scripts/plot_nu_sweep.py \\
        --base-name test_moment_obs_ABC_driven_BGK_truly_resonant \\
        --nu-values 0.5 0.2 0.05 0.02 0.005 \\
        --out results/test_moment_obs_ABC_driven_BGK_truly_resonant_nu_sweep
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FORMULATIONS = ["none", "A", "B", "C"]
METRICS = ["e_rho", "e_u", "e_T", "e_f"]
COLORS = {"none": "tab:gray", "A": "tab:blue", "B": "tab:orange", "C": "tab:green"}


def _nu_dir_name(base: str, nu: float) -> str:
    nu_str = f"{nu:.6g}".replace(".", "p")
    return f"{base}_nu{nu_str}"


def _read_summary(path: Path) -> dict[str, dict]:
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            rows[row["formulation"]] = row
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-name", required=True)
    ap.add_argument("--nu-values", type=float, nargs="+", required=True)
    ap.add_argument("--results-dir", type=Path,
                    default=Path("/ocean/projects/mth210003p/lyuliyao/DA/plasma/results"))
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory.")
    args = ap.parse_args()

    nus = sorted(args.nu_values)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # Collect: data[formulation][metric] = list aligned with nus.
    data: dict[str, dict[str, list[float]]] = {
        f: {m: [] for m in METRICS} for f in FORMULATIONS
    }
    for nu in nus:
        d = args.results_dir / _nu_dir_name(args.base_name, nu)
        csv_path = d / "recover_summary.csv"
        if not csv_path.exists():
            print(f"[plot_nu_sweep] missing {csv_path}; skipping nu={nu}")
            for f in FORMULATIONS:
                for m in METRICS:
                    data[f][m].append(float("nan"))
            continue
        rows = _read_summary(csv_path)
        for f in FORMULATIONS:
            row = rows.get(f, {})
            for m in METRICS:
                key = f"{m}_late"
                v = row.get(key, "nan")
                try:
                    data[f][m].append(float(v))
                except (TypeError, ValueError):
                    data[f][m].append(float("nan"))

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True)
    titles = {"e_rho": r"$e_\rho$  (density)",
              "e_u":   r"$e_u$  (bulk velocity)",
              "e_T":   r"$e_T$  (temperature)",
              "e_f":   r"$e_f$  (phase-space density)"}
    panels = [("e_rho", axes[0, 0]), ("e_u", axes[0, 1]),
              ("e_T", axes[1, 0]), ("e_f", axes[1, 1])]
    for metric, ax in panels:
        for f in FORMULATIONS:
            y = data[f][metric]
            ax.loglog(nus, y, "o-", color=COLORS[f], label=f,
                      linewidth=2.0, markersize=8)
        ax.set_title(titles[metric], fontsize=14)
        ax.set_xlabel(r"$\nu$ (collision frequency)", fontsize=12)
        ax.set_ylabel(f"late mean of {metric}", fontsize=12)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=11)
        ax.tick_params(labelsize=11)
    fig.suptitle(f"Nu sweep: late error vs collision frequency "
                 f"({args.base_name})", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig_path = out / "nu_sweep_curves.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[plot_nu_sweep] wrote {fig_path}")

    # Tabulate.
    table_path = out / "nu_sweep_table.csv"
    with open(table_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["nu"] + [f"{m}_{f}" for m in METRICS for f in FORMULATIONS])
        for i, nu in enumerate(nus):
            row = [nu]
            for m in METRICS:
                for f in FORMULATIONS:
                    row.append(data[f][m][i])
            w.writerow(row)
    print(f"[plot_nu_sweep] wrote {table_path}")

    # SUMMARY.md
    md = []
    md.append(f"# {args.base_name} — nu sweep")
    md.append("")
    md.append(f"nu values: {nus}")
    md.append("")
    md.append("## Late-time error vs nu")
    md.append("")
    md.append("![nu sweep](nu_sweep_curves.png)")
    md.append("")
    md.append("## Table (late means)")
    md.append("")
    header = ["nu"] + [f"{m}({f})" for m in METRICS for f in FORMULATIONS]
    md.append("| " + " | ".join(header) + " |")
    md.append("| " + " | ".join("---" for _ in header) + " |")
    for i, nu in enumerate(nus):
        cells = [f"{nu:g}"]
        for m in METRICS:
            for f in FORMULATIONS:
                v = data[f][m][i]
                cells.append(f"{v:.3g}" if np.isfinite(v) else "nan")
        md.append("| " + " | ".join(cells) + " |")
    md.append("")
    (out / "SUMMARY.md").write_text("\n".join(md))
    print(f"[plot_nu_sweep] wrote {out / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
