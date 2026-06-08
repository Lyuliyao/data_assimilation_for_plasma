"""Compare velocity-nudging, position-nudging, and a no-nudge baseline.

Runs the same config three times (variant = none, velocity, position),
stitches the diagnostic time series into a single plot, and writes a CSV
summary. The no-nudge baseline is the Test-0 identifiability reference:
for truth/assim ICs with the same rho_0 but different velocity marginals,
it shows how much of e_phi / e_f is there without any assimilation.

Usage:
    python scripts/compare_velocity_vs_position.py --config configs/test0_identifiability.yaml \
        --gamma 1.0 --plot
"""
from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import numpy as np

from mfda.assimilation import run
from mfda.config import load


VARIANTS = ("none", "velocity", "position")
COLORS = {"none": "C2", "velocity": "C0", "position": "C1"}
LABELS = {"none": "no-nudge", "velocity": "velocity-nudge", "position": "position-nudge"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--plot", action="store_true",
                    help="write comparison PNG to results/<name>/compare.png")
    args = ap.parse_args()

    cfg = load(args.config)
    # The new NudgeCfg stores gamma per-channel; fall back to whatever the
    # first enabled channel's gamma was so we keep the historical "gamma at
    # which we ran the comparison" label in the CSV/plot.
    def _channel_gamma(nudge) -> float:
        for ch in (nudge.position_snapshot, nudge.velocity_snapshot,
                   nudge.velocity_dtobs):
            if ch.enabled:
                return ch.gamma
        return 1.0
    gamma = args.gamma if args.gamma is not None else _channel_gamma(cfg.nudge)

    outs: dict[str, object] = {}
    for v in VARIANTS:
        cfg_v = copy.deepcopy(cfg)
        cfg_v.nudge.set_legacy_variant(v, gamma)
        print(f"[compare] running variant={v} (gamma={gamma})...")
        outs[v] = run(cfg_v)

    outdir = Path(cfg.outputs_dir) / cfg.name
    outdir.mkdir(parents=True, exist_ok=True)

    # CSV summary: final-time scalars per variant.
    with (outdir / "compare_summary.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["variant", "gamma", "e_phi_final", "e_rho_final", "e_f_final",
                     "energy_final", "mode1_final", "mode1_truth_final"])
        for name in VARIANTS:
            out = outs[name]
            modes_a = np.asarray(out.assim_log["modes"])
            modes_t = np.asarray(out.truth_log["modes"])
            mode1_a = float(modes_a[-1, 1]) if modes_a.ndim == 2 and modes_a.shape[1] > 1 else float("nan")
            mode1_t = float(modes_t[-1, 1]) if modes_t.ndim == 2 and modes_t.shape[1] > 1 else float("nan")
            wr.writerow([
                name, gamma,
                out.assim_log["e_phi"][-1],
                out.assim_log["e_rho"][-1],
                out.assim_log["e_f"][-1],
                out.assim_log["energy"][-1],
                mode1_a, mode1_t,
            ])
    print(f"wrote {outdir / 'compare_summary.csv'}")

    if args.plot:
        import matplotlib.pyplot as plt
        t = outs["velocity"].t
        modes_truth = np.asarray(outs["velocity"].truth_log["modes"])

        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)

        # e_phi
        ax = axes[0, 0]
        for v in VARIANTS:
            ax.semilogy(t, outs[v].assim_log["e_phi"], color=COLORS[v], label=LABELS[v])
        ax.set_ylabel(r"$\|\phi - \phi^\star\|_{L^2}$")
        ax.legend(fontsize=8)

        # e_rho
        ax = axes[0, 1]
        for v in VARIANTS:
            ax.semilogy(t, outs[v].assim_log["e_rho"], color=COLORS[v], label=LABELS[v])
        ax.set_ylabel(r"$\|\rho - \rho^\star\|_{L^2}$")

        # e_f  (phase-space L2 via histogram) — the identifiability diagnostic
        ax = axes[0, 2]
        for v in VARIANTS:
            ax.plot(t, outs[v].assim_log["e_f"], color=COLORS[v], label=LABELS[v])
        ax.set_ylabel(r"$\|f - f^\star\|_{L^2_{x,v}}$ (hist)")

        # electric energy
        ax = axes[1, 0]
        ax.plot(t, np.asarray(outs["velocity"].truth_log["energy"]),
                color="k", lw=1.2, label="truth")
        for v in VARIANTS:
            ax.plot(t, outs[v].assim_log["energy"], color=COLORS[v], label=LABELS[v])
        ax.set_ylabel("Electric energy")
        ax.set_xlabel("t")

        # |E_k| for k=1 (the lowest nonzero mode, typically the dominant Landau mode)
        ax = axes[1, 1]
        if modes_truth.ndim == 2 and modes_truth.shape[1] > 1:
            ax.semilogy(t, modes_truth[:, 1], color="k", lw=1.2, label="truth")
            for v in VARIANTS:
                m = np.asarray(outs[v].assim_log["modes"])
                ax.semilogy(t, m[:, 1], color=COLORS[v], label=LABELS[v])
        ax.set_ylabel(r"$|\hat E_{k=1}|$")
        ax.set_xlabel("t")

        # |E_k| for k=2 (second mode; useful to see harmonic generation / two-stream)
        ax = axes[1, 2]
        if modes_truth.ndim == 2 and modes_truth.shape[1] > 2:
            ax.semilogy(t, modes_truth[:, 2], color="k", lw=1.2, label="truth")
            for v in VARIANTS:
                m = np.asarray(outs[v].assim_log["modes"])
                ax.semilogy(t, m[:, 2], color=COLORS[v], label=LABELS[v])
        ax.set_ylabel(r"$|\hat E_{k=2}|$")
        ax.set_xlabel("t")

        for ax in axes.flat:
            ax.grid(alpha=0.3)
        fig.suptitle(f"{cfg.name}: velocity vs position nudging vs no-nudge "
                     f"(gamma={gamma})")
        fig.tight_layout()
        fig.savefig(outdir / "compare.png", dpi=150)
        print(f"wrote {outdir / 'compare.png'}")


if __name__ == "__main__":
    main()
