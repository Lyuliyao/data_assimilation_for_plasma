"""Plot the spatially-integrated kinetic stress M_total(t) = integral v^2 f
over (x, v), comparing truth, combined-nudge assim, and no-nudge assim.

Reads results/<combined>/snapshots.npz and results/<nonudge>/snapshots.npz
(both produced by scripts/run_with_snapshots.py with the M_total logging
extension). Truth M_total is taken from the combined run's truth_log
(it's deterministic from seed; the no-nudge truth_log should match).

Usage:
    python scripts/plot_mtotal_timeseries.py \
        --combined-config configs/test1_temperature_combined5_best.yaml \
        --nonudge-config  configs/test1_temperature_combined5_nonudge.yaml \
        --output results/test1_temperature_combined5_best/M_total_timeseries.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mfda.config import load


def _load_log(snapshots_path: Path) -> dict:
    data = np.load(snapshots_path)
    out = {"t": data["t"]}
    for k in data.files:
        if k.startswith("assim_log_extra_"):
            out[f"assim_extra_{k[len('assim_log_extra_'):]}"] = data[k]
        elif k.startswith("truth_log_extra_"):
            out[f"truth_extra_{k[len('truth_log_extra_'):]}"] = data[k]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined-config", required=True,
                    help="Config for the run with combined nudge enabled.")
    ap.add_argument("--nonudge-config", required=True,
                    help="Config for the run with all channels disabled.")
    ap.add_argument("--output", default=None,
                    help="Output PNG path. Default: combined config's results dir.")
    args = ap.parse_args()

    cfg_c = load(args.combined_config)
    cfg_n = load(args.nonudge_config)
    snaps_c = Path(cfg_c.outputs_dir) / cfg_c.name / "snapshots.npz"
    snaps_n = Path(cfg_n.outputs_dir) / cfg_n.name / "snapshots.npz"

    if not snaps_c.exists():
        raise SystemExit(f"missing combined snapshots: {snaps_c}")
    if not snaps_n.exists():
        raise SystemExit(f"missing no-nudge snapshots: {snaps_n}")

    log_c = _load_log(snaps_c)
    log_n = _load_log(snaps_n)

    if "truth_extra_M_total" not in log_c:
        raise SystemExit(
            "M_total not found in combined snapshots — the simulation must "
            "have been run after the M_total logging was added to "
            "mfda.assimilation.run()."
        )

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(log_c["t"], log_c["truth_extra_M_total"],
            color="k", lw=1.6, label="truth")
    ax.plot(log_c["t"], log_c["assim_extra_M_total"],
            color="C3", lw=1.4, label="assim, combined nudge")
    ax.plot(log_n["t"], log_n["assim_extra_M_total"],
            color="C2", lw=1.4, ls="--", label="assim, no nudge")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$M_{\rm total}(t) = \iint v^2 f(x, v, t)\,dx\,dv$")
    ax.set_title(f"{cfg_c.name}: kinetic stress integral over time")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = Path(args.output) if args.output else (
        Path(cfg_c.outputs_dir) / cfg_c.name / "M_total_timeseries.png"
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot-mtotal] wrote {out_path}")


if __name__ == "__main__":
    main()
