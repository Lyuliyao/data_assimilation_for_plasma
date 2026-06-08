"""Overlay diagnostics from the four assim modes (none / phi / phi+dphi /
phi+dphi+d2phi) on the position_mismatch test.

Reads:
    results/<case>/<mode>/assim_diagnostics.npz  per mode

Writes:
    results/<truth>/assim_modes_compare.png  — 2x1: e_phi(t), e_rho(t)
    results/<truth>/assim_modes_M_total.png  — M_total(t) overlay

Usage:
    python scripts/compare_assim_modes.py --truth-dir cases/position_mismatch/results/truth
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODES = ("none", "phi", "phi_dphi", "phi_dphi_d2phi")
LABELS = {
    "none":           "no nudge",
    "phi":            "phi snapshot",
    "phi_dphi":       "phi + d_t phi",
    "phi_dphi_d2phi": "phi + d_t phi + d_t^2 phi",
}
COLORS = {"none": "C2", "phi": "C1", "phi_dphi": "C0", "phi_dphi_d2phi": "C3"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth-dir", required=True)
    args = ap.parse_args()

    truth_dir = Path(args.truth_dir)
    parent = truth_dir.parent
    name = truth_dir.name

    diags: dict[str, dict[str, np.ndarray]] = {}
    for mode in MODES:
        d = parent / mode / "assim_diagnostics.npz"
        if d.exists():
            data = np.load(d)
            diags[mode] = {k: data[k] for k in data.files}
            print(f"[compare] loaded {mode}: e_phi[-1] = {diags[mode]['e_phi'][-1]:.4e}")
        else:
            print(f"[compare] missing {d}")

    # ---- e_phi(t), e_rho(t) ----
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for mode, d in diags.items():
        axes[0].plot(d["t"], d["e_phi"], color=COLORS[mode],
                     label=LABELS[mode], lw=1.2)
        axes[1].plot(d["t"], d["e_rho"], color=COLORS[mode],
                     label=LABELS[mode], lw=1.2)
    axes[0].set_ylabel(r"$e_\phi(t) = \|\phi_{\rm assim} - \phi_{\rm truth}\|_{L^2}$")
    axes[0].set_yscale("log"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)
    axes[1].set_ylabel(r"$e_\rho(t) = \|\rho_{\rm assim} - \rho_{\rm truth}\|_{L^2}$")
    axes[1].set_yscale("log"); axes[1].grid(alpha=0.3)
    axes[1].set_xlabel("t")
    fig.suptitle(f"{name}: assim error vs truth, four nudging modes")
    fig.tight_layout()
    out_a = truth_dir / "assim_modes_compare.png"
    fig.savefig(out_a, dpi=140)
    plt.close(fig)
    print(f"[compare] wrote {out_a}")

    # ---- M_total(t) overlay ----
    fig2, ax = plt.subplots(figsize=(10, 4))
    for mode, d in diags.items():
        if "M_total" not in d:
            continue
        ax.plot(d["t"], d["M_total"], color=COLORS[mode],
                label=LABELS[mode], lw=1.2)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$M_{\rm total}(t) = \int v^2 f(x, v, t)\,dx\,dv$")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.set_title(f"{name}: kinetic stress integral over time")
    fig2.tight_layout()
    out_b = truth_dir / "assim_modes_M_total.png"
    fig2.savefig(out_b, dpi=140)
    plt.close(fig2)
    print(f"[compare] wrote {out_b}")


if __name__ == "__main__":
    main()
