"""Quick β-sweep overlay for the d_t^2 phi channel on current_mismatch.

Reads the assim_diagnostics.npz from each beta variant and overlays
e_phi(t) and M_total(t) so the noise-amplification penalty of larger
beta is visible.

Usage:
    python scripts/plot_beta_sweep.py --truth-dir cases/current_mismatch/results/truth
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth-dir", required=True)
    args = ap.parse_args()

    base = Path(args.truth_dir).parent
    name = Path(args.truth_dir).name
    runs = [
        ("phi+d_tφ  (no d²φ)",   "phi_dphi",                 "k", "-"),
        ("β = 0.01 (default)",   "phi_dphi_d2phi",           "C0", "-"),
        ("β = 0.1",              "phi_dphi_d2phi_beta0.1",  "C3", "--"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for label, sub, color, ls in runs:
        p = base / sub / "assim_diagnostics.npz"
        if not p.exists():
            print(f"[skip] {p}")
            continue
        d = np.load(p)
        axes[0].semilogy(d["t"], d["e_phi"], color=color, ls=ls, lw=1.4, label=label)
        axes[1].plot(d["t"], d["M_total"] - d["M_total"][0],
                     color=color, ls=ls, lw=1.4, label=label)
    axes[0].set_ylabel(r"$e_\phi(t) = \|\phi_{\rm assim} - \phi_{\rm truth}\|_2$")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=10)
    axes[0].set_title(f"{name}: β-sweep on the d²φ channel "
                      "(larger β amplifies shot-noise residuals)")
    axes[1].set_ylabel(r"$M_{\rm total}(t) - M_{\rm total}(0)$")
    axes[1].set_xlabel("t")
    axes[1].grid(alpha=0.3)
    axes[1].axhline(0, color="k", lw=0.6, alpha=0.4)
    fig.tight_layout()
    out = Path(args.truth_dir) / "beta_sweep_d2phi.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[β-sweep] wrote {out}")


if __name__ == "__main__":
    main()
