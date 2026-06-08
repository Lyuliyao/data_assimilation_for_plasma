"""2D2V recover-check aggregator: read 3 per-formulation npz files, write plots + SUMMARY.

Assumes `run_recover_single_2D.py` has already written
`results/<cfg.name>/{none,A_var,B}.npz`. Produces the same figures as the
monolithic legacy runner: error_curves, density/phi/diff snapshots,
fourier_modes, residuals_and_b, plus SUMMARY.md and manifest.json.

The plotting helpers used to live in `run_recover_check_2D.py`; they now
import from this module so the monolithic and array paths stay in sync.
"""
from __future__ import annotations

import argparse
import json
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

from mfda.config_2d import load_moment_2d


FORMULATIONS = ["none", "A", "B", "C"]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _window_mean(t, y, lo, hi):
    mask = (t >= lo) & (t <= hi)
    return float(np.nanmean(y[mask])) if mask.any() else float("nan")


def resolve_out_dir(cfg) -> Path:
    project_root = Path(__file__).resolve().parent.parent
    out_root = Path(cfg.outputs_dir)
    if not out_root.is_absolute():
        out_root = (project_root / out_root).resolve()
    return out_root / cfg.name


def load_runs(out_dir: Path) -> dict[str, dict]:
    runs = {}
    for f in FORMULATIONS:
        path = out_dir / f"{f}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing per-formulation result: {path}. Run "
                f"run_recover_single_2D.py with --formulation {f} first."
            )
        with np.load(path, allow_pickle=False) as npz:
            runs[f] = {k: npz[k] for k in npz.files}
    return runs


def aggregate(cfg_path: Path) -> Path:
    cfg = load_moment_2d(cfg_path)
    out_dir = resolve_out_dir(cfg)
    print(f"[aggregate-2D] reading from: {out_dir}", flush=True)
    runs = load_runs(out_dir)
    plot_error_curves(out_dir, runs)
    plot_density_grid(out_dir, runs, cfg)
    plot_phi_grid(out_dir, runs, cfg)
    plot_diff_grid(out_dir, runs, cfg)
    plot_fourier_modes(out_dir, runs)
    plot_residuals_and_b(out_dir, runs)
    write_summary(out_dir, runs, cfg)
    print("[aggregate-2D] done.", flush=True)
    return out_dir


# ---------------------------------------------------------------------------
# Plots.
# ---------------------------------------------------------------------------


def _smooth_log(y, w=11):
    """Centered moving average in log space (matches the log y-axis), edge-padded."""
    y = np.maximum(np.asarray(y, float), 1e-16)
    if len(y) < w or w < 2:
        return y
    pad = w // 2
    lyp = np.pad(np.log(y), pad, mode="edge")
    sm = np.convolve(lyp, np.ones(w) / w, mode="same")[pad:pad + len(y)]
    return np.exp(sm)


def plot_error_curves(out_dir, runs):
    fig, axes = plt.subplots(2, 2, figsize=figsz(1.0, 0.68), sharex=True)
    panels = [("e_rho", axes[0, 0]), ("e_u", axes[0, 1]),
              ("e_T", axes[1, 0]), ("e_phi", axes[1, 1])]
    titles = {"e_rho": r"$e_\rho$", "e_u": r"$e_u$",
              "e_T": r"$e_T$", "e_phi": r"$e_\varphi$"}
    for metric, ax in panels:
        for f in FORMULATIONS:
            t = runs[f]["t"]
            ser = np.maximum(runs[f][f"assim_log_{metric}"], 1e-16)
            ax.semilogy(t, ser, color=COLORS[f], alpha=0.16, linewidth=0.7)
            ax.semilogy(t, _smooth_log(ser), color=COLORS[f],
                        label=LABELS.get(f, f), linewidth=1.7)
        ax.set_title(titles[metric])
        ax.grid(True, which="both", alpha=0.28)
        tidy_log_yaxis(ax)
        ax.legend(ncol=2, columnspacing=1.0, handlelength=1.3)
    for ax in (axes[1, 0], axes[1, 1]):
        ax.set_xlabel("$t$")
    fig.tight_layout()
    fig.savefig(out_dir / "error_curves.png")
    plt.close(fig)
    print(f"[aggregate-2D]   wrote {out_dir / 'error_curves.png'}", flush=True)


def _pick_4_times(snap_t):
    n_t = len(snap_t)
    if n_t >= 4:
        return np.linspace(0, n_t - 1, 4).astype(int)
    return np.arange(n_t)


def _symmetric_clip(arrays, pct=99.0):
    flat = np.concatenate([np.abs(a).ravel() for a in arrays])
    v = float(np.percentile(flat, pct))
    return v if v > 0 else 1.0


def _grid_imshow_2x4(out_dir, runs, cfg, field_extractor, title, fname,
                    pct=99.0, cmap="RdBu_r"):
    snap_t = runs[FORMULATIONS[0]]["snap_t"]
    idx = _pick_4_times(snap_t)
    extent = (0.0, cfg.Lx, 0.0, cfg.Ly)
    truth_arr = field_extractor(runs[FORMULATIONS[0]], "truth")
    arrays = [truth_arr[ti] for ti in idx]
    for f in FORMULATIONS:
        arr = field_extractor(runs[f], "assim")
        for ti in idx:
            arrays.append(arr[ti])
    v = _symmetric_clip(arrays, pct=pct)
    panels = ["truth"] + FORMULATIONS
    fig, axes = plt.subplots(len(panels), len(idx),
                              figsize=(3.5 * len(idx), 3.0 * len(panels)),
                              sharex=True, sharey=True)
    for col, ti in enumerate(idx):
        axes[0, col].imshow(truth_arr[ti].T, origin="lower", extent=extent,
                             vmin=-v, vmax=v, cmap=cmap, aspect="auto")
        axes[0, col].set_title(f"t = {snap_t[ti]:.1f}")
        for row, f in enumerate(FORMULATIONS, start=1):
            arr = field_extractor(runs[f], "assim")
            axes[row, col].imshow(arr[ti].T, origin="lower", extent=extent,
                                   vmin=-v, vmax=v, cmap=cmap, aspect="auto")
    for row, label in enumerate(panels):
        axes[row, 0].set_ylabel(f"{label}\ny")
    for col in range(len(idx)):
        axes[-1, col].set_xlabel("x")
    fig.suptitle(f"{title}   (sym. scale ±{v:.3g})")
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=140)
    plt.close(fig)
    print(f"[aggregate-2D]   wrote {out_dir / fname}", flush=True)


def plot_density_grid(out_dir, runs, cfg):
    def extract(rec, which):
        key = "rho_truth_snaps" if which == "truth" else "rho_assim_snaps"
        return rec[key] - 1.0
    _grid_imshow_2x4(out_dir, runs, cfg, extract,
                     title=r"$\delta\rho = \rho - 1$",
                     fname="density_snapshots.png")


def plot_phi_grid(out_dir, runs, cfg):
    def extract(rec, which):
        key = "phi_truth_snaps" if which == "truth" else "phi_assim_snaps"
        return rec[key]
    _grid_imshow_2x4(out_dir, runs, cfg, extract,
                     title=r"$\varphi(x, y, t)$",
                     fname="phi_snapshots.png")


def plot_diff_grid(out_dir, runs, cfg):
    snap_t = runs[FORMULATIONS[0]]["snap_t"]
    idx = _pick_4_times(snap_t)
    extent = (0.0, cfg.Lx, 0.0, cfg.Ly)
    drho = {f: runs[f]["rho_assim_snaps"] - runs[f]["rho_truth_snaps"] for f in FORMULATIONS}
    dphi = {f: runs[f]["phi_assim_snaps"] - runs[f]["phi_truth_snaps"] for f in FORMULATIONS}
    v_rho = _symmetric_clip([drho[f][ti] for f in FORMULATIONS for ti in idx])
    v_phi = _symmetric_clip([dphi[f][ti] for f in FORMULATIONS for ti in idx])
    nF = len(FORMULATIONS)
    n_rows = 2 * nF
    fig, axes = plt.subplots(n_rows, len(idx),
                              figsize=(3.5 * len(idx), 2.6 * n_rows),
                              sharex=True, sharey=True)
    for col, ti in enumerate(idx):
        axes[0, col].set_title(f"t = {snap_t[ti]:.1f}")
        for r_off, f in enumerate(FORMULATIONS):
            axes[r_off, col].imshow(drho[f][ti].T, origin="lower", extent=extent,
                                     vmin=-v_rho, vmax=v_rho, cmap="RdBu_r", aspect="auto")
        for r_off, f in enumerate(FORMULATIONS):
            axes[nF + r_off, col].imshow(dphi[f][ti].T, origin="lower", extent=extent,
                                          vmin=-v_phi, vmax=v_phi, cmap="RdBu_r", aspect="auto")
    row_labels = [f"Δρ ({f})" for f in FORMULATIONS] + [f"Δφ ({f})" for f in FORMULATIONS]
    for row, lab in enumerate(row_labels):
        axes[row, 0].set_ylabel(lab)
    for col in range(len(idx)):
        axes[-1, col].set_xlabel("x")
    fig.suptitle(rf"model$-$truth   ($\Delta\rho$ scale ±{v_rho:.3g},  $\Delta\varphi$ scale ±{v_phi:.3g})")
    fig.tight_layout()
    fig.savefig(out_dir / "diff_snapshots.png", dpi=140)
    plt.close(fig)
    print(f"[aggregate-2D]   wrote {out_dir / 'diff_snapshots.png'}", flush=True)


def plot_fourier_modes(out_dir, runs):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    t_truth = runs[FORMULATIONS[0]]["t"]
    phi_truth = runs[FORMULATIONS[0]]["truth_phi_hat"]
    rho_truth = runs[FORMULATIONS[0]]["truth_rho_hat"]
    axes[0, 0].plot(t_truth, np.abs(phi_truth), color="k", linewidth=1.8, label="truth")
    for f in FORMULATIONS:
        axes[0, 0].plot(runs[f]["t"], np.abs(runs[f]["assim_phi_hat"]),
                         color=COLORS[f], linewidth=1.3, label=f)
    axes[0, 0].set_title(r"$|\widehat{\varphi}_{k_d}(t)|$")
    axes[0, 0].set_ylabel("amplitude")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 1].plot(t_truth, np.abs(rho_truth), color="k", linewidth=1.8, label="truth")
    for f in FORMULATIONS:
        axes[0, 1].plot(runs[f]["t"], np.abs(runs[f]["assim_rho_hat"]),
                         color=COLORS[f], linewidth=1.3, label=f)
    axes[0, 1].set_title(r"$|\widehat{\rho}_{k_d}(t)|$")
    axes[0, 1].grid(True, alpha=0.3)
    for f in FORMULATIONS:
        ph_a = np.unwrap(np.angle(runs[f]["assim_phi_hat"]))
        ph_t = np.unwrap(np.angle(runs[f]["truth_phi_hat"]))
        axes[1, 0].plot(runs[f]["t"], ph_a - ph_t,
                         color=COLORS[f], linewidth=1.3, label=f)
    axes[1, 0].axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
    axes[1, 0].set_title(r"$\arg \widehat{\varphi}_{k_d}^{\,\rm model} - \arg \widehat{\varphi}_{k_d}^{\,\rm truth}$ (rad)")
    axes[1, 0].set_xlabel("t")
    axes[1, 0].set_ylabel("phase error (rad)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    for f in FORMULATIONS:
        amp_err = np.abs(runs[f]["assim_phi_hat"]) - np.abs(runs[f]["truth_phi_hat"])
        axes[1, 1].plot(runs[f]["t"], amp_err, color=COLORS[f], linewidth=1.3, label=f)
    axes[1, 1].axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
    axes[1, 1].set_title(r"$|\widehat{\varphi}_{k_d}^{\,\rm model}| - |\widehat{\varphi}_{k_d}^{\,\rm truth}|$")
    axes[1, 1].set_xlabel("t")
    axes[1, 1].grid(True, alpha=0.3)
    fig.suptitle("Fourier mode at driver wavevector")
    fig.tight_layout()
    fig.savefig(out_dir / "fourier_modes.png", dpi=140)
    plt.close(fig)
    print(f"[aggregate-2D]   wrote {out_dir / 'fourier_modes.png'}", flush=True)


def plot_residuals_and_b(out_dir, runs):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for f in FORMULATIONS:
        t = runs[f]["t"]
        axes[0].semilogy(t, np.maximum(runs[f]["r0_norm"], 1e-16),
                          color=COLORS[f], linestyle="-",  linewidth=1.3, label=f"r0 ({f})")
        axes[0].semilogy(t, np.maximum(runs[f]["r1_norm"], 1e-16),
                          color=COLORS[f], linestyle="--", linewidth=1.3, label=f"r1 ({f})")
        axes[0].semilogy(t, np.maximum(runs[f]["r2_norm"], 1e-16),
                          color=COLORS[f], linestyle=":",  linewidth=1.3, label=f"r2 ({f})")
    axes[0].set_xlabel("t")
    axes[0].set_title("Moment-residual L2 norms")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(ncol=3, fontsize=8)
    for f in FORMULATIONS:
        tn = runs[f]["t_nudge"]
        if tn.size == 0:
            continue
        bx = runs[f]["bx_rms"]
        by = runs[f]["by_rms"]
        bvx = runs[f]["bvx_rms"]
        bvy = runs[f]["bvy_rms"]
        b_pos = np.sqrt(bx * bx + by * by)
        b_vel = np.sqrt(bvx * bvx + bvy * bvy)
        axes[1].semilogy(tn, np.maximum(b_pos, 1e-16),
                          color=COLORS[f], linestyle="-",  linewidth=1.3, label=f"|b_x| ({f})")
        axes[1].semilogy(tn, np.maximum(b_vel, 1e-16),
                          color=COLORS[f], linestyle="--", linewidth=1.3, label=f"|b_v| ({f})")
    axes[1].set_xlabel("t")
    axes[1].set_title("Nudging-correction RMS over particles")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "residuals_and_b.png", dpi=140)
    plt.close(fig)
    print(f"[aggregate-2D]   wrote {out_dir / 'residuals_and_b.png'}", flush=True)


def write_summary(out_dir, runs, cfg):
    sha = _git_sha()
    lines = [f"# {cfg.name}\n",
             f"git SHA: `{sha}`\n",
             "## Late-time errors (t > t_end / 2)\n",
             "| formulation | e_rho_late | e_u_late | e_T_late | e_phi_late |",
             "| --- | --- | --- | --- | --- |"]
    for f in FORMULATIONS:
        t = runs[f]["t"]
        lo, hi = float(t.max()) / 2.0, float(t.max())
        late_rho = _window_mean(t, runs[f]["assim_log_e_rho"], lo, hi)
        late_u   = _window_mean(t, runs[f]["assim_log_e_u"], lo, hi)
        late_T   = _window_mean(t, runs[f]["assim_log_e_T"], lo, hi)
        late_phi = _window_mean(t, runs[f]["assim_log_e_phi"], lo, hi)
        lines.append(f"| {f} | {late_rho:.4g} | {late_u:.4g} | {late_T:.4g} | {late_phi:.4g} |")
    lines += [
        "", "## Error curves", "", "![error curves](error_curves.png)",
        "", "## Density snapshots (δρ)", "", "![density snapshots](density_snapshots.png)",
        "", "## Potential snapshots (φ)", "", "![phi snapshots](phi_snapshots.png)",
        "", "## Model − truth differences", "", "![diff snapshots](diff_snapshots.png)",
        "", "## Fourier mode at driver wavevector", "", "![fourier modes](fourier_modes.png)",
        "", "## Residuals and nudging-correction RMS", "", "![residuals and b](residuals_and_b.png)",
        "",
    ]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines))
    with open(out_dir / "manifest.json", "w") as fh:
        json.dump({"config": cfg.model_dump(), "git_sha": sha}, fh, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to YAML")
    args = ap.parse_args()
    aggregate(Path(args.config))


if __name__ == "__main__":
    main()
