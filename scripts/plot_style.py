"""Shared figure style for the paper.

One source of truth so every figure is visually uniform:
  * ONE color per method, never switched (COLORS / color()).
  * paper-matching fonts: serif text + Computer-Modern math (mathtext 'cm'),
    consistent sizes (apply_style()).  usetex is avoided (no dvipng on the
    cluster); 'cm' mathtext reproduces the LaTeX look closely.
  * a log-space temporal smoother (smooth_log) for noisy error curves.
  * shared plotters: moment_error_curves(), velocity_marginal().

Import this in every plotting script:  from plot_style import ...
"""
from __future__ import annotations

import matplotlib as mpl
import numpy as np

# --- ONE color per method (do not switch across figures) -------------------
COLORS = {
    "none":     "#7f7f7f",   # gray
    "aot":      "#d62728",   # red
    "A":        "#1f77b4",   # blue
    "B":        "#ff7f0e",   # orange
    "C":        "#2ca02c",   # green
    "naive_kl": "#9467bd",   # purple
    "truth":    "#000000",   # black
    "A_var":    "#8c564b",   # brown (legacy; not used in the paper)
}
LABELS = {
    "none": "none", "aot": "AOT", "A": "A", "B": "B", "C": "C",
    "naive_kl": r"naive $\mathrm{KL}$", "truth": "truth",
}
# Stable left-to-right order so legends/bars are consistent everywhere.
ORDER = ["none", "aot", "A", "B", "C", "naive_kl"]


def color(f: str) -> str:
    return COLORS.get(f, "#333333")


def label(f: str) -> str:
    return LABELS.get(f, f)


# Text width of the paper (arxiv.sty), in inches: 469.755pt / 72.27.
LW = 6.50


def figsize(frac=1.0, aspect=0.62):
    """Figure size whose WIDTH equals the on-page display width (frac*LW),
    so \\includegraphics[width=frac\\linewidth] does NOT rescale (scale=1) and
    fonts render at their native pt. `aspect` = height/width."""
    w = LW * frac
    return (w, w * aspect)


def apply_style() -> None:
    """Paper-matching rcParams: serif text, Computer-Modern math, ~paper pt.

    Sizes are chosen for scale=1 (figsize width == display width), so they
    land near the 10pt body text rather than shrinking under rescaling.
    """
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        # 'standard' (not 'tight') so the saved width equals the figsize width
        # exactly (frac*TEXTWIDTH); tight_layout keeps labels inside the canvas.
        "savefig.bbox": "standard",
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "mathtext.rm": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.7,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linewidth": 0.5,
    })


def tidy_log_yaxis(ax):
    """Make a semilogy y-axis legible across very different dynamic ranges:
      * < 0.5 decade  -> switch to LINEAR (a log decade is cramped/ugly here),
                         with ~4 evenly spaced labels;
      * 0.5-1.6 decade-> log, decade labels + sparse labeled minors (2,5);
      * > 1.6 decade  -> log, decade labels only, unlabeled minors.
    Avoids both the single-bare-tick and the over-dense-tick failure modes."""
    from matplotlib import ticker
    lo, hi = ax.get_ylim()
    if lo <= 0 or not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return
    ndec = np.log10(hi) - np.log10(lo)
    if ndec < 0.5:
        ax.set_yscale("linear")
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune=None))
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    elif ndec < 1.6:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))
        ax.yaxis.set_minor_locator(
            ticker.LogLocator(base=10.0, subs=(2.0, 5.0), numticks=12))
        ax.yaxis.set_minor_formatter(
            ticker.LogFormatterSciNotation(base=10.0, labelOnlyBase=False,
                                           minor_thresholds=(2.5, 0.5)))
    else:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))
        ax.yaxis.set_minor_locator(
            ticker.LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=12))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())


def smooth_log(y, w: int = 9):
    """Centered moving average in log space (matches a log y-axis), edge-padded."""
    y = np.maximum(np.asarray(y, float), 1e-16)
    if len(y) < w or w < 2:
        return y
    pad = w // 2
    lyp = np.pad(np.log(y), pad, mode="edge")
    sm = np.convolve(lyp, np.ones(w) / w, mode="same")[pad:pad + len(y)]
    return np.exp(sm)


_TITLES = {"e_rho": r"$e_\rho$", "e_u": r"$e_u$", "e_T": r"$e_T$",
           "e_f": r"$e_f$", "e_phi": r"$e_\varphi$"}


def moment_error_curves(out_path, runs, forms, metrics=("e_rho", "e_u", "e_T", "e_f"),
                        smooth=True, suptitle=None):
    """2x2 semilogy of moment errors vs t. runs[f] = {'t':..., metric:...}."""
    import matplotlib.pyplot as plt
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize(1.0, 0.68), sharex=True)
    for m, ax in zip(metrics, axes.flat):
        for f in [g for g in ORDER if g in forms]:
            if m not in runs[f]:
                continue
            t = np.asarray(runs[f]["t"], float)
            ser = np.maximum(np.asarray(runs[f][m], float), 1e-16)
            if smooth:
                ax.semilogy(t, ser, color=color(f), alpha=0.15, lw=0.7)
                ax.semilogy(t, smooth_log(ser), color=color(f), label=label(f), lw=1.7)
            else:
                ax.semilogy(t, ser, color=color(f), label=label(f), lw=1.7)
        ax.set_title(_TITLES.get(m, m))
        ax.grid(True, which="both", alpha=0.28)
        tidy_log_yaxis(ax)
        ax.legend(ncol=2, columnspacing=1.0, handlelength=1.3)
    for ax in axes[-1, :]:
        ax.set_xlabel("$t$")
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def velocity_marginal(out_path, finals, forms, v_min, v_max, nbins=120, title=None):
    """f(v) at final time: truth (black) vs each formulation.
    finals[f] = {'v':..., 'w':...}; finals['truth'] for the reference."""
    import matplotlib.pyplot as plt
    apply_style()
    fig, ax = plt.subplots(figsize=figsize(0.72, 0.66))
    bins = np.linspace(v_min, v_max, nbins + 1)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    tr = finals["truth"]
    h, _ = np.histogram(tr["v"], bins=bins, weights=tr["w"], density=True)
    ax.plot(ctr, h, color=color("truth"), lw=2.4, label=label("truth"), zorder=5)
    for f in [g for g in ORDER if g in forms]:
        d = finals[f]
        h, _ = np.histogram(d["v"], bins=bins, weights=d["w"], density=True)
        ax.plot(ctr, h, color=color(f), lw=1.5, label=label(f))
    ax.set_xlabel("$v$")
    ax.set_ylabel("$f(v)$ at final time")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
