"""Save Fourier modes of phi, j, and m2 for every case x method.

Mirrors the particle-sum approach in notebooks/result.ipynb.

For each ``cases/<case>/results/<method>/particles.h5``:

    rho_hat[k, n] = (1/L) sum_p w_p exp(-i k_phys x_p[n])
    j_hat  [k, n] = (1/L) sum_p w_p v_p[n] exp(-i k_phys x_p[n])
    m2_hat [k, n] = (1/L) sum_p w_p v_p[n]**2 exp(-i k_phys x_p[n])

with k_phys = 2 pi k / L. The potential mode is recovered from
periodic Poisson (-Delta phi = rho - 1) with the zero-mean gauge:

    phi_hat[k, n] = rho_hat[k, n] / k_phys**2     (k != 0)

Output: ``cases/<case>/results/<method>/fourier_modes.npz`` with arrays
``t``, ``ks``, ``L``, ``phi_hat``, ``j_hat``, ``m2_hat`` plus
``stride`` and ``normalize`` metadata.

Usage:
    python scripts/save_fourier_modes.py
    python scripts/save_fourier_modes.py --ks 1,2,3 --stride 10 --force
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def particle_moment_modes_snapshot(
    x: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    L: float,
    ks: np.ndarray,
    normalize: str = "density",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_mod = np.mod(np.asarray(x), L)
    phase = np.exp(-1j * 2 * np.pi * ks[:, None] * x_mod[None, :] / L)

    rho_hat = phase @ w
    j_hat = phase @ (w * v)
    m2_hat = phase @ (w * v * v)

    if normalize == "density":
        rho_hat = rho_hat / L
        j_hat = j_hat / L
        m2_hat = m2_hat / L
    elif normalize == "mass":
        mass = float(np.sum(w))
        rho_hat = rho_hat / mass
        j_hat = j_hat / mass
        m2_hat = m2_hat / mass
    elif normalize != "none":
        raise ValueError(f"unknown normalize: {normalize}")
    return rho_hat, j_hat, m2_hat


def L_from_manifest(method_dir: Path) -> float:
    for name in ("truth_manifest.json", "assim_manifest.json"):
        p = method_dir / name
        if p.exists():
            cfg = json.loads(p.read_text())["config"]
            return float(cfg["domain"]["L"])
    raise FileNotFoundError(f"no manifest in {method_dir}")


def process_method(
    method_dir: Path,
    ks: np.ndarray,
    stride: int,
    normalize: str,
    out_name: str,
    force: bool,
) -> None:
    h5_path = method_dir / "particles.h5"
    out_path = method_dir / out_name
    if not h5_path.exists():
        return
    if out_path.exists() and not force:
        print(f"[skip] {out_path} (exists; pass --force to overwrite)")
        return

    L = L_from_manifest(method_dir)
    with h5py.File(h5_path, "r") as f:
        n_steps = f["x"].shape[0]
        idx = np.arange(0, n_steps, stride)
        wp = np.asarray(f["w"])
        t = np.asarray(f["t"][idx]) if "t" in f else idx.astype(float)

        K, T = len(ks), len(idx)
        rho_hat = np.empty((K, T), dtype=np.complex128)
        j_hat = np.empty((K, T), dtype=np.complex128)
        m2_hat = np.empty((K, T), dtype=np.complex128)

        for m, n in enumerate(idx):
            x = f["x"][n, :]
            v = f["v"][n, :]
            rho_hat[:, m], j_hat[:, m], m2_hat[:, m] = particle_moment_modes_snapshot(
                x, v, wp, L, ks, normalize=normalize
            )

    k_phys = 2 * np.pi * ks / L
    phi_hat = rho_hat / (k_phys[:, None] ** 2)

    np.savez(
        out_path,
        t=t,
        ks=ks,
        L=L,
        phi_hat=phi_hat,
        j_hat=j_hat,
        m2_hat=m2_hat,
        stride=np.int64(stride),
        normalize=normalize,
    )
    print(f"[ok] {out_path}  K={K}, T={T}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases-root", default="cases", help="root holding cases/<name>/results/")
    ap.add_argument("--ks", default="1,2,3,4,5,6", help="comma-separated integer modes")
    ap.add_argument("--stride", type=int, default=20, help="time stride into particles.h5")
    ap.add_argument(
        "--normalize",
        choices=("density", "mass", "none"),
        default="density",
        help='"density" divides by L (matches notebooks/result.ipynb)',
    )
    ap.add_argument("--out-name", default="fourier_modes.npz")
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = ap.parse_args()

    ks = np.array([int(s) for s in args.ks.split(",") if s.strip()], dtype=int)
    if np.any(ks == 0):
        raise SystemExit("k=0 is the (enforced) zero-mean gauge; remove it from --ks")

    root = Path(args.cases_root)
    method_dirs = sorted(p for p in root.glob("*/results/*") if p.is_dir())
    if not method_dirs:
        raise SystemExit(f"no method dirs found under {root}/*/results/*")

    for method_dir in method_dirs:
        try:
            process_method(
                method_dir, ks, args.stride, args.normalize, args.out_name, args.force
            )
        except Exception as e:
            print(f"[err] {method_dir}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
