"""Aggregate per-config compare_summary.csv / gamma_sweep_*.csv files into a
single identifiability-floor table.

For each (config, variant, gamma) row present in the source summaries, writes
an entry in results/test0/identifiability_floor_table.csv with:
    config, variant, gamma, seed, e_phi_floor, e_f_floor,
    energy_drift, mode1_final, mode1_truth_final

The 'floor' here just means 'final-time value'; it is the quantity we asymptote
toward as the nudger settles, not a strict asymptote (runs are finite).

docs/test0_next_steps.md §3 also lists a vortex_count_truth column; that
diagnostic is not yet implemented (no vortex counter in mfda.diagnostics), so
that column is omitted here. When the diagnostic lands, add it to the read_*
helpers below.

Usage:
    python scripts/floor_table.py \
        --configs test0_identifiability_linear test0_identifiability_stable \
        --output results/test0/identifiability_floor_table.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


_VARIANT_TO_CHANNELS = {
    "none": "none",
    "velocity": "vs",      # snapshot velocity
    "position": "ps",      # snapshot position
}


def _row_from_record(rec: dict, config_name: str) -> dict:
    """Convert a CSV record into the floor-table row schema.

    Schema variations seen in the wild:
      compare_summary.csv     — variant, gamma  (no seed / no alpha)
      gamma_sweep_*.csv       — variant, gamma, seed
      alpha_sweep_*.csv       — variant, gamma, alpha, seed
      combined_sweep.csv      — channels, gamma, alpha, seed  (no variant)

    The `channels` short codes are: ps (position_snapshot), vs
    (velocity_snapshot), pd (position_dtobs), vd (velocity_dtobs), e.g.
    "ps+pd+vd" for the combined default. For the legacy variant rows we
    map velocity->vs, position->ps, none->none.
    """
    if "channels" in rec:
        channels = rec["channels"]
        variant = channels  # store the canonical channels string in `variant`
    else:
        variant = rec["variant"]
        channels = _VARIANT_TO_CHANNELS.get(variant, variant)
    return {
        "config": config_name,
        "channels": channels,
        "variant": variant,
        "gamma": float(rec["gamma"]),
        "alpha": float(rec["alpha"]) if rec.get("alpha") not in (None, "") else float("nan"),
        "seed": int(rec["seed"]) if rec.get("seed") not in (None, "") else 0,
        "e_phi_floor": float(rec["e_phi_final"]),
        "e_f_floor": float(rec["e_f_final"]),
        "energy_drift": float(rec["energy_final"]),
        "mode1_final": float(rec.get("mode1_final") or "nan"),
        "mode1_truth_final": float(rec.get("mode1_truth_final") or "nan"),
        "e_j_final": float(rec["e_j_final"]) if rec.get("e_j_final") not in (None, "") else float("nan"),
    }


def _read_csv(path: Path, config_name: str) -> list[dict]:
    """Parse any of compare_summary.csv / gamma_sweep_*.csv / alpha_sweep_*.csv."""
    rows: list[dict] = []
    with path.open() as f:
        for rec in csv.DictReader(f):
            rows.append(_row_from_record(rec, config_name))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True,
                    help="config names (directory stems under results/), "
                         "e.g. test0_identifiability_linear test0_identifiability_stable")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--output", default="results/test0/identifiability_floor_table.csv")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    # Dict keyed on (config, variant, gamma, alpha, seed) to drop duplicates.
    # When the same key appears in both compare_summary.csv and a sweep CSV,
    # later inserts overwrite earlier ones — sweep CSVs are read after the
    # compare summary so their values win (γ-sweep is the more deliberate
    # measurement; compare_summary.csv is just one γ slice).
    rows_by_key: dict[tuple, dict] = {}

    def _key(r: dict) -> tuple:
        # NaN does not equal NaN, so coerce to a sentinel for keying.
        a = r["alpha"]
        a_key = "nan" if a != a else a
        return (r["config"], r["variant"], r["gamma"], a_key, r["seed"])

    for cfg_name in args.configs:
        cfg_dir = results_root / cfg_name
        if not cfg_dir.is_dir():
            print(f"[floor_table] skipping {cfg_name}: results dir not found ({cfg_dir})")
            continue
        # compare_summary.csv first so sweeps can override.
        cs_path = cfg_dir / "compare_summary.csv"
        if cs_path.exists():
            for row in _read_csv(cs_path, cfg_name):
                rows_by_key[_key(row)] = row
        for sweep in sorted(cfg_dir.glob("gamma_sweep_*.csv")):
            for row in _read_csv(sweep, cfg_name):
                rows_by_key[_key(row)] = row
        for sweep in sorted(cfg_dir.glob("alpha_sweep_*.csv")):
            for row in _read_csv(sweep, cfg_name):
                rows_by_key[_key(row)] = row
        for sweep in sorted(cfg_dir.glob("combined_sweep*.csv")):
            for row in _read_csv(sweep, cfg_name):
                rows_by_key[_key(row)] = row

    if not rows_by_key:
        print("[floor_table] no input rows found; nothing to write")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["config", "channels", "variant", "gamma", "alpha", "seed",
                  "e_phi_floor", "e_f_floor", "energy_drift",
                  "mode1_final", "mode1_truth_final", "e_j_final"]

    all_rows = list(rows_by_key.values())
    # Sort with NaN alpha first (snapshot rows) then ascending alpha.
    all_rows.sort(key=lambda r: (
        r["config"], r["variant"], r["gamma"],
        (1, r["alpha"]) if r["alpha"] == r["alpha"] else (0, 0.0),
        r["seed"],
    ))

    with out_path.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(all_rows)

    print(f"wrote {out_path} ({len(all_rows)} rows, {len(args.configs)} configs)")


if __name__ == "__main__":
    main()
