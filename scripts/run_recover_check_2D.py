"""2D2V recover-check: single-process driver (none / A_var / B sequentially).

Equivalent to running `run_recover_single_2D.py` three times then
`aggregate_recover_check_2D.py`, but in a single process. Kept for local
dev / smoke tests. For SLURM, prefer the array path:
    slurm/submit_recover_check_2D.sh <config.yaml>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate_recover_check_2D import FORMULATIONS, aggregate
from run_recover_single_2D import run_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to YAML")
    args = ap.parse_args()
    cfg_path = Path(args.config)
    for f in FORMULATIONS:
        run_one(cfg_path, f)
    aggregate(cfg_path)


if __name__ == "__main__":
    main()
