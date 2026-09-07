#!/usr/bin/env python3
"""Add deterministic synthetic-observation masks to one Italy master CSV.

Run this separately for Italy PRIMARY and Italy SECONDARY. Unlike the old
version, this script updates the master file that run_italy.py actually reads,
so observed_sparse20/30/40 are present in the same CSV used for fitting.

The dropout pattern is deterministic per (arm, percentage), while remaining
stable across split/joint modes and PINN seeds within an arm. This makes the
within-arm comparisons paired and reproducible. The masks are generated over
the full arm, but run_italy.py correctly slices each mask to the selected
training origin.

Example:
  PYTHONPATH=src python src/sparsify.py \
    --master data/processed/italy_secondary_national_master.csv \
    --country italy_secondary \
    --pct 20 30 40 \
    --inplace
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SEED_BASE = 20260101


def arm_seed(country: str, pct: int) -> int:
    """Stable, arm-specific seed; do not use Python's salted hash()."""
    digest = hashlib.sha256(country.encode("utf-8")).digest()
    arm_offset = int.from_bytes(digest[:4], "little")
    return (SEED_BASE + pct + arm_offset) % (2**32)


def make_mask(n_days: int, pct: int, country: str) -> np.ndarray:
    if not 0 < pct < 100:
        raise ValueError(f"pct must be between 1 and 99, got {pct}")
    rng = np.random.default_rng(arm_seed(country, pct))
    observed = np.ones(n_days, dtype=np.int8)
    n_drop = int(round(n_days * pct / 100))
    drop_idx = rng.choice(n_days, size=n_drop, replace=False)
    observed[drop_idx] = 0
    return observed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True, type=Path,
                    help="Italy-primary or Italy-secondary master used directly by run_italy.py")
    ap.add_argument("--country", required=True,
                    choices=["italy_primary", "italy_secondary"],
                    help="Arm label used to create deterministic but arm-specific masks")
    ap.add_argument("--pct", type=int, nargs="+", default=[20, 30, 40])
    target = ap.add_mutually_exclusive_group(required=True)
    target.add_argument("--inplace", action="store_true",
                        help="Overwrite --master after retaining a timestamp-free .pre_sparsify.csv backup")
    target.add_argument("--out", type=Path,
                        help="Write an augmented master to this path; use this path as run_italy.py --master")
    ap.add_argument("--force", action="store_true",
                    help="Replace existing observed_sparseNN columns instead of stopping")
    args = ap.parse_args()

    if not args.master.exists():
        raise SystemExit(f"Master file not found: {args.master}")

    df = pd.read_csv(args.master, index_col=0, parse_dates=True)
    if "I_proxy" not in df.columns:
        raise SystemExit(f"{args.master} lacks I_proxy; it is not a compatible Italy master.")

    cols = [f"observed_sparse{pct}" for pct in args.pct]
    existing = [col for col in cols if col in df.columns]
    if existing and not args.force:
        raise SystemExit(
            f"Mask column(s) already exist: {existing}. Nothing changed. "
            "Use --force only if you intentionally want to regenerate them."
        )

    for pct in args.pct:
        col = f"observed_sparse{pct}"
        mask = make_mask(len(df), pct, args.country)
        df[col] = mask
        print(
            f"{args.country} {col}: observed={int(mask.sum())}/{len(mask)} "
            f"({100 * mask.mean():.1f}%), dropped={int((mask == 0).sum())}, "
            f"seed={arm_seed(args.country, pct)}"
        )

    if args.inplace:
        destination = args.master
        backup = args.master.with_name(args.master.stem + ".pre_sparsify.csv")
        if not backup.exists():
            pd.read_csv(args.master).to_csv(backup, index=False)
            print(f"Backup written: {backup}")
    else:
        destination = args.out

    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination)
    print(f"Augmented master written: {destination}")


if __name__ == "__main__":
    main()
