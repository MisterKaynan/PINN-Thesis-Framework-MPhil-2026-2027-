#!/usr/bin/env python3
"""Add deterministic synthetic-observation masks to one Italy master CSV.

Run this separately for Italy PRIMARY and Italy SECONDARY. Unlike the old
version, this script updates the master file that run_italy.py actually reads,
so observed_sparse20/30/40 are present in the same CSV used for fitting.

The dropout pattern is deterministic per arm and stable across split/joint
modes and PINN seeds within an arm. This makes within-arm comparisons paired
and reproducible. Percentage masks are nested:
the 30% dropout set contains the 20% set, and the 40% set contains both.
Masks are generated over the full arm; run_italy.py slices each mask to each
training origin and reports the realized missingness for that prefix.

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


def arm_seed(country: str) -> int:
    """Return a stable arm-specific seed shared by all percentages."""
    digest = hashlib.sha256(country.encode("utf-8")).digest()
    arm_offset = int.from_bytes(digest[:4], "little")
    return (SEED_BASE + arm_offset) % (2**32)


def make_nested_masks(
    n_days: int, percentages: list[int], country: str
) -> dict[int, np.ndarray]:
    """Generate nested masks from one deterministic ordering of dropped days."""
    if n_days < 0:
        raise ValueError(f"n_days must be non-negative, got {n_days}")
    if len(set(percentages)) != len(percentages):
        raise ValueError("pct values must be unique")
    if any(not 0 < pct < 100 for pct in percentages):
        raise ValueError(f"pct values must be between 1 and 99, got {percentages}")

    # A single permutation makes each higher-drop mask extend every lower-drop
    # mask, preserving paired comparisons while keeping each full-arm count exact.
    rng = np.random.default_rng(arm_seed(country))
    drop_order = rng.permutation(n_days)
    masks = {}
    for pct in percentages:
        observed = np.ones(n_days, dtype=np.int8)
        n_drop = int(round(n_days * pct / 100))
        observed[drop_order[:n_drop]] = 0
        masks[pct] = observed
    return masks


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

    # Validate input and requested percentages before changing or creating files.
    if not args.master.exists():
        raise SystemExit(f"Master file not found: {args.master}")
    if len(set(args.pct)) != len(args.pct):
        ap.error("--pct values must be unique")
    if any(not 0 < pct < 100 for pct in args.pct):
        ap.error("--pct values must be between 1 and 99")

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

    # Generate all requested levels together to guarantee nesting rather than
    # drawing independent dropout locations for each percentage.
    masks = make_nested_masks(len(df), args.pct, args.country)
    for pct, mask in masks.items():
        col = f"observed_sparse{pct}"
        df[col] = mask
        print(
            f"{args.country} {col}: observed={int(mask.sum())}/{len(mask)} "
            f"({100 * mask.mean():.1f}%), dropped={int((mask == 0).sum())}, "
            f"seed={arm_seed(args.country)}"
        )

    # Preserve the original master before the first in-place augmentation.
    if args.inplace:
        destination = args.master
        backup = args.master.with_name(args.master.stem + ".pre_sparsify.csv")
        if not backup.exists():
            pd.read_csv(args.master).to_csv(backup, index=False)
            print(f"Backup written: {backup}")
    else:
        destination = args.out

    # Write all masks together so the selected arm's master remains the runner's
    # single source for incidence and synthetic observation indicators.
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination)
    print(f"Augmented master written: {destination}")


if __name__ == "__main__":
    main()
