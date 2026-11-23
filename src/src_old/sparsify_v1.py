#!/usr/bin/env python3
"""
Impose controlled synthetic missingness on Italy's dense series so it can be
degraded step-wise toward Ghana-like sparsity. This is the axis that lets the
thesis interpolate between "Italy dense" (Millevoi's original regime) and
"Ghana authentic mask" (the course project's regime), rather than jumping
directly between the two real datasets.

DESIGN
    Regime "sparseNN" drops NN% of days at random, with a FIXED seed per
    regime (not per run) so every method/mode sees the identical dropout
    pattern -- this makes joint vs split comparisons paired and fair.

Usage: python3 sparsify.py --master italy_national_master.csv --pct 20 30 40
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SPARSIFY_SEED_BASE = 20260101  # fixed per-regime seed derivation, not run-to-run random


def make_mask(n_days: int, pct: int) -> np.ndarray:
    rng = np.random.default_rng(SPARSIFY_SEED_BASE + pct)
    keep = np.ones(n_days, dtype=bool)
    drop_n = int(round(n_days * pct / 100))
    drop_idx = rng.choice(n_days, size=drop_n, replace=False)
    keep[drop_idx] = False
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True, type=Path)
    ap.add_argument("--pct", type=int, nargs="+", default=[20, 30, 40])
    ap.add_argument("--outdir", required=True, type=Path)
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(a.master, index_col=0, parse_dates=True)
    n = len(master)
    for pct in a.pct:
        mask = make_mask(n, pct)
        out = master.copy()
        out[f"observed_sparse{pct}"] = mask.astype(int)
        dest = a.outdir / f"italy_sparse{pct}.csv"
        out.to_csv(dest)
        print(f"sparse{pct}: kept {mask.sum()}/{n} days ({100*mask.sum()/n:.1f}%) -> {dest}")


if __name__ == "__main__":
    main()
