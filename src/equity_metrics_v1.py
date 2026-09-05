#!/usr/bin/env python3
"""
Equity/robustness metrics comparing Italy (benchmark) and Ghana (focal)
across data-quality regimes. Operationalises R3/H2: does the mechanistic
constraint narrow the gap between high-data and low-data settings, relative
to heuristic baselines?

METRICS
    mase_gap        MASE(low-data regime) - MASE(dense/daily regime), paired
                     by origin, within the SAME country and method.
    peak_timing_err |argmax(pred window) - argmax(true window)| in days.
    ensemble_spread Across-seed IQR of MASE, used as a deterministic-model
                     proxy for predictive uncertainty (ECE/CRPS require a
                     probabilistic head, which the base PINN does not have;
                     this is declared explicitly as the uncertainty proxy).

Usage:
    python3 equity_metrics.py --grid grid.csv --italy-grid italy_grid.csv \
        --out equity_summary.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_mase_gap(df: pd.DataFrame, dense_regime: str = "daily") -> pd.DataFrame:
    pinn = df[df.method.str.startswith("pinn")].copy()
    dense = pinn[pinn.regime == dense_regime].set_index(["country", "method", "origin", "seed"])["mase"]
    rows = []
    for regime in pinn.regime.unique():
        if regime == dense_regime:
            continue
        sub = pinn[pinn.regime == regime].set_index(["country", "method", "origin", "seed"])["mase"]
        joined = pd.concat([sub.rename("mase_sparse"), dense.rename("mase_dense")], axis=1).dropna()
        joined["mase_gap"] = joined["mase_sparse"] - joined["mase_dense"]
        joined["mase_gap_pct"] = 100 * joined["mase_gap"] / joined["mase_dense"]
        joined["regime"] = regime
        rows.append(joined.reset_index())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def compute_ensemble_spread(df: pd.DataFrame) -> pd.DataFrame:
    pinn = df[df.method.str.startswith("pinn")]
    g = pinn.groupby(["country", "method", "regime", "origin"])["mase"]
    out = g.agg(median="median", q25=lambda s: s.quantile(0.25),
                q75=lambda s: s.quantile(0.75)).reset_index()
    out["iqr"] = out["q75"] - out["q25"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, type=Path, help="Ghana grid.csv")
    ap.add_argument("--italy-grid", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    ghana = pd.read_csv(a.grid)
    ghana["country"] = "ghana"
    italy = pd.read_csv(a.italy_grid)
    if "country" not in italy.columns:
        italy["country"] = "italy"

    combined = pd.concat([ghana, italy], ignore_index=True)

    gap = compute_mase_gap(combined)
    spread = compute_ensemble_spread(combined)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    gap.to_csv(a.out, index=False)
    spread.to_csv(a.out.with_name(a.out.stem + "_spread.csv"), index=False)

    print("=== Median MASE gap (%) by country x regime x method ===")
    print(gap.groupby(["country", "method", "regime"])["mase_gap_pct"]
             .median().round(1).to_string())
    print("\n=== Ensemble (across-seed) IQR by country x regime x method ===")
    print(spread.groupby(["country", "method", "regime"])["iqr"]
             .median().round(3).to_string())


if __name__ == "__main__":
    main()
