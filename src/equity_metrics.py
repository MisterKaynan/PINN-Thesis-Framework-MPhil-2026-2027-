#!/usr/bin/env python3
"""
Equity/robustness metrics -- v2, updated for the two-arm Italy design.

CHANGES FROM v1 (equity_metrics-2.py as attached)
    1. v1 accepted a single --italy-grid and force-labelled every row
       country="italy" if missing. This silently pools italy_primary and
       italy_secondary if both happen to be concatenated upstream --
       exactly the "confound smuggling" the two-arm design was meant to
       prevent. v2 requires the grid's own "country" column to already
       distinguish italy_primary vs italy_secondary (as produced by the
       updated notebook Section 7 merge step) and REFUSES to silently
       default an ambiguous "italy" label.
    2. The core Objective 1/2 equity comparison (compute_mase_gap,
       compute_ensemble_spread) now runs ONLY on rows where country is
       "ghana" or "italy_primary" by default. italy_secondary rows are
       excluded from the primary equity_summary.csv UNLESS
       --include-secondary is passed, and even then they are written to a
       SEPARATE output file, never merged into the primary comparison.
    3. Added a print-time guard: if italy_secondary rows are detected in
       the input grid without --include-secondary, a warning is printed
       (not silently dropped without notice) so the exclusion is visible
       in every run's log.

Usage:
    python3 equity_metrics.py --grid grid.csv --italy-grid italy_grid.csv \
        --out equity_summary.csv
    # to additionally emit the labelled secondary-arm read:
    python3 equity_metrics.py --grid grid.csv --italy-grid italy_grid.csv \
        --out equity_summary.csv --include-secondary
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PRIMARY_COUNTRIES = {"ghana", "italy_primary"}


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
    ap.add_argument("--italy-grid", required=True, type=Path,
                    help="Italy grid CSV; 'country' column must be "
                         "'italy_primary' or 'italy_secondary' per row")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--include-secondary", action="store_true",
                    help="Also emit a SEPARATE, labelled secondary-arm read "
                         "(regime-shift robustness only, Objective 3). Never "
                         "merged into the primary equity comparison.")
    a = ap.parse_args()

    ghana = pd.read_csv(a.grid)
    if "country" not in ghana.columns:
        ghana["country"] = "ghana"
    italy = pd.read_csv(a.italy_grid)
    if "country" not in italy.columns:
        raise SystemExit(
            "italy-grid is missing a 'country' column with values "
            "'italy_primary'/'italy_secondary'. Refusing to silently guess "
            "-- see prepare_italy.py two-arm design. Re-export with the "
            "country column set explicitly.")

    has_secondary = (italy["country"] == "italy_secondary").any()
    if has_secondary and not a.include_secondary:
        n_sec = (italy["country"] == "italy_secondary").sum()
        print(f"NOTE: {n_sec} italy_secondary rows found in --italy-grid but "
              f"--include-secondary was not passed. These rows are EXCLUDED "
              f"from equity_summary.csv (primary comparison uses ghana + "
              f"italy_primary only). Pass --include-secondary for a separate, "
              f"labelled robustness read.")

    combined_primary = pd.concat(
        [ghana, italy[italy["country"] != "italy_secondary"]], ignore_index=True
    )
    combined_primary = combined_primary[combined_primary["country"].isin(PRIMARY_COUNTRIES)]

    gap = compute_mase_gap(combined_primary)
    spread = compute_ensemble_spread(combined_primary)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    gap.to_csv(a.out, index=False)
    spread.to_csv(a.out.with_name(a.out.stem + "_spread.csv"), index=False)

    print("=== PRIMARY equity comparison (ghana vs italy_primary) ===")
    print("Median MASE gap (%) by country x regime x method:")
    print(gap.groupby(["country", "method", "regime"])["mase_gap_pct"]
             .median().round(1).to_string())
    print("\nEnsemble (across-seed) IQR by country x regime x method:")
    print(spread.groupby(["country", "method", "regime"])["iqr"]
             .median().round(3).to_string())

    if a.include_secondary and has_secondary:
        sec = italy[italy["country"] == "italy_secondary"]
        gap_sec = compute_mase_gap(sec)
        spread_sec = compute_ensemble_spread(sec)
        sec_out = a.out.with_name(a.out.stem + "_secondary_robustness.csv")
        gap_sec.to_csv(sec_out, index=False)
        spread_sec.to_csv(sec_out.with_name(sec_out.stem + "_spread.csv"), index=False)
        print(f"\n=== SECONDARY robustness read (italy_secondary ONLY) -> {sec_out} ===")
        print("Objective 3 regime-shift stress test -- NOT a data-richness comparison.")
        print(gap_sec.groupby(["method", "regime"])["mase_gap_pct"].median().round(1).to_string())


if __name__ == "__main__":
    main()
