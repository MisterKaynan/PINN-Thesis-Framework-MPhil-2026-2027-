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
import json
from pathlib import Path

import numpy as np
import pandas as pd

PRIMARY_COUNTRIES = {"ghana", "italy_primary"}
MASE_DENOM_EPS = 1e-8


def compute_mase_gap(df: pd.DataFrame, dense_regime: str = "daily") -> pd.DataFrame:
    """Compare each sparse MASE to its paired dense fit at the same run keys."""
    pinn = df[df.method.str.startswith("pinn")].copy()
    pinn["mase"] = pd.to_numeric(pinn["mase"], errors="coerce")
    pinn = pinn[np.isfinite(pinn["mase"]) & (pinn["mase"] >= 0)]
    key = ["country", "method", "origin", "seed"]
    dense = pinn[pinn.regime == dense_regime].set_index(key)["mase"]
    rows = []
    for regime in pinn.regime.unique():
        if regime == dense_regime:
            continue
        sub = pinn[pinn.regime == regime].set_index(["country", "method", "origin", "seed"])["mase"]
        joined = pd.concat([sub.rename("mase_sparse"), dense.rename("mase_dense")], axis=1).dropna()
        joined["mase_gap"] = joined["mase_sparse"] - joined["mase_dense"]
        # Absolute gap remains interpretable when dense MASE is zero. The
        # epsilon-stabilized log ratio preserves direction without division by
        # a near-zero dense value; percent change is undefined near zero.
        joined["mase_log_ratio"] = np.log(
            (joined["mase_sparse"] + MASE_DENOM_EPS)
            / (joined["mase_dense"] + MASE_DENOM_EPS)
        )
        safe_dense = joined["mase_dense"] > MASE_DENOM_EPS
        joined["mase_gap_pct"] = np.nan
        joined.loc[safe_dense, "mase_gap_pct"] = (
            100 * joined.loc[safe_dense, "mase_gap"]
            / joined.loc[safe_dense, "mase_dense"]
        )
        joined["regime"] = regime
        rows.append(joined.reset_index())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def restrict_to_common_origins(ghana: pd.DataFrame,
                               italy_primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """Keep only origins with PINN rows in both countries' core-comparison grids."""
    ghana_pinn = ghana[ghana["method"].astype(str).str.startswith("pinn")]
    italy_pinn = italy_primary[italy_primary["method"].astype(str).str.startswith("pinn")]
    ghana_origins = set(ghana_pinn["origin"].dropna().astype(int))
    italy_origins = set(italy_pinn["origin"].dropna().astype(int))
    common_origins: list[int] = sorted(ghana_origins & italy_origins)
    if not common_origins:
        raise ValueError(
            "No common origins between Ghana and Italy-primary grids; "
            "cannot make a paired core comparison."
        )

    ghana_common = ghana.loc[ghana["origin"].isin(common_origins)].copy()
    italy_common = italy_primary.loc[italy_primary["origin"].isin(common_origins)].copy()
    return ghana_common, italy_common, common_origins


def audit_primary_comparison(combined: pd.DataFrame, gap: pd.DataFrame,
                             spread: pd.DataFrame, common_origins: list[int]) -> dict:
    """Assert primary outputs are paired, arm-clean, and numerically consistent."""
    if set(combined["country"].unique()) != PRIMARY_COUNTRIES:
        raise ValueError("primary comparison must contain exactly Ghana and Italy-primary")

    pinn = combined[combined["method"].astype(str).str.startswith("pinn")].copy()
    pinn["mase"] = pd.to_numeric(pinn["mase"], errors="coerce")
    pinn = pinn[np.isfinite(pinn["mase"]) & (pinn["mase"] >= 0)]
    keys = ["country", "method", "origin", "regime", "seed"]
    if pinn.duplicated(keys).any():
        raise ValueError("duplicate PINN rows found for a country/method/origin/regime/seed")
    if not set(pinn["origin"].dropna().astype(int)).issubset(common_origins):
        raise ValueError("PINN rows contain origins outside the common-origin set")
    if gap.empty:
        raise ValueError("no paired dense/sparse PINN rows to audit")
    if not set(gap["country"].unique()).issubset(PRIMARY_COUNTRIES):
        raise ValueError("gap output contains a country outside the primary comparison")
    if not set(gap["origin"].astype(int)).issubset(common_origins):
        raise ValueError("gap output contains origins outside the common-origin set")

    # Recompute each gap from its sparse and dense observations to catch bad or
    # mismatched merges; the seed and method must match as well as country/origin.
    key = ["country", "method", "origin", "seed"]
    dense = pinn[pinn["regime"] == "daily"][key + ["mase"]].rename(
        columns={"mase": "mase_dense"}
    )
    sparse = pinn[pinn["regime"] != "daily"][key + ["regime", "mase"]].rename(
        columns={"mase": "mase_sparse"}
    )
    verified = sparse.merge(dense, on=key, how="inner", validate="many_to_one")
    compare_keys = key + ["regime"]
    expected = verified.set_index(compare_keys)
    actual = gap.set_index(compare_keys)
    if not actual.index.isin(expected.index).all():
        raise ValueError("gap output contains rows without matching dense/sparse source fits")
    expected_rows = expected.reindex(actual.index)
    if not np.allclose(
        actual["mase_gap"].to_numpy(dtype=float),
        (expected_rows["mase_sparse"] - expected_rows["mase_dense"]).to_numpy(dtype=float),
        rtol=1e-10, atol=1e-12,
    ):
        raise ValueError("absolute MASE gaps do not match their source rows")
    dense_values = expected_rows["mase_dense"].to_numpy(dtype=float)
    sparse_values = expected_rows["mase_sparse"].to_numpy(dtype=float)
    expected_pct = np.full(len(actual), np.nan)
    safe = dense_values > MASE_DENOM_EPS
    expected_pct[safe] = 100 * (sparse_values[safe] - dense_values[safe]) / dense_values[safe]
    if not np.allclose(
        actual["mase_gap_pct"].to_numpy(dtype=float), expected_pct,
        rtol=1e-10, atol=1e-12, equal_nan=True,
    ):
        raise ValueError("percentage MASE gaps violate the denominator safeguard")
    expected_log = np.log(
        (sparse_values + MASE_DENOM_EPS) / (dense_values + MASE_DENOM_EPS)
    )
    if not np.isfinite(actual["mase_log_ratio"].to_numpy(dtype=float)).all() or not np.allclose(
        actual["mase_log_ratio"].to_numpy(dtype=float),
        expected_log, rtol=1e-10, atol=1e-12,
    ):
        raise ValueError("log MASE ratios are missing, non-finite, or inconsistent")

    if not set(spread["country"].unique()).issubset(PRIMARY_COUNTRIES):
        raise ValueError("spread output contains a country outside the primary comparison")
    if not set(spread["origin"].astype(int)).issubset(common_origins):
        raise ValueError("spread output contains origins outside the common-origin set")
    return {
        "audit_status": "passed",
        "common_origins": common_origins,
        "primary_countries": sorted(PRIMARY_COUNTRIES),
        "valid_pinn_input_rows": int(len(pinn)),
        "paired_dense_sparse_rows": int(len(gap)),
        "spread_rows": int(len(spread)),
        "undefined_percent_gap_rows": int(gap["mase_gap_pct"].isna().sum()),
        "duplicate_pinn_keys": 0,
        "gap_values_verified": True,
        "denominator_safeguard_verified": True,
        "log_ratio_values_verified": True,
    }


def compute_ensemble_spread(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize seed-to-seed MASE spread for each country/run condition."""
    pinn = df[df.method.str.startswith("pinn")]
    g = pinn.groupby(["country", "method", "regime", "origin"])["mase"]
    out = g.agg(median="median", q25=lambda s: s.quantile(0.25),
                q75=lambda s: s.quantile(0.75)).reset_index()
    out["iqr"] = out["q75"] - out["q25"]
    return out


def main():
    """Load country grids, apply paired-origin rules, then write reviewable outputs."""
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

    allowed_italy = {"italy_primary", "italy_secondary"}
    unexpected_italy = set(italy["country"].dropna().unique()) - allowed_italy
    if unexpected_italy:
        raise SystemExit(
            f"Unexpected country labels in --italy-grid: {sorted(unexpected_italy)}"
        )
    has_secondary = (italy["country"] == "italy_secondary").any()
    if has_secondary and not a.include_secondary:
        n_sec = (italy["country"] == "italy_secondary").sum()
        print(f"NOTE: {n_sec} italy_secondary rows found in --italy-grid but "
              f"--include-secondary was not passed. These rows are EXCLUDED "
              f"from equity_summary.csv (primary comparison uses ghana + "
              f"italy_primary only). Pass --include-secondary for a separate, "
              f"labelled robustness read.")

    # Exclude secondary Italy, then pair the core comparison on shared origins
    # so differences do not reflect mismatched training-window lengths.
    ghana_primary = ghana[ghana["country"] == "ghana"].copy()
    italy_primary = italy[italy["country"] == "italy_primary"].copy()
    try:
        ghana_primary, italy_primary, common_origins = restrict_to_common_origins(
            ghana_primary, italy_primary
        )
    except ValueError as exc:
        ap.error(str(exc))
    combined_primary = pd.concat(
        [ghana_primary, italy_primary], ignore_index=True
    )
    combined_primary = combined_primary[
        combined_primary["country"].isin(PRIMARY_COUNTRIES)
    ]

    gap = compute_mase_gap(combined_primary)
    spread = compute_ensemble_spread(combined_primary)
    if gap.empty:
        ap.error(
            "No paired PINN dense/sparse MASE rows remain at the common origins. "
            "Check that both grids contain matching method, seed, and regime rows."
        )

    # Run hard checks before writing output so a failed audit cannot leave
    # convincing-looking but invalid comparison tables behind.
    try:
        audit = audit_primary_comparison(
            combined_primary, gap, spread, common_origins
        )
    except ValueError as exc:
        ap.error(f"Primary comparison audit failed: {exc}")

    # Persist result tables and a companion audit record with pairing details.
    a.out.parent.mkdir(parents=True, exist_ok=True)
    gap.to_csv(a.out, index=False)
    spread.to_csv(a.out.with_name(a.out.stem + "_spread.csv"), index=False)
    audit["ghana_rows_after_origin_filter"] = int(len(ghana_primary))
    audit["italy_primary_rows_after_origin_filter"] = int(len(italy_primary))
    audit["mase_denominator_epsilon"] = MASE_DENOM_EPS
    audit["percent_gap_policy"] = (
        "NaN when dense MASE <= epsilon; use absolute MASE gap or "
        "epsilon-stabilized log ratio for comparison."
    )
    audit_path = a.out.with_name(a.out.stem + "_audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")

    print("=== PRIMARY equity comparison (ghana vs italy_primary) ===")
    print(f"Common origins retained: {common_origins}")
    print("Median MASE gap (%) by country x regime x method:")
    print(gap.groupby(["country", "method", "regime"])["mase_gap_pct"]
             .median().round(1).to_string())
    print("\nMedian absolute MASE gap:")
    print(gap.groupby(["country", "method", "regime"])["mase_gap"]
             .median().round(3).to_string())
    print("\nMedian log MASE ratio:")
    print(gap.groupby(["country", "method", "regime"])["mase_log_ratio"]
             .median().round(3).to_string())
    print(f"\nAudit: {audit_path} ({audit['audit_status']})")
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
