#!/usr/bin/env python3
"""Italy daily preprocessing. Primary defaults fixed r=6; secondary defaults raw.
Restricted correction uses raw incidence after the r=6 validity cutoff. Dates/
cutoff are inherited experiment choices, not newly validated epidemiological
claims. No illustrative time-varying ratios allowed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

PRIMARY_START = "2020-02-20"
PRIMARY_END = "2020-05-20"
R6_VALID_END = "2020-08-31"
PRIMARY_D = 5


def atomic_csv(df, path, **kwargs):
    """Write a CSV through a temporary file so interrupted writes do not leave partial output."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_csv(tmp, **kwargs)
        os.replace(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()

def load_raw(path):
    """Load the required source fields and reject data needing manual resolution."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    required = ["date", "new_infections", "new_hospitalisations"]
    if not set(required).issubset(df.columns):
        raise ValueError(f"Required columns: {required}")
    df = df[required].copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    if df["date"].isna().any(): raise ValueError("Missing dates")
    if (df["date"] != df["date"].dt.normalize()).any():
        raise ValueError("Expected daily dates without time components")
    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates: aggregate explicitly before preprocessing")
    for col in required[1:]:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col]).all(): raise ValueError(f"Non-finite source values: {col}")
        if (df[col] < 0).any(): raise ValueError(f"Negative revisions in {col}; resolve and audit first")
    return df.sort_values("date").reset_index(drop=True)

def process(raw, arm, start=None, end=None, mode=None, missing_days="error", valid_end=R6_VALID_END):
    """Build daily, weekly, and audit outputs for one Italy experiment arm."""
    # Defaults intentionally differ by arm: primary uses the fixed study window
    # and correction; secondary uses all available dates and reported counts.
    mode = mode or ("fixed" if arm == "primary" else "none")
    if mode == "timevarying":
        raise ValueError("timevarying disabled: provide a reviewed sourced schedule in a separate implementation")
    if mode not in {"fixed", "restrict", "none"}: raise ValueError(mode)
    start = start or (PRIMARY_START if arm == "primary" else str(raw.date.min().date()))
    end = end or (PRIMARY_END if arm == "primary" else str(raw.date.max().date()))
    if pd.Timestamp(start) > pd.Timestamp(end): raise ValueError("start exceeds end")
    selected = raw.loc[raw.date.between(pd.Timestamp(start), pd.Timestamp(end))].copy()
    if selected.empty: raise ValueError("No input rows in selected window")
    # A fixed correction outside its declared validity window is a configuration
    # error. "restrict" instead switches to reported incidence after valid_end.
    if mode == "fixed" and pd.Timestamp(end) > pd.Timestamp(valid_end):
        raise ValueError("Fixed r=6 beyond configured cutoff is blocked; use none or restrict")

    # Reindex onto a complete daily calendar. Missing source dates are errors
    # unless the caller explicitly opts into zero imputation; presence is kept
    # separately so imputed dates remain visible in the output and audit.
    grid = pd.date_range(start, end, freq="D")
    missing = grid.difference(selected.date)
    if len(missing) and missing_days == "error":
        raise ValueError(f"{len(missing)} missing daily rows; first: {missing[0]}. Explicit --missing-days zero required for zero imputation")
    master = selected.set_index("date").reindex(grid)
    master.index.name = "date"
    master["source_row_present"] = master.index.isin(selected.date).astype(int)
    if len(missing):
        master.loc[missing, ["new_infections", "new_hospitalisations"]] = 0.0
    master["country"] = f"italy_{arm}"
    incidence = master["new_infections"].astype(float)

    # "restrict" is piecewise by design: scaled incidence through the cutoff,
    # then raw reported incidence. It never uses zero to stand for uncertainty.
    if mode == "none":
        corrected = incidence.copy()
    elif mode == "fixed":
        corrected = incidence * 6.0
    else:
        # Apply r=6 only while it is considered valid; keep observed raw
        # incidence afterward rather than representing an unknown correction
        # as NaN (which downstream zero-filling could misread as no cases).
        within_validity = master.index <= pd.Timestamp(valid_end)
        corrected = (incidence * 6.0).where(within_validity, incidence)
    master["daily_incidence_corrected"] = corrected
    master["correction_defined"] = corrected.notna().astype(int)

    # Incidence proxy is a trailing D-day sum. If any input in its window is
    # undefined, keep the proxy undefined instead of silently summing partial data.
    rolling = corrected.rolling(PRIMARY_D, min_periods=1).sum()
    unknown = corrected.isna().astype(int).rolling(PRIMARY_D, min_periods=1).sum() > 0
    master["I_proxy"] = rolling.mask(unknown)
    master["H_proxy"] = master.new_hospitalisations.rolling(PRIMARY_D, min_periods=1).sum()

    # Weekly incidence sums require at least one defined value and are marked
    # unknown if any corrected daily value in that week is undefined.
    weekly = master[["daily_incidence_corrected", "new_hospitalisations"]].resample("7D").sum(min_count=1)
    undefined_week = corrected.isna().resample("7D").max()
    weekly.loc[undefined_week, "daily_incidence_corrected"] = np.nan
    weekly.columns = ["weekly_incidence", "weekly_hospitalisations"]
    weekly.index.name = "date"
    defined = corrected.notna()
    total = float(corrected.sum()) if defined.all() else None

    # The audit distinguishes complete corrected totals from defined-only sums
    # and records the correction basis, missing-day policy, and training status.
    audit = dict(version=3, arm=arm, country=f"italy_{arm}", date_start=str(grid[0].date()),
        date_end=str(grid[-1].date()), n_days=len(master), n_weeks=len(weekly),
        ascertainment_mode=mode, ratio=1.0 if mode == "none" else 6.0,
        valid_until=valid_end if mode == "restrict" else None,
        total_cases_raw=float(incidence.sum()), total_cases_corrected=total,
        total_cases_corrected_defined_only=float(corrected.sum()) if defined.any() else None,
        corrected_undefined_days=int((~defined).sum()), proxy_undefined_days=int(master.I_proxy.isna().sum()),
        total_hospitalisations=float(master.new_hospitalisations.sum()), primary_infectious_period_D=PRIMARY_D,
        missing_days_policy=missing_days, imputed_zero_days=len(missing),
        training_ready=bool(np.isfinite(master.I_proxy).all() and master.I_proxy.max() > 0),
        target_basis=("reported incidence" if mode == "none" else
                      f"r=6-adjusted through {valid_end}; reported incidence afterward"
                      if mode == "restrict" else "ascertainment-adjusted incidence"),
        notes=["Restricted mode uses raw reported incidence after valid_until; "
               "unknown correction is never represented as zero.",
               "Rolling proxies are not summed as case totals.",
               "Configured dates and correction assumptions require study-specific justification.",
               "Zero imputation applies only to dates missing from the source and "
               "only when --missing-days zero is explicitly selected."])
    return master, weekly, audit


def main():
    # Parse CLI settings, then turn expected input/configuration failures into
    # standard argparse errors before creating any output files.
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--arm", choices=["primary", "secondary"], default="primary")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--ascertainment", choices=["fixed", "restrict", "none", "timevarying"])
    ap.add_argument("--valid-until", default=R6_VALID_END)
    ap.add_argument("--missing-days", choices=["error", "zero"], default="error")
    ap.add_argument("--overwrite", action="store_true", help="Explicitly replace existing arm outputs; back them up first")
    a = ap.parse_args()
    try:
        master, weekly, audit = process(load_raw(a.input), a.arm, a.start, a.end,
            a.ascertainment, a.missing_days, a.valid_until)
    except (ValueError, OSError) as exc: ap.error(str(exc))
    prefix = f"italy_{a.arm}"

    # Refuse accidental replacement of a partial or complete output set unless
    # overwrite was explicitly requested.
    names = [f"{prefix}_national_master.csv", f"{prefix}_weekly.csv", f"{prefix}_millevoi_format.csv", f"{prefix}_audit_summary.json"]
    paths = [a.outdir / name for name in names]
    if not a.overwrite and any(p.exists() for p in paths):
        ap.error("Output already exists. Use a new outdir or back up then pass --overwrite")
    a.outdir.mkdir(parents=True, exist_ok=True)

    # Export the legacy runner's expected I/H columns while retaining the richer
    # master and weekly products for reproducibility and later review.
    export = pd.DataFrame({"Date": master.index.strftime("%Y-%m-%d"), "I_data": master.I_proxy.values, "H_data": master.H_proxy.values})
    audit["input_sha256"] = hashlib.sha256(a.input.read_bytes()).hexdigest()

    # Write the three CSV products atomically and publish the audit after adding
    # the source-file hash, so outputs can be traced to their exact input.
    atomic_csv(master, paths[0])
    atomic_csv(weekly, paths[1])
    atomic_csv(export, paths[2], index=False)
    tmp = paths[3].with_suffix(".json.tmp")
    tmp.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, paths[3])
    print(json.dumps(audit, indent=2, allow_nan=False))
    if not audit["training_ready"]:
        print("WARNING: Output is NOT training-ready. Undefined target values were preserved, not zero-filled.")

if __name__ == "__main__":
    main()
