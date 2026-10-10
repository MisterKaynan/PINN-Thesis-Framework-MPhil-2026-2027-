#!/usr/bin/env python3
"""Italy daily preprocessing. Primary defaults fixed r=6; secondary defaults raw.
Restricted correction preserves NaNs and is NOT directly compatible with legacy
training runners. Dates/cutoff are inherited experiment choices, not newly
validated epidemiological claims. No illustrative time-varying ratios allowed.
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
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_csv(tmp, **kwargs)
        os.replace(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()

def load_raw(path):
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
    mode = mode or ("fixed" if arm == "primary" else "none")
    if mode == "timevarying":
        raise ValueError("timevarying disabled: provide a reviewed sourced schedule in a separate implementation")
    if mode not in {"fixed", "restrict", "none"}: raise ValueError(mode)
    start = start or (PRIMARY_START if arm == "primary" else str(raw.date.min().date()))
    end = end or (PRIMARY_END if arm == "primary" else str(raw.date.max().date()))
    if pd.Timestamp(start) > pd.Timestamp(end): raise ValueError("start exceeds end")
    selected = raw.loc[raw.date.between(pd.Timestamp(start), pd.Timestamp(end))].copy()
    if selected.empty: raise ValueError("No input rows in selected window")
    if mode == "fixed" and pd.Timestamp(end) > pd.Timestamp(valid_end):
        raise ValueError("Fixed r=6 beyond configured cutoff is blocked; use none or restrict")
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
    if mode == "none": corrected = incidence.copy()
    elif mode == "fixed": corrected = incidence * 6.0
    else: corrected = (incidence * 6.0).where(master.index <= pd.Timestamp(valid_end))
    master["daily_incidence_corrected"] = corrected
    master["correction_defined"] = corrected.notna().astype(int)
    # Preserve undefined values anywhere in a trailing window, including initial partial windows.
    rolling = corrected.rolling(PRIMARY_D, min_periods=1).sum()
    unknown = corrected.isna().astype(int).rolling(PRIMARY_D, min_periods=1).sum() > 0
    master["I_proxy"] = rolling.mask(unknown)
    master["H_proxy"] = master.new_hospitalisations.rolling(PRIMARY_D, min_periods=1).sum()
    weekly = master[["daily_incidence_corrected", "new_hospitalisations"]].resample("7D").sum(min_count=1)
    undefined_week = corrected.isna().resample("7D").max()
    weekly.loc[undefined_week, "daily_incidence_corrected"] = np.nan
    weekly.columns = ["weekly_incidence", "weekly_hospitalisations"]
    weekly.index.name = "date"
    defined = corrected.notna()
    total = float(corrected.sum()) if defined.all() else None
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
        target_basis="reported incidence" if mode == "none" else "ascertainment-adjusted incidence",
        notes=["Rolling proxies are not summed as case totals.",
               "Configured dates and correction assumptions require study-specific justification.",
               "Restricted outputs with undefined I_proxy must not be passed to legacy training runners."])
    return master, weekly, audit

def main():
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
    names = [f"{prefix}_national_master.csv", f"{prefix}_weekly.csv", f"{prefix}_millevoi_format.csv", f"{prefix}_audit_summary.json"]
    paths = [a.outdir / name for name in names]
    if not a.overwrite and any(p.exists() for p in paths):
        ap.error("Output already exists. Use a new outdir or back up then pass --overwrite")
    a.outdir.mkdir(parents=True, exist_ok=True)
    export = pd.DataFrame({"Date": master.index.strftime("%Y-%m-%d"), "I_data": master.I_proxy.values, "H_data": master.H_proxy.values})
    audit["input_sha256"] = hashlib.sha256(a.input.read_bytes()).hexdigest()
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
