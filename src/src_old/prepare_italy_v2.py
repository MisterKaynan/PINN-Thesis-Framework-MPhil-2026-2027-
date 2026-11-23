#!/usr/bin/env python3
"""
Italy ISS preprocessing -- v2, TWO-ARM design.

Produces a PRIMARY single-wave benchmark (r=6-valid, Millevoi-comparable)
and a SECONDARY full-range robustness arm (regime-shift stress test), from
the same raw data/raw/italy_case_hosp.csv built by build_italy_national.py.

WHY TWO ARMS (see thesis discussion: two-tier Italy design)
    - PRIMARY: a single-wave window where the r=6 ascertainment ratio
      (Millevoi et al. 2024, citing ISTAT serological estimates from the
      first 2020 wave) is actually valid. This is the "clean" high-
      information benchmark used for Objectives 1 and 2 -- core split-PINN
      validation and Ghana comparability claims.
    - SECONDARY: the full multi-wave range (2020-2022), explicitly labelled
      as a regime-shift robustness stress test (Objective 3 only), NOT a
      clean data-richness comparison. The flat r=6 correction is either
      dropped, restricted to its valid sub-window, or replaced by a
      declared time-varying schedule -- never silently applied across
      variant eras and vaccination periods where it has no evidential basis.

USAGE
    # Primary arm (default): single-wave window, r=6 throughout
    python3 prepare_italy.py --arm primary \\
        --input data/raw/italy_case_hosp.csv --outdir data/processed

    # Secondary arm: full range, ascertainment mode selectable
    python3 prepare_italy.py --arm secondary --ascertainment restrict \\
        --input data/raw/italy_case_hosp.csv --outdir data/processed
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PRIMARY_D = 5
PRIMARY_START = "2020-02-20"   # first ISS-reported day of Italy's wave 1
PRIMARY_END = "2020-05-20"     # matches Millevoi et al.'s 90-day case window
R6_VALID_END = "2020-08-31"    # ISTAT serological estimate is defensible only
                                 # through the tail of wave 1 / pre-wave-2 lull

# Declared, citation-flagged wave-period ratios for --ascertainment timevarying.
# These are ILLUSTRATIVE placeholders reflecting the widely-reported qualitative
# trend (ascertainment IMPROVED after mass testing scale-up in wave 2/3, then
# DEGRADED again once home antigen self-testing became dominant in 2022) --
# replace with values traced to a specific published estimate before using in
# the thesis; do not present these numbers as validated ISTAT figures.
WAVE_RATIO_SCHEDULE = [
    ("2020-01-01", "2020-08-31", 6.0),   # Millevoi/ISTAT wave-1 estimate
    ("2020-09-01", "2021-06-30", 3.0),   # PLACEHOLDER: wave 2/3, more testing capacity
    ("2021-07-01", "2022-10-23", 8.0),   # PLACEHOLDER: Omicron + home-testing under-capture
]


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    for c in ("new_infections", "new_hospitalisations"):
        if c not in df.columns:
            raise ValueError(f"expected column '{c}' in Italy raw input")
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df.sort_values("date").reset_index(drop=True)


def slice_window(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["date"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)


def apply_ascertainment(df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, dict]:
    """Returns (df_with_corrected_column, meta_dict_describing_choice)."""
    df = df.copy()
    if mode == "fixed":
        df["daily_incidence_corrected"] = df["new_infections"] * 6.0
        meta = {"ascertainment_mode": "fixed", "ratio": 6.0,
                "warning": "Flat r=6 applied across full range -- valid only "
                           "for the 2020 wave-1 window; interpret results "
                           "outside that window with caution."}
    elif mode == "restrict":
        valid = df["date"] <= pd.Timestamp(R6_VALID_END)
        df["daily_incidence_corrected"] = np.where(
            valid, df["new_infections"] * 6.0, df["new_infections"] * np.nan
        )
        meta = {"ascertainment_mode": "restrict", "ratio": 6.0,
                "valid_until": R6_VALID_END,
                "note": "Corrected infections are NaN beyond the r=6-valid "
                        "window; raw new_infections retained for that period "
                        "instead. Use raw counts, not corrected, past this date."}
    elif mode == "timevarying":
        ratio = pd.Series(np.nan, index=df.index)
        for start, end, r in WAVE_RATIO_SCHEDULE:
            mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
            ratio[mask] = r
        df["ascertainment_ratio_used"] = ratio
        df["daily_incidence_corrected"] = df["new_infections"] * ratio
        meta = {"ascertainment_mode": "timevarying",
                "schedule": WAVE_RATIO_SCHEDULE,
                "warning": "Wave-period ratios beyond 2020 are PLACEHOLDERS "
                           "pending a cited source -- replace before "
                           "reporting thesis results."}
    elif mode == "none":
        df["daily_incidence_corrected"] = df["new_infections"].astype(float)
        meta = {"ascertainment_mode": "none",
                "note": "No ascertainment correction applied; raw counts used as-is."}
    else:
        raise ValueError(mode)
    return df, meta


def add_proxies(nat: pd.DataFrame) -> pd.DataFrame:
    out = nat.copy()
    out["I_proxy"] = out["daily_incidence_corrected"].rolling(PRIMARY_D, min_periods=1).sum()
    out["H_proxy"] = out["new_hospitalisations"].rolling(PRIMARY_D, min_periods=1).sum()
    return out


def weekly_aggregate(nat: pd.DataFrame) -> pd.DataFrame:
    d = nat.set_index("date")
    wk = d[["daily_incidence_corrected", "new_hospitalisations"]].resample("7D").sum()
    wk.columns = ["weekly_incidence", "weekly_hospitalisations"]
    return wk


def millevoi_export(master: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Date": master.index.strftime("%Y-%m-%d"),
        "I_data": master["I_proxy"].values,
        "H_data": master["H_proxy"].values,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--arm", choices=["primary", "secondary"], default="primary")
    ap.add_argument("--start", default=None, help="override window start (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="override window end (YYYY-MM-DD)")
    ap.add_argument("--ascertainment", choices=["fixed", "restrict", "timevarying", "none"],
                    default=None, help="ascertainment mode; default depends on --arm")
    args = ap.parse_args()

    raw = load_raw(args.input)

    if args.arm == "primary":
        start = args.start or PRIMARY_START
        end = args.end or PRIMARY_END
        asc_mode = args.ascertainment or "fixed"
        prefix = "italy_primary"
    else:
        start = args.start  # None -> full range
        end = args.end
        asc_mode = args.ascertainment or "restrict"
        prefix = "italy_secondary"

    sliced = slice_window(raw, start, end)
    if sliced.empty:
        raise SystemExit(f"No rows in window start={start} end={end} -- check dates")

    corrected, asc_meta = apply_ascertainment(sliced, asc_mode)
    nat = corrected.set_index("date")
    full_idx = pd.date_range(nat.index.min(), nat.index.max(), freq="D")
    nat = nat.reindex(full_idx).fillna(0)
    nat.index.name = "date"

    master = add_proxies(nat)
    weekly = weekly_aggregate(nat.reset_index())

    args.outdir.mkdir(parents=True, exist_ok=True)
    master.to_csv(args.outdir / f"{prefix}_national_master.csv")
    weekly.to_csv(args.outdir / f"{prefix}_weekly.csv", index=False)
    millevoi_export(master).to_csv(args.outdir / f"{prefix}_millevoi_format.csv", index=False)

    summary = {
        "arm": args.arm,
        "date_start": str(master.index.min().date()),
        "date_end": str(master.index.max().date()),
        "n_days": len(master),
        "total_cases_raw": float(sliced["new_infections"].sum()),
        "total_cases_corrected": float(np.nansum(master["I_proxy"])),
        "total_hospitalisations": float(sliced["new_hospitalisations"].sum()),
        "primary_infectious_period_D": PRIMARY_D,
        "n_weeks": len(weekly),
        **asc_meta,
    }
    (args.outdir / f"{prefix}_audit_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
