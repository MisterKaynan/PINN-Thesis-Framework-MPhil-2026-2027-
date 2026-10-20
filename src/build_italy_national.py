#!/usr/bin/env python3
"""
Builds data/raw/italy_case_hosp.csv for prepare_italy.py.

UPDATED (v2): the InPhyT/ISS archive under "iss_bydate_italy_*" already
provides pre-aggregated NATIONAL daily incidence for confirmed cases and
hospitalisations -- no age/sex summation is needed. This supersedes the
v1 script that summed age x sex disaggregated files (iss_age_date_italy_*),
which is still supported below via --mode age_sex for cases where only the
disaggregated files are available.

INPUT SCHEMA (national, "bydate" files)
    date,incidence,incidence_7_days_moving_average
    e.g. iss_bydate_italy_confirmed.csv, iss_bydate_italy_hospitalizations.csv

    The "incidence" column is the raw (non-averaged) daily national count --
    this is what should feed new_infections / new_hospitalisations. The
    7-day moving average column is NOT used, since prepare_italy.py applies
    its own D=5 day trailing rolling-sum proxy on top of raw daily incidence,
    and stacking two different rolling windows would double-smooth the data.

OUTPUT COVERAGE
    Missing calendar dates are zero-filled only after duplicate-date,
    per-source gap, and merge-coverage checks. The output retains per-source
    *_source_present flags, per-measure *_observed flags, and
    source_row_present (1 only when every required source observed that date).
    The command prints the coverage audit before writing the output.

USAGE (preferred, national bydate files)
    python3 build_italy_national.py --mode bydate \\
        --confirmed iss_bydate_italy_confirmed.csv \\
        --hospitalisations iss_bydate_italy_hospitalizations.csv \\
        --out data/raw/italy_case_hosp.csv

USAGE (fallback, age x sex disaggregated files)
    python3 build_italy_national.py --mode age_sex \\
        --hosp-female iss_age_date_italy_hospitalizations_female.csv \\
        --hosp-male   iss_age_date_italy_hospitalizations_male.csv \\
        --inf-female  iss_age_date_italy_confirmed_female.csv \\
        --inf-male    iss_age_date_italy_confirmed_male.csv \\
        --out data/raw/italy_case_hosp.csv
"""
import argparse
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

AGE_COLS = ["0_5", "6_12", "13_19", "20_29", "30_39", "40_49",
            "50_59", "60_69", "70_79", "80_89", "90_+"]


def merge_with_coverage_audit(series_by_source: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Align source series, audit date coverage, then zero-fill missing values."""
    normalized = {}
    duplicate_details = []

    # Normalize timestamps to calendar days so times in source values cannot
    # create separate records for what should be the same daily observation.
    for name, source in series_by_source.items():
        current = source.copy()
        current.index = pd.DatetimeIndex(pd.to_datetime(current.index, errors="raise")).normalize()
        if current.index.isna().any():
            raise ValueError(f"{name}: source contains missing dates")

        duplicate_rows = int(current.index.duplicated(keep="first").sum())
        if duplicate_rows:
            duplicate_dates = current.index[current.index.duplicated(keep=False)].unique()
            duplicate_details.append(
                f"{name}: {duplicate_rows} extra row(s) across "
                f"{len(duplicate_dates)} duplicate date(s) "
                f"({', '.join(str(date.date()) for date in duplicate_dates)})"
            )
        normalized[name] = current

    # Duplicates would make reindexing ambiguous and can double-count totals.
    # Reject them before constructing the daily grid or imputing any values.
    if duplicate_details:
        raise ValueError("Duplicate source dates detected before zero-filling: "
                         + "; ".join(duplicate_details))
    if not normalized or any(source.empty for source in normalized.values()):
        raise ValueError("Every source must contain at least one dated observation")

    calendar_start = min(source.index.min() for source in normalized.values())
    calendar_end = max(source.index.max() for source in normalized.values())
    calendar = pd.date_range(calendar_start, calendar_end, freq="D")
    out = pd.DataFrame(index=calendar)
    out.index.name = "date"
    audit = {}

    # Compare every source with its own date span and the shared merged calendar:
    # the first gap count finds missing internal dates, the second finds dates
    # absent because another source extends the calendar farther.
    for name, source in normalized.items():
        dates = source.index
        source_calendar = pd.date_range(dates.min(), dates.max(), freq="D")
        present = calendar.isin(dates)
        audit[name] = {
            "source_rows": int(len(source)),
            "unique_dates": int(dates.nunique()),
            "date_start": str(dates.min().date()),
            "date_end": str(dates.max().date()),
            "missing_days_within_source_span": int(len(source_calendar.difference(dates))),
            "days_missing_from_merged_calendar": int((~present).sum()),
            "merged_calendar_days": int(len(calendar)),
        }
        out[name] = source.reindex(calendar)
        out[f"{name}_source_present"] = present.astype(int)

    # Joint presence means all inputs contributed a row on that date; individual
    # flags remain available to identify which source is absent.
    source_flags = [f"{name}_source_present" for name in normalized]
    out["source_row_present"] = out[source_flags].all(axis=1).astype(int)
    out.attrs["source_coverage_audit"] = {
        "merged_calendar_days": int(len(calendar)),
        "all_sources_present_days": int(out["source_row_present"].sum()),
        "days_missing_from_at_least_one_source": int((out["source_row_present"] == 0).sum()),
        "sources": audit,
    }

    # Coverage and presence flags are captured above; only now convert absent
    # source observations to zero so downstream count-based steps can proceed.
    return out.fillna(0)


def load_bydate(path: Path) -> pd.Series:
    """Read the national daily incidence field; do not use the smoothed column."""
    df = pd.read_csv(path, parse_dates=["date"])
    if "incidence" not in df.columns:
        raise ValueError(f"{path.name}: expected a column named 'incidence'")
    s = df.set_index("date")["incidence"].clip(lower=0)
    return s


def national_sum_age_sex(path: Path) -> pd.Series:
    """Sum available age groups per row while preserving dates for audit."""
    df = pd.read_csv(path, parse_dates=["date"])
    cols = [c for c in AGE_COLS if c in df.columns]
    missing = set(AGE_COLS) - set(cols)
    if missing:
        print(f"  warning: {path.name} missing age columns {missing}, summing what's present")
    # Keep one row per source record so duplicate dates remain detectable by
    # merge_with_coverage_audit before any date-based aggregation.
    return pd.Series(df[cols].sum(axis=1).to_numpy(), index=df["date"], name=path.stem)


def build_bydate(confirmed_path: Path, hosp_path: Path) -> pd.DataFrame:
    """Build output from already-national confirmed and hospitalisation series."""
    merged = merge_with_coverage_audit({
        "confirmed": load_bydate(confirmed_path),
        "hospitalisations": load_bydate(hosp_path),
    })

    # Keep training input names stable while carrying both field-level and
    # all-sources-observed flags into the generated CSV.
    out = pd.DataFrame(index=merged.index)
    out["new_infections"] = merged["confirmed"]
    out["new_hospitalisations"] = merged["hospitalisations"]
    for source in ("confirmed", "hospitalisations"):
        out[f"{source}_source_present"] = merged[f"{source}_source_present"]
    out["new_infections_observed"] = merged["confirmed_source_present"]
    out["new_hospitalisations_observed"] = merged["hospitalisations_source_present"]
    out["source_row_present"] = merged["source_row_present"]
    out.attrs["source_coverage_audit"] = merged.attrs["source_coverage_audit"]
    return out


def build_age_sex(hosp_f: Path, hosp_m: Path, inf_f: Path, inf_m: Path) -> pd.DataFrame:
    """Build national totals from four age-by-sex source files."""
    merged = merge_with_coverage_audit({
        "hospitalisations_female": national_sum_age_sex(hosp_f),
        "hospitalisations_male": national_sum_age_sex(hosp_m),
        "infections_female": national_sum_age_sex(inf_f),
        "infections_male": national_sum_age_sex(inf_m),
    })
    out = pd.DataFrame(index=merged.index)

    # Zero-filled components allow totals to be formed, but separate observed
    # flags distinguish an actual zero from a source with no row that day.
    out["new_hospitalisations"] = (
        merged["hospitalisations_female"] + merged["hospitalisations_male"]
    )
    out["new_infections"] = merged["infections_female"] + merged["infections_male"]
    for source in merged.attrs["source_coverage_audit"]["sources"]:
        out[f"{source}_source_present"] = merged[f"{source}_source_present"]
    out["new_hospitalisations_observed"] = merged[
        ["hospitalisations_female_source_present", "hospitalisations_male_source_present"]
    ].all(axis=1).astype(int)
    out["new_infections_observed"] = merged[
        ["infections_female_source_present", "infections_male_source_present"]
    ].all(axis=1).astype(int)
    out["source_row_present"] = (
        out["new_hospitalisations_observed"] & out["new_infections_observed"]
    )
    out.attrs["source_coverage_audit"] = merged.attrs["source_coverage_audit"]
    return out


def print_coverage_audit(out: pd.DataFrame) -> None:
    """Print source gaps and merged-calendar coverage captured before imputation."""
    audit = out.attrs["source_coverage_audit"]
    print("\nSource coverage audit (before zero-filling):")
    print(f"  merged calendar: {audit['merged_calendar_days']} days; "
          f"all sources present: {audit['all_sources_present_days']}; "
          f"days missing from at least one source: "
          f"{audit['days_missing_from_at_least_one_source']}")
    for name, source in audit["sources"].items():
        print(f"  {name}: rows={source['source_rows']}, "
              f"unique dates={source['unique_dates']}, "
              f"missing within source span={source['missing_days_within_source_span']}, "
              f"missing from merged calendar={source['days_missing_from_merged_calendar']} "
              f"({source['date_start']} to {source['date_end']})")


def sanity_check(out: pd.DataFrame) -> None:
    """Print a coarse totals-based check to catch likely input-file mistakes."""
    total_inf = out["new_infections"].sum()
    total_hosp = out["new_hospitalisations"].sum()
    ratio = total_inf / max(total_hosp, 1)
    print(f"\nSanity check: total infections={total_inf:,.0f}  "
          f"total hospitalisations={total_hosp:,.0f}  ratio={ratio:.2f}x")
    if total_inf < total_hosp:
        print("!! WARNING: infections < hospitalisations overall. This is "
              "epidemiologically implausible -- re-check input files before "
              "proceeding to prepare_italy.py.")
    elif ratio < 2:
        print("!! WARNING: infection/hospitalisation ratio looks low for a "
              "national COVID series -- double check file identity.")
    else:
        print("Ratio looks plausible for a national COVID-19 series.")


def main():
    """Validate mode-specific inputs, build the series, and write the audited CSV."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bydate", "age_sex"], default="bydate")
    ap.add_argument("--confirmed", type=Path)
    ap.add_argument("--hospitalisations", type=Path)
    ap.add_argument("--hosp-female", type=Path)
    ap.add_argument("--hosp-male", type=Path)
    ap.add_argument("--inf-female", type=Path)
    ap.add_argument("--inf-male", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    # Validate required paths before invoking either input-building workflow.
    if a.mode == "bydate":
        if not (a.confirmed and a.hospitalisations):
            ap.error("--mode bydate requires --confirmed and --hospitalisations")
        out = build_bydate(a.confirmed, a.hospitalisations)
    else:
        if not (a.hosp_female and a.hosp_male and a.inf_female and a.inf_male):
            ap.error("--mode age_sex requires --hosp-female/--hosp-male/--inf-female/--inf-male")
        out = build_age_sex(a.hosp_female, a.hosp_male, a.inf_female, a.inf_male)

    # Audit is attached during the merge before imputation; print it before
    # rounding and writing so zero-filled values cannot obscure source gaps.
    print_coverage_audit(out)
    out = out.round(0).astype(int)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out)

    # Summaries operate on the final integer series that downstream preprocessing
    # will read, not on intermediate per-source series.
    print(f"Wrote {a.out}  ({len(out)} days, "
          f"{out['new_infections'].sum():,} infections, "
          f"{out['new_hospitalisations'].sum():,} hospitalisations)")
    print(out.head(3).to_string())
    sanity_check(out)


if __name__ == "__main__":
    main()
