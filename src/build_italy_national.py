#!/usr/bin/env python3
"""
Builds data/raw/italy_case_hosp.csv for prepare_italy.py.

REVISED (v2): the InPhyT/ISS archive under "iss_bydate_italy_*" already
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
from pathlib import Path

import pandas as pd

AGE_COLS = ["0_5", "6_12", "13_19", "20_29", "30_39", "40_49",
            "50_59", "60_69", "70_79", "80_89", "90_+"]


def load_bydate(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"])
    if "incidence" not in df.columns:
        raise ValueError(f"{path.name}: expected a column named 'incidence'")
    s = df.set_index("date")["incidence"].clip(lower=0)
    return s


def national_sum_age_sex(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"])
    cols = [c for c in AGE_COLS if c in df.columns]
    missing = set(AGE_COLS) - set(cols)
    if missing:
        print(f"  warning: {path.name} missing age columns {missing}, summing what's present")
    return df.set_index("date")[cols].sum(axis=1)


def build_bydate(confirmed_path: Path, hosp_path: Path) -> pd.DataFrame:
    inf = load_bydate(confirmed_path).rename("new_infections")
    hosp = load_bydate(hosp_path).rename("new_hospitalisations")
    out = pd.concat([inf, hosp], axis=1)
    out = out.reindex(pd.date_range(out.index.min(), out.index.max(), freq="D"))
    out.index.name = "date"
    return out.fillna(0)


def build_age_sex(hosp_f: Path, hosp_m: Path, inf_f: Path, inf_m: Path) -> pd.DataFrame:
    hf, hm = national_sum_age_sex(hosp_f), national_sum_age_sex(hosp_m)
    inf_female, inf_male = national_sum_age_sex(inf_f), national_sum_age_sex(inf_m)
    hosp_total = hf.add(hm, fill_value=0).rename("new_hospitalisations")
    inf_total = inf_female.add(inf_male, fill_value=0).rename("new_infections")
    out = pd.concat([inf_total, hosp_total], axis=1)
    out = out.reindex(pd.date_range(out.index.min(), out.index.max(), freq="D"))
    out.index.name = "date"
    return out.fillna(0)


def sanity_check(out: pd.DataFrame) -> None:
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

    if a.mode == "bydate":
        if not (a.confirmed and a.hospitalisations):
            ap.error("--mode bydate requires --confirmed and --hospitalisations")
        out = build_bydate(a.confirmed, a.hospitalisations)
    else:
        if not (a.hosp_female and a.hosp_male and a.inf_female and a.inf_male):
            ap.error("--mode age_sex requires --hosp-female/--hosp-male/--inf-female/--inf-male")
        out = build_age_sex(a.hosp_female, a.hosp_male, a.inf_female, a.inf_male)

    out = out.round(0).astype(int)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out)

    print(f"Wrote {a.out}  ({len(out)} days, "
          f"{out['new_infections'].sum():,} infections, "
          f"{out['new_hospitalisations'].sum():,} hospitalisations)")
    print(out.head(3).to_string())
    sanity_check(out)


if __name__ == "__main__":
    main()
