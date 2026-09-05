#!/usr/bin/env python3
"""
Italy ISS preprocessing, mirroring prepare_ghana.py's structure and design-
decision documentation so both countries feed the same PINN reader.

DESIGN DECISIONS
1. Input is assumed to already be national daily incidence and hospitalisation
   counts (ISS integrated surveillance style), not a Civil-Protection-style
   cumulative/regional file. If starting from raw DPC data, aggregate to
   national level and difference to incidence BEFORE this script.
2. Infections are scaled by the ascertainment ratio r=6 (Millevoi et al. 2024,
   Sec on real-data case, citing ISTAT serological estimates) to match the
   I_data convention used by the reference PINN and the course project.
3. Hospitalisation series is retained natively -- this is the key structural
   asymmetry with Ghana that the SEIQHRS synthesis (synth_hospitalisation.py)
   is designed to approximate on the Ghana side.
4. A D=5 day trailing rolling-sum infection proxy is used, identical window to
   Ghana's I_proxy, for architectural parity across countries.
5. Millevoi-format export mirrors ghana_millevoi_format.csv's column layout.

Usage: python3 prepare_italy.py --input <csv> --outdir <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ASCERTAINMENT_RATIO = 6.0   # Millevoi et al. (2024), ISTAT serological estimate
PRIMARY_D = 5


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    for c in ("new_infections", "new_hospitalisations"):
        if c not in df.columns:
            raise ValueError(f"expected column '{c}' in Italy raw input")
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df.sort_values("date").reset_index(drop=True)


def build_national(df: pd.DataFrame) -> pd.DataFrame:
    d = df.set_index("date")
    full = pd.date_range(d.index.min(), d.index.max(), freq="D")
    d = d.reindex(full).fillna(0)
    d.index.name = "date"
    d["daily_incidence_corrected"] = d["new_infections"] * ASCERTAINMENT_RATIO
    return d


def add_proxies(nat: pd.DataFrame) -> pd.DataFrame:
    out = nat.copy()
    out["I_proxy"] = (
        out["daily_incidence_corrected"].rolling(PRIMARY_D, min_periods=1).sum()
    )
    out["H_proxy"] = out["new_hospitalisations"].rolling(PRIMARY_D, min_periods=1).sum()
    return out


def weekly_aggregate(nat: pd.DataFrame) -> pd.DataFrame:
    wk = nat[["daily_incidence_corrected", "new_hospitalisations"]].resample("7D").sum()
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
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(args.input)
    nat = build_national(raw)
    master = add_proxies(nat)
    weekly = weekly_aggregate(nat)

    master.to_csv(args.outdir / "italy_national_master.csv")
    weekly.to_csv(args.outdir / "italy_weekly.csv")
    millevoi_export(master).to_csv(args.outdir / "italy_millevoi_format.csv", index=False)

    summary = {
        "date_start": str(master.index.min().date()),
        "date_end": str(master.index.max().date()),
        "n_days": len(master),
        "total_cases_raw": float(raw["new_infections"].sum()),
        "total_cases_corrected": float(master["daily_incidence_corrected"].sum()),
        "total_hospitalisations": float(master["new_hospitalisations"].sum()),
        "ascertainment_ratio": ASCERTAINMENT_RATIO,
        "primary_infectious_period_D": PRIMARY_D,
        "n_weeks": len(weekly),
    }
    (args.outdir / "italy_audit_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
