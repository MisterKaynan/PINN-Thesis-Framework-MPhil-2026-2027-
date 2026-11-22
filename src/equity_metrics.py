#!/usr/bin/env python3
"""Paired degradation metrics; primary and secondary outputs remain separate."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

KEY = ["country", "method", "origin", "seed"]

def load_grid(path, country=None, allow_legacy_italy=False):
    df = pd.read_csv(path)
    required = {"method", "regime", "origin", "seed", "mase"}
    if not required.issubset(df):
        raise ValueError(f"{path}: missing {sorted(required-set(df))}")
    if "country" not in df:
        if country is None:
            raise ValueError(f"{path}: country column required")
        df["country"] = country
    if df.country.isna().any():
        raise ValueError(f"{path}: missing country labels")
    if allow_legacy_italy and country in {"italy_primary", "italy_secondary"} and set(df.country) == {"italy"}:
        print(f"EXPLICIT legacy migration: {path}: italy -> {country}")
        df["country"] = country
    allowed = {country} if country else {"italy_primary", "italy_secondary"}
    if not set(df.country).issubset(allowed):
        raise ValueError(f"{path}: unexpected labels {set(df.country)-allowed}")
    df = df[df.method.astype(str).str.startswith("pinn")].copy()
    if df.empty:
        raise ValueError(f"{path}: no PINN results")
    for c in ["origin", "seed", "mase"]:
        df[c] = pd.to_numeric(df[c], errors="raise")
        if not np.isfinite(df[c]).all():
            raise ValueError(f"{path}: non-finite {c}")
    if (df.mase < 0).any():
        raise ValueError("Negative MASE")
    if df.duplicated(KEY+["regime"]).any():
        raise ValueError(f"{path}: duplicate fit keys; select a single experiment/run, do not silently average")
    return df

def compute_mase_gap(df, dense_regime="daily"):
    df = df[df.method.astype(str).str.startswith("pinn")].copy()
    if df.duplicated(KEY+["regime"]).any():
        raise ValueError("Duplicate fits in metrics input")
    dense = df[df.regime == dense_regime][KEY+["mase"]].rename(columns={"mase":"mase_dense"})
    sparse = df[df.regime != dense_regime][KEY+["regime", "mase"]].rename(columns={"mase":"mase_sparse"})
    joined = sparse.merge(dense, on=KEY, how="left", validate="many_to_one", indicator=True)
    missing = int((joined._merge != "both").sum())
    if missing:
        print(f"WARNING: {missing} sparse fits have no matching daily fit; excluded")
    joined = joined[joined._merge == "both"].drop(columns="_merge")
    joined["mase_gap"] = joined.mase_sparse - joined.mase_dense
    joined["mase_gap_pct"] = 100*joined.mase_gap / joined.mase_dense.where(joined.mase_dense > 0)
    if (joined.mase_dense == 0).any():
        print("WARNING: daily MASE zero; percentage gap undefined, absolute gap retained")
    return joined

def compute_ensemble_spread(df):
    out = df.groupby(["country","method","regime","origin"]).mase.agg(
        median="median", q25=lambda x:x.quantile(.25), q75=lambda x:x.quantile(.75), n_seeds="count").reset_index()
    out["iqr"] = out.q75-out.q25
    return out

def write_metrics(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    gap = compute_mase_gap(df)
    gap.to_csv(path, index=False)
    compute_ensemble_spread(df).to_csv(path.with_name(path.stem+"_spread.csv"), index=False)
    return gap

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid",required=True,type=Path)
    ap.add_argument("--italy-grid",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--include-secondary",action="store_true")
    a=ap.parse_args()
    gh=load_grid(a.grid,"ghana"); it=load_grid(a.italy_grid)
    primary=it[it.country=="italy_primary"]
    if primary.empty: raise ValueError("No Italy primary fits supplied")
    write_metrics(pd.concat([gh,primary],ignore_index=True),a.out)
    sec=it[it.country=="italy_secondary"]
    if a.include_secondary:
        if sec.empty: raise ValueError("--include-secondary requested but no secondary fits supplied")
        write_metrics(sec,a.out.with_name(a.out.stem+"_secondary_robustness.csv"))
    elif not sec.empty: print("Secondary rows excluded from primary comparison")

if __name__=="__main__": main()
