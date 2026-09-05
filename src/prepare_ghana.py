#!/usr/bin/env python3
"""
HERA Ghana COVID-19 preprocessing for the CSCD 618 / DSCD 604 final project.

Produces the analysis dataset, the authentic observation mask, and the
Millevoi-compatible export used to train the reduced SIR PINN.

DESIGN DECISIONS (each is defensible in the report; none is silent)

1. The source series is INCIDENT (daily new), not cumulative. Verified by audit:
   values are small and non-monotonic. We therefore do NOT forward-fill and do
   NOT apply a cumulative-maximum monotonicity adjustment. Doing either would
   corrupt an incident series.

2. Negative values are data revisions. We clip them to zero at REGION level and
   log every instance rather than overwriting silently.

3. The date grid is already complete (434 consecutive days, 17 regions, no
   missing rows). Missingness is expressed as ZEROS, not absent records.

4. Observation mask. A national zero during an active epidemic is a reporting
   gap, not an absence of transmission. Audit shows every zero-run of 3+ days
   falls inside a period of substantial surrounding activity. Two masks:
     - mask_simple  : observed iff national incidence > 0        [PRIMARY]
     - mask_context : observed iff incidence > 0, OR the surrounding window is
                      quiet enough that a true zero is plausible [SENSITIVITY]

5. Infection proxy. Reported daily incidence is not the SIR state I(t).
   We use a trailing rolling sum over D days, D = 5 to match Millevoi et al.'s
   COVID infectious period. D = 7, 10, 14 exported for sensitivity.

6. `Non spécifié` is a real reporting category (first appears 2020-10-08).
   Included in the primary national series; an excluding variant is exported.

Usage:  python3 prepare_ghana.py --input <csv> --outdir <dir>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLS = ["CONTAMINES", "DECES", "GUERIS"]
PROXY_WINDOWS = [5, 7, 10, 14]
PRIMARY_D = 5          # matches Millevoi et al. (2024), infectious period D = 5 d
QUIET_THRESHOLD = 10   # cases in +/- 3 day window below which a zero is plausible


def load_raw(path: Path) -> pd.DataFrame:
    """Read the HERA export. Semicolon-delimited, BOM'd, day-first dates."""
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df["DATE"] = pd.to_datetime(df["DATE"], format="%d/%m/%Y")
    # DECES arrives as a zero-padded string and must be coerced, not cast.
    for c in RAW_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df.sort_values(["DATE", "REGION"]).reset_index(drop=True)


def log_revisions(df: pd.DataFrame) -> pd.DataFrame:
    """Capture every negative value before clipping, for the audit trail."""
    neg = df[(df[RAW_COLS] < 0).any(axis=1)]
    return neg[["DATE", "REGION"] + RAW_COLS].copy()


def build_national(df: pd.DataFrame, include_unspecified: bool = True) -> pd.DataFrame:
    """Clip negatives at region level, then aggregate to a national daily series."""
    d = df.copy()
    if not include_unspecified:
        d = d[d.REGION != "Non spécifié"]
    d[RAW_COLS] = d[RAW_COLS].clip(lower=0)
    nat = d.groupby("DATE")[RAW_COLS].sum()
    nat = nat.rename(columns={
        "CONTAMINES": "daily_incidence",
        "DECES": "daily_deaths",
        "GUERIS": "daily_recoveries",
    })
    full = pd.date_range(nat.index.min(), nat.index.max(), freq="D")
    return nat.reindex(full).fillna(0).astype(int).rename_axis("date")


def build_masks(nat: pd.DataFrame) -> pd.DataFrame:
    """
    Authentic reporting mask.

    mask_simple: a day counts as observed iff incidence > 0. This is the
    primary mask and the one used for the R7-style experiment.

    mask_context: a zero is treated as GENUINELY zero (and so still observed)
    when the surrounding +/-3 days are quiet, since early in an outbreak a true
    zero day is plausible. Otherwise a zero is treated as unreported.
    """
    inc = nat["daily_incidence"]
    mask_simple = inc > 0

    ctx = inc.rolling(7, center=True, min_periods=1).sum() - inc
    plausible_zero = (inc == 0) & (ctx <= QUIET_THRESHOLD)
    mask_context = mask_simple | plausible_zero

    out = pd.DataFrame({
        "observed_simple": mask_simple.astype(int),
        "observed_context": mask_context.astype(int),
        "context_activity": ctx.astype(int),
    }, index=nat.index)

    # Label consecutive unreported stretches so gap length can be analysed.
    unrep = ~mask_simple
    grp = (unrep != unrep.shift()).cumsum()
    out["gap_length"] = np.where(
        unrep, unrep.groupby(grp).transform("size"), 0
    )
    return out


def add_proxies(nat: pd.DataFrame) -> pd.DataFrame:
    """Trailing rolling-sum infection proxies for the SIR state I(t)."""
    out = nat.copy()
    for d in PROXY_WINDOWS:
        out[f"I_proxy_D{d}"] = (
            nat["daily_incidence"].rolling(d, min_periods=1).sum().astype(int)
        )
    out["I_proxy"] = out[f"I_proxy_D{PRIMARY_D}"]
    return out


def weekly_aggregate(nat: pd.DataFrame) -> pd.DataFrame:
    """
    Weekly regime. Incidence is a FLOW, so weeks are summed, never subsampled.
    Subsampling every 7th day would be valid only for a prevalence measure.
    """
    wk = nat[["daily_incidence", "daily_deaths", "daily_recoveries"]].resample("7D").sum()
    wk.columns = ["weekly_incidence", "weekly_deaths", "weekly_recoveries"]
    wk["days_reported"] = (
        (nat["daily_incidence"] > 0).astype(int).resample("7D").sum()
    )
    return wk


def millevoi_export(master: pd.DataFrame, masks: pd.DataFrame) -> pd.DataFrame:
    """
    Column layout mirroring the repo's Case4.txt / RealData.txt so the PINN
    training code can consume Ghana and Italy through the same reader.

    NOTE: there is no hospitalisation field in HERA. Millevoi's Cases 5-7 use
    H data and are NOT reproducible here. Ghana sits in Case 4 territory:
    reduced model, infection data only.
    """
    return pd.DataFrame({
        "Date": master.index.strftime("%Y-%m-%d"),
        "I_data": master["I_proxy"].values,
        "New_cases": master["daily_incidence"].values,
        "Observed": masks["observed_simple"].values,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(args.input)
    revisions = log_revisions(raw)

    nat = build_national(raw, include_unspecified=True)
    nat_excl = build_national(raw, include_unspecified=False)
    masks = build_masks(nat)
    master = add_proxies(nat)
    master = master.join(masks)
    weekly = weekly_aggregate(nat)

    master.to_csv(args.outdir / "ghana_national_master.csv")
    nat_excl.to_csv(args.outdir / "ghana_national_excl_unspecified.csv")
    weekly.to_csv(args.outdir / "ghana_weekly.csv")
    revisions.to_csv(args.outdir / "revision_log.csv", index=False)
    millevoi_export(master, masks).to_csv(
        args.outdir / "ghana_millevoi_format.csv", index=False
    )

    n = len(master)
    obs = int(masks["observed_simple"].sum())
    gaps = masks.loc[masks["gap_length"] > 0, "gap_length"]
    summary = {
        "date_start": str(master.index.min().date()),
        "date_end": str(master.index.max().date()),
        "n_days": n,
        "n_regions": int(raw.REGION.nunique()),
        "total_cases": int(master["daily_incidence"].sum()),
        "total_deaths": int(master["daily_deaths"].sum()),
        "total_recoveries": int(master["daily_recoveries"].sum()),
        "observed_days_simple": obs,
        "observed_pct_simple": round(100 * obs / n, 1),
        "observed_days_context": int(masks["observed_context"].sum()),
        "unreported_days": n - obs,
        "longest_gap": int(gaps.max()) if len(gaps) else 0,
        "n_gaps": int((masks["gap_length"].diff() > 0).sum()),
        "negative_records_clipped": int(len(revisions)),
        "cases_excl_unspecified": int(nat_excl["daily_incidence"].sum()),
        "primary_infectious_period_D": PRIMARY_D,
        "n_weeks": int(len(weekly)),
    }
    (args.outdir / "audit_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print("\nWrote:", *[p.name for p in sorted(args.outdir.glob('*'))], sep="\n  ")


if __name__ == "__main__":
    main()
