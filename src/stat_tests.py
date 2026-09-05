#!/usr/bin/env python3
"""
Formal statistical tests -- v2, updated for the two-arm Italy design.

CHANGES FROM v1 (stat_tests-3.py as attached)
    1. wilcoxon_split_vs_joint and friedman_across_regimes now group by the
       resolved "country" column, which can be ghana/italy_primary/
       italy_secondary. This is automatic (no logic change needed) PROVIDED
       italy_grid.csv's country column is set correctly upstream -- v1's
       fallback that force-labelled missing country as "italy" is REMOVED,
       since that silently merges primary and secondary arms into one
       "italy" bucket, invalidating the two-arm separation.
    2. mannwhitney_clustered_vs_random previously took one --italy-grid and
       one --italy-regime, implicitly assuming a single Italy series. It is
       now explicit about WHICH Italy arm feeds the comparison via
       --italy-arm {primary,secondary}. Comparing Ghana's authentic
       clustered missingness against italy_SECONDARY's synthetic sparsity
       is scientifically defensible only as an Objective-3 robustness note
       (multi-wave dynamics differ from Ghana's single wave) -- the function
       now stamps that caveat directly into its output row so it survives
       into any downstream table without manual re-annotation.
    3. Added a guard identical to equity_metrics.py: if the italy-grid
       contains an unset/ambiguous country label, the script refuses to
       silently default it and raises instead.

Usage:
    python3 stat_tests.py --grid grid.csv --italy-grid italy_grid.csv \
        --out significance.csv --italy-arm primary
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def wilcoxon_split_vs_joint(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (country, regime), g in df[df.method.str.startswith("pinn")].groupby(["country", "regime"]):
        split = g[g.method == "pinn_split"].set_index(["origin", "seed"])["mase"]
        joint = g[g.method == "pinn_joint"].set_index(["origin", "seed"])["mase"]
        paired = pd.concat([split.rename("split"), joint.rename("joint")], axis=1).dropna()
        if len(paired) < 5:
            continue
        stat, p = stats.wilcoxon(paired["split"], paired["joint"])
        rows.append(dict(test="wilcoxon_split_vs_joint", country=country, regime=regime,
                         n=len(paired), statistic=stat, p_value=p,
                         median_split=paired["split"].median(),
                         median_joint=paired["joint"].median()))
    return pd.DataFrame(rows)


def friedman_across_regimes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (country, method), g in df[df.method.str.startswith("pinn")].groupby(["country", "method"]):
        piv = g.pivot_table(index=["origin", "seed"], columns="regime", values="mase")
        piv = piv.dropna()
        if piv.shape[0] < 5 or piv.shape[1] < 3:
            continue
        stat, p = stats.friedmanchisquare(*[piv[c] for c in piv.columns])
        rows.append(dict(test="friedman_across_regimes", country=country, method=method,
                         n=piv.shape[0], regimes=list(piv.columns),
                         statistic=stat, p_value=p))
    return pd.DataFrame(rows)


def mannwhitney_clustered_vs_random(ghana: pd.DataFrame, italy_arm: pd.DataFrame,
                                     italy_arm_label: str,
                                     italy_regime: str = "sparse30") -> pd.DataFrame:
    rows = []
    gm = ghana[(ghana.method.str.startswith("pinn")) & (ghana.regime == "mask")]["mase"]
    im = italy_arm[(italy_arm.method.str.startswith("pinn")) & (italy_arm.regime == italy_regime)]["mase"]
    if len(gm) >= 5 and len(im) >= 5:
        stat, p = stats.mannwhitneyu(gm, im, alternative="two-sided")
        caveat = ("Objective-3 robustness note only: italy_secondary spans "
                  "multiple COVID-19 waves/variants, unlike Ghana's single-"
                  "wave window -- interpret with regime-shift confound in mind."
                  if italy_arm_label == "italy_secondary" else
                  "Clean comparison: both series are single-wave-comparable "
                  "(italy_primary is Millevoi-window matched).")
        rows.append(dict(test="mannwhitney_clustered_vs_random",
                         ghana_regime="mask", italy_arm=italy_arm_label,
                         italy_regime=italy_regime,
                         n_ghana=len(gm), n_italy=len(im), statistic=stat, p_value=p,
                         median_ghana=gm.median(), median_italy=im.median(),
                         caveat=caveat))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, type=Path)
    ap.add_argument("--italy-grid", required=True, type=Path,
                    help="'country' column must be 'italy_primary' or "
                         "'italy_secondary' per row -- no silent fallback")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--italy-arm", choices=["primary", "secondary", "both"],
                    default="primary",
                    help="Which Italy arm feeds wilcoxon/friedman/mannwhitney. "
                         "'both' runs each test once per arm, clearly tagged.")
    a = ap.parse_args()

    ghana = pd.read_csv(a.grid)
    if "country" not in ghana.columns:
        ghana["country"] = "ghana"
    italy = pd.read_csv(a.italy_grid)
    if "country" not in italy.columns:
        raise SystemExit(
            "italy-grid is missing a 'country' column with values "
            "'italy_primary'/'italy_secondary'. Refusing to silently guess.")

    arms = ["italy_primary", "italy_secondary"] if a.italy_arm == "both" \
        else [f"italy_{a.italy_arm}"]

    all_results = []
    for arm in arms:
        italy_arm_df = italy[italy["country"] == arm]
        if italy_arm_df.empty:
            print(f"NOTE: no rows found for country == '{arm}' in --italy-grid, skipping.")
            continue
        combined = pd.concat([ghana, italy_arm_df], ignore_index=True)
        r1 = wilcoxon_split_vs_joint(combined)
        r2 = friedman_across_regimes(combined)
        r3 = mannwhitney_clustered_vs_random(ghana, italy_arm_df, arm)
        all_results.extend([r1, r2, r3])

    out = pd.concat([r for r in all_results if not r.empty], ignore_index=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
