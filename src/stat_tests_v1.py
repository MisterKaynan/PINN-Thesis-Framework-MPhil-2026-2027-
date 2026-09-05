#!/usr/bin/env python3
"""
Formal statistical tests supporting H1/H2/H3, filling the "2.7.5 Statistical
Tests" section the course project did not need (it reported medians/IQR
only, per its own Table 2).

TESTS
    1. Wilcoxon signed-rank: split vs joint MASE, paired by (origin, seed),
       within each (country, regime). Supports H1's accuracy claim with a
       p-value rather than a point percentage difference alone.
    2. Friedman test: MASE across regimes (daily/weekly/mask-or-sparse),
       within a fixed (country, method). Tests whether degradation across
       regimes is statistically distinguishable, supporting H1/H2.
    3. Mann-Whitney U: MASE-gap distributions, Italy-sparse-at-X% vs
       Ghana-mask (nominal missingness matched as closely as available).
       Tests whether CLUSTERED authentic missingness (Ghana) degrades
       performance differently from UNIFORM-RANDOM synthetic missingness
       (Italy) at a similar overall percentage -- a novel empirical question
       this pipeline is positioned to answer.

Usage:
    python3 stat_tests.py --grid grid.csv --italy-grid italy_grid.csv \
        --out significance.csv
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


def mannwhitney_clustered_vs_random(ghana: pd.DataFrame, italy: pd.DataFrame,
                                     italy_regime: str = "sparse30") -> pd.DataFrame:
    rows = []
    gm = ghana[(ghana.method.str.startswith("pinn")) & (ghana.regime == "mask")]["mase"]
    im = italy[(italy.method.str.startswith("pinn")) & (italy.regime == italy_regime)]["mase"]
    if len(gm) >= 5 and len(im) >= 5:
        stat, p = stats.mannwhitneyu(gm, im, alternative="two-sided")
        rows.append(dict(test="mannwhitney_clustered_vs_random",
                         ghana_regime="mask", italy_regime=italy_regime,
                         n_ghana=len(gm), n_italy=len(im), statistic=stat, p_value=p,
                         median_ghana=gm.median(), median_italy=im.median()))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, type=Path)
    ap.add_argument("--italy-grid", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    ghana = pd.read_csv(a.grid)
    ghana["country"] = "ghana"
    italy = pd.read_csv(a.italy_grid)
    if "country" not in italy.columns:
        italy["country"] = "italy"
    combined = pd.concat([ghana, italy], ignore_index=True)

    r1 = wilcoxon_split_vs_joint(combined)
    r2 = friedman_across_regimes(combined)
    r3 = mannwhitney_clustered_vs_random(ghana, italy)

    out = pd.concat([r1, r2, r3], ignore_index=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
