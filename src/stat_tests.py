#!/usr/bin/env python3
"""Origin-aware statistical comparisons for Ghana and the two Italy arms."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# Seeds are repeated fits nested within the same forecast origin, not
# independent forecast cases. Ten origins is the reporting threshold used here
# to distinguish origin-count-adequate results from explicitly exploratory
# results; it is a screening rule, not a universal statistical guarantee.
# Expanding windows may overlap, so this threshold does not prove origin
# independence.
MIN_INDEPENDENT_ORIGINS = 10
REQUIRED_COLUMNS = {"country", "method", "regime", "origin", "seed", "mase"}
PAIR_COLUMNS = ["country", "method", "regime", "origin", "seed"]
PINN_METHODS = {"pinn_split", "pinn_joint"}


def validate_results(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Reject malformed or duplicate fit records before computing summaries."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")

    clean = df.copy()
    for column in ("country", "method", "regime", "origin", "seed"):
        if clean[column].isna().any():
            raise ValueError(f"{label} contains missing values in '{column}'")
    for column in ("country", "method", "regime"):
        if not clean[column].map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"{label} contains non-text values in '{column}'")

    clean["mase"] = pd.to_numeric(clean["mase"], errors="raise")
    if not np.isfinite(clean["mase"].to_numpy(dtype=float)).all():
        raise ValueError(f"{label} contains non-finite MASE values")

    duplicates = clean.duplicated(PAIR_COLUMNS, keep=False)
    if duplicates.any():
        examples = clean.loc[duplicates, PAIR_COLUMNS].head(5).to_dict("records")
        raise ValueError(f"{label} contains duplicate fit keys: {examples}")
    return clean


def inference_label(*origin_counts: int) -> str:
    """Describe whether origin counts meet this script's reporting threshold."""
    if min(origin_counts) >= MIN_INDEPENDENT_ORIGINS:
        return "origin_count_adequate"
    return "exploratory_insufficient_independent_origins"


def wilcoxon_split_vs_joint(df: pd.DataFrame) -> pd.DataFrame:
    """Compare split and joint fits after collapsing repeated seeds by origin."""
    rows = []
    pinn = df[df["method"].isin(PINN_METHODS)]
    for (country, regime), group in pinn.groupby(["country", "regime"], sort=True):
        # First match split/joint runs within each seed, then average those
        # paired fits at each origin so repeated seeds do not inflate n.
        paired_seeds = group[group["method"].isin(PINN_METHODS)].pivot(
            index=["origin", "seed"], columns="method", values="mase"
        )
        if not {"pinn_split", "pinn_joint"}.issubset(paired_seeds.columns):
            continue
        paired_seeds = paired_seeds.dropna(subset=["pinn_split", "pinn_joint"])
        origin_scores = paired_seeds.groupby(level="origin")[
            ["pinn_split", "pinn_joint"]
        ].mean()
        differences = origin_scores["pinn_split"] - origin_scores["pinn_joint"]
        n_origins = len(origin_scores)

        # With fewer than two origins there is no meaningful signed-rank test;
        # retain a result row with descriptive summaries and an explicit status.
        if n_origins >= 2:
            if np.allclose(differences.to_numpy(), 0.0):
                statistic, p_value = 0.0, 1.0
            else:
                statistic, p_value = stats.wilcoxon(
                    differences, alternative="two-sided"
                )
            test_status = "computed"
        else:
            statistic, p_value = np.nan, np.nan
            test_status = "not_tested_fewer_than_two_origins"

        rows.append(
            {
                "test": "wilcoxon_split_vs_joint",
                "country": country,
                "regime": regime,
                "n_origins": n_origins,
                "n_paired_seed_fits": len(paired_seeds),
                "statistic": statistic,
                "p_value": p_value,
                "median_split": origin_scores["pinn_split"].median(),
                "median_joint": origin_scores["pinn_joint"].median(),
                "inference_label": inference_label(n_origins),
                "test_status": test_status,
            }
        )
    return pd.DataFrame(rows)


def friedman_across_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """Compare regimes using one seed-averaged score per origin and regime."""
    rows = []
    pinn = df[df["method"].isin(PINN_METHODS)]
    for (country, method), group in pinn.groupby(["country", "method"], sort=True):
        # The origin is the repeated-measures unit. Averaging seeds before
        # pivoting ensures the Friedman sample count is independent origins.
        origin_scores = group.groupby(["origin", "regime"])["mase"].mean().unstack()
        complete = origin_scores.dropna()
        regimes = list(complete.columns)
        n_origins = len(complete)

        if len(regimes) >= 3 and n_origins >= 2:
            statistic, p_value = stats.friedmanchisquare(
                *[complete[regime] for regime in regimes]
            )
            test_status = "computed"
        else:
            statistic, p_value = np.nan, np.nan
            test_status = "not_tested_requires_two_origins_and_three_regimes"

        rows.append(
            {
                "test": "friedman_across_regimes",
                "country": country,
                "method": method,
                "n_origins": n_origins,
                "regimes": regimes,
                "statistic": statistic,
                "p_value": p_value,
                "inference_label": inference_label(n_origins),
                "test_status": test_status,
            }
        )
    return pd.DataFrame(rows)


def mannwhitney_clustered_vs_random(
    ghana: pd.DataFrame,
    italy_arm: pd.DataFrame,
    italy_arm_label: str,
    italy_regime: str = "sparse30",
) -> pd.DataFrame:
    """Compare Ghana and Italy regime scores at the forecast-origin level."""
    ghana_fits = ghana[
        ghana["method"].str.startswith("pinn") & (ghana["regime"] == "mask")
    ]
    italy_fits = italy_arm[
        italy_arm["method"].str.startswith("pinn")
        & (italy_arm["regime"] == italy_regime)
    ]

    # Seeds measure fit variability within an origin; each origin contributes
    # one mean score to the between-country comparison.
    ghana_by_origin = ghana_fits.groupby("origin")["mase"].mean()
    italy_by_origin = italy_fits.groupby("origin")["mase"].mean()
    n_ghana, n_italy = len(ghana_by_origin), len(italy_by_origin)
    n_ghana_fits, n_italy_fits = len(ghana_fits), len(italy_fits)

    if min(n_ghana, n_italy) >= 2:
        statistic, p_value = stats.mannwhitneyu(
            ghana_by_origin, italy_by_origin, alternative="two-sided"
        )
        test_status = "computed"
    else:
        statistic, p_value = np.nan, np.nan
        test_status = "not_tested_fewer_than_two_origins_in_a_group"

    caveat = (
        "Exploratory robustness comparison: italy_secondary spans multiple "
        "COVID-19 waves/variants, unlike Ghana's single-wave window; regime "
        "shifts may confound the comparison."
        if italy_arm_label == "italy_secondary"
        else "Between-country comparison of Ghana clustered missingness and "
        "Italy-primary controlled sparsity; country and data-source differences "
        "limit causal interpretation."
    )
    return pd.DataFrame(
        [
            {
                "test": "mannwhitney_clustered_vs_random",
                "ghana_regime": "mask",
                "italy_arm": italy_arm_label,
                "italy_regime": italy_regime,
                "n_ghana_origins": n_ghana,
                "n_italy_origins": n_italy,
                "n_ghana_seed_fits": n_ghana_fits,
                "n_italy_seed_fits": n_italy_fits,
                "statistic": statistic,
                "p_value": p_value,
                "median_ghana": ghana_by_origin.median(),
                "median_italy": italy_by_origin.median(),
                "inference_label": inference_label(n_ghana, n_italy),
                "test_status": test_status,
                "caveat": caveat,
            }
        ]
    )


def main() -> None:
    # Parse file inputs and arm selection. The `both` option creates two
    # explicitly labelled Italy comparisons, not a pooled Italy sample.
    parser = argparse.ArgumentParser(
        description="Run origin-aware exploratory statistical comparisons."
    )
    parser.add_argument("--grid", required=True, type=Path)
    parser.add_argument(
        "--italy-grid",
        required=True,
        type=Path,
        help="'country' must identify each Italy row as italy_primary or italy_secondary",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--italy-arm",
        choices=["primary", "secondary", "both"],
        default="primary",
        help="Italy arm(s) for separate comparisons; Ghana-only tests run once.",
    )
    args = parser.parse_args()

    # Validate identity and fit keys before testing. A missing Ghana country
    # label is unambiguous for this input; Italy arm labels are never guessed.
    ghana = pd.read_csv(args.grid)
    if "country" not in ghana.columns:
        ghana["country"] = "ghana"
    ghana = validate_results(ghana, "--grid")
    if set(ghana["country"].astype(str)) != {"ghana"}:
        raise ValueError("--grid must contain Ghana rows only; cross-country pooling is refused")

    italy = validate_results(pd.read_csv(args.italy_grid), "--italy-grid")
    italy_countries = set(italy["country"].astype(str))
    allowed_italy = {"italy_primary", "italy_secondary"}
    if not italy_countries or not italy_countries.issubset(allowed_italy):
        raise ValueError(
            "--italy-grid country values must be italy_primary and/or italy_secondary"
        )

    arms = (
        ["italy_primary", "italy_secondary"]
        if args.italy_arm == "both"
        else [f"italy_{args.italy_arm}"]
    )
    selected_italy = italy[italy["country"].isin(arms)]
    if selected_italy.empty:
        raise ValueError(f"--italy-grid has no rows for selected arm(s): {arms}")

    # Run within-country tests on a single combined table; their explicit
    # country grouping prevents cross-country pooling and avoids rerunning
    # Ghana's Wilcoxon/Friedman tests once for each selected Italy arm.
    within_country = pd.concat([ghana, selected_italy], ignore_index=True)
    results = [
        wilcoxon_split_vs_joint(within_country),
        friedman_across_regimes(within_country),
    ]

    # The between-country test is intentionally repeated only for each
    # requested Italy comparator, with Ghana fits summarized by origin.
    for arm in arms:
        italy_arm = selected_italy[selected_italy["country"] == arm]
        if not italy_arm.empty:
            results.append(
                mannwhitney_clustered_vs_random(ghana, italy_arm, arm)
            )

    nonempty_results = [result for result in results if not result.empty]
    output = (
        pd.concat(nonempty_results, ignore_index=True, sort=False)
        if nonempty_results
        else pd.DataFrame()
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    if output.empty:
        print("No eligible statistical comparisons were produced.")
    else:
        print(output.to_string(index=False))


if __name__ == "__main__":
    main()
