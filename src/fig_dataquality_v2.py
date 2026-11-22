#!/usr/bin/env python3
"""Plot country-level data quality and Italy-arm incidence proxies.

The Italy-secondary panel is only produced from a master regenerated with
``--ascertainment none``. Its audit metadata and post-cutoff proxy coverage are
validated before plotting, preventing stale or zero-corrupted outputs from
being presented as a multi-wave robustness series.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Keep input and output locations centralized so the checks and plotted series
# always refer to the same processed products.
OUT = Path("results/figures/fig_data_quality.png")
GHANA_PATH = Path("data/processed/ghana_national_master.csv")
ITALY_PRIMARY_PATH = Path("data/processed/italy_primary_national_master.csv")
ITALY_SECONDARY_PATH = Path("data/processed/italy_secondary_national_master.csv")
ITALY_SECONDARY_AUDIT_PATH = Path("data/processed/italy_secondary_audit_summary.json")
R6_VALID_END = pd.Timestamp("2020-08-31")


def load_daily_master(path: Path, label: str) -> pd.DataFrame:
    """Load a dated master and reject malformed or incomplete daily axes."""
    master = pd.read_csv(path, index_col=0, parse_dates=True)
    if master.empty:
        raise ValueError(f"{label} master is empty: {path}")
    if master.index.isna().any():
        raise ValueError(f"{label} master has invalid dates: {path}")
    if master.index.has_duplicates:
        raise ValueError(f"{label} master has duplicate dates: {path}")
    if not master.index.is_monotonic_increasing:
        raise ValueError(f"{label} master dates are not sorted: {path}")
    expected_dates = pd.date_range(master.index.min(), master.index.max(), freq="D")
    if not master.index.equals(expected_dates):
        raise ValueError(f"{label} master does not contain a complete daily calendar: {path}")
    return master


def validate_italy_master(master: pd.DataFrame, label: str) -> None:
    """Ensure the plotted Italian proxy is present, finite, and non-negative."""
    required = {"I_proxy"}
    missing = required - set(master.columns)
    if missing:
        raise ValueError(f"{label} master is missing required columns: {sorted(missing)}")

    proxy = pd.to_numeric(master["I_proxy"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(proxy).all():
        raise ValueError(f"{label} I_proxy contains missing or non-finite values")
    if (proxy < 0).any():
        raise ValueError(f"{label} I_proxy contains negative values")
    if not np.any(proxy > 0):
        raise ValueError(f"{label} I_proxy is entirely zero; refusing to plot it")


def validate_secondary_master(master: pd.DataFrame, audit_path: Path) -> dict[str, int]:
    """Require raw secondary ascertainment and proxy coverage matching its source."""
    # The sidecar binds the plotted proxy to the intended secondary
    # preprocessing mode; a finite-looking proxy from a stale r=6 run is not
    # accepted as a substitute for the raw reported-incidence robustness arm.
    if not audit_path.exists():
        raise ValueError(
            f"Missing secondary preprocessing audit: {audit_path}. "
            "Regenerate the Italy-secondary products with --ascertainment none."
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("arm") != "secondary" or audit.get("ascertainment_mode") != "none":
        raise ValueError(
            "Italy-secondary audit must identify arm='secondary' and "
            "ascertainment_mode='none'. Regenerate it with "
            "--arm secondary --ascertainment none before plotting."
        )

    # After the r=6 validity boundary, the secondary target should use raw
    # reported incidence. A positive daily source value must therefore also
    # produce a positive trailing proxy on that day. This catches zero-filled
    # or otherwise corrupted proxies even when every stored value is finite.
    required = {"new_infections", "I_proxy"}
    missing = required - set(master.columns)
    if missing:
        raise ValueError(
            f"Italy-secondary master is missing source/proxy columns: {sorted(missing)}"
        )
    after_cutoff = master.index > R6_VALID_END
    if not after_cutoff.any():
        raise ValueError(
            "Italy-secondary master has no dates after the r=6 validity boundary"
        )

    incidence = pd.to_numeric(
        master.loc[after_cutoff, "new_infections"], errors="coerce"
    ).to_numpy(dtype=float)
    proxy = pd.to_numeric(
        master.loc[after_cutoff, "I_proxy"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(incidence).all():
        raise ValueError("Italy-secondary post-cutoff new_infections is not finite")
    if (incidence < 0).any():
        raise ValueError("Italy-secondary post-cutoff new_infections contains negatives")
    if not np.isfinite(proxy).all():
        raise ValueError("Italy-secondary post-cutoff I_proxy is not finite")

    source_positive = int(np.count_nonzero(incidence > 0))
    proxy_positive = int(np.count_nonzero(proxy > 0))
    if source_positive == 0:
        raise ValueError(
            "Italy-secondary source has no positive post-cutoff incidence; "
            "coverage cannot be validated"
        )
    if proxy_positive < source_positive:
        raise ValueError(
            "Italy-secondary post-cutoff I_proxy has "
            f"{proxy_positive}/{len(proxy)} positive days but raw "
            f"new_infections has {source_positive}/{len(incidence)}. "
            "The proxy is stale or corrupted; regenerate the secondary "
            "master with --ascertainment none before plotting."
        )
    return {
        "post_cutoff_days": len(proxy),
        "post_cutoff_source_positive_days": source_positive,
        "post_cutoff_proxy_positive_days": proxy_positive,
    }


def panel_ghana(fig: plt.Figure, outer_gs, master: pd.DataFrame) -> None:
    """Render Ghana incidence with authentic reporting gaps and gap lengths."""
    inner = outer_gs.subgridspec(2, 1, height_ratios=[2.4, 1.6], hspace=0.32)
    ax0 = fig.add_subplot(inner[0])
    ax1 = fig.add_subplot(inner[1], sharex=ax0)

    ax0.fill_between(master.index, master.daily_incidence, color="#c8d8e8", lw=0)
    ax0.plot(master.index, master.daily_incidence, color="#2c5f8a", lw=0.8)
    for date in master.index[master.observed_simple == 0]:
        ax0.axvline(date, color="#d94f4f", alpha=0.16, lw=1.4)
    ax0.set_ylabel("Daily reported cases")
    ax0.set_title(
        "Ghana COVID-19 national incidence with unreported days shaded "
        "(HERA, 12 Mar 2020 - 19 May 2021)",
        fontsize=10,
        loc="left",
    )
    unreported_pct = 100 * (master.observed_simple == 0).mean()
    handles = [
        plt.Line2D([], [], color="#2c5f8a", lw=1.2, label="Reported incidence"),
        plt.Line2D(
            [],
            [],
            color="#d94f4f",
            lw=4,
            alpha=0.3,
            label=f"No report ({(master.observed_simple == 0).sum()} d, {unreported_pct:.1f}%)",
        ),
    ]
    ax0.legend(handles=handles, fontsize=8, frameon=False, loc="upper left")

    # The lower panel retains Ghana's original gap-length audit rather than
    # applying that country-specific measure to Italy's complete feed.
    gap_lengths = (
        master.gap_length.where(master.gap_length > 0, 0)
        if "gap_length" in master.columns
        else pd.Series(0, index=master.index)
    )
    ax1.bar(master.index, gap_lengths, color="#d94f4f", width=1.0)
    ax1.set_ylabel("Gap length (days)", fontsize=8)
    ax1.set_xlabel("Date")
    if gap_lengths.max() > 0:
        peak_date = gap_lengths.idxmax()
        ax1.set_ylim(0, gap_lengths.max() * 1.25)
        ax1.annotate(
            f"{int(gap_lengths.max())}-day gap",
            xy=(peak_date, gap_lengths.max()),
            xytext=(peak_date, gap_lengths.max() * 1.08),
            va="bottom",
            fontsize=7.5,
            arrowprops=dict(arrowstyle="->", lw=0.7, color="#555"),
        )
    for axis in (ax0, ax1):
        axis.spines[["top", "right"]].set_visible(False)


def panel_italy(
    fig: plt.Figure,
    outer_gs,
    master: pd.DataFrame,
    label: str,
    note: str,
    color: str,
    ylabel: str,
) -> None:
    """Render a validated Italian incidence proxy and its arm-specific note."""
    axis = fig.add_subplot(outer_gs)
    axis.fill_between(master.index, master.I_proxy, color=color, alpha=0.25, lw=0)
    axis.plot(master.index, master.I_proxy, color=color, lw=0.9)
    axis.set_ylabel(ylabel)
    axis.set_title(label, fontsize=10, loc="left")
    axis.annotate(
        note,
        xy=(0.99, 0.92),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=7.5,
        style="italic",
        color="#555",
    )
    axis.spines[["top", "right"]].set_visible(False)


def main() -> None:
    # Load and validate every available Italy input before creating a figure.
    # A stale secondary product therefore fails closed without writing an
    # incomplete or misleading PNG.
    ghana = load_daily_master(GHANA_PATH, "Ghana")
    italy_primary = None
    if ITALY_PRIMARY_PATH.exists():
        italy_primary = load_daily_master(ITALY_PRIMARY_PATH, "Italy-primary")
        validate_italy_master(italy_primary, "Italy-primary")

    italy_secondary = None
    secondary_coverage = None
    if ITALY_SECONDARY_PATH.exists():
        italy_secondary = load_daily_master(ITALY_SECONDARY_PATH, "Italy-secondary")
        validate_italy_master(italy_secondary, "Italy-secondary")
        secondary_coverage = validate_secondary_master(
            italy_secondary, ITALY_SECONDARY_AUDIT_PATH
        )

    # The Ghana panel is always present; Italy panels are added only when their
    # validated master files exist, preserving the original optional-arm layout.
    panel_count = 1 + int(italy_primary is not None) + int(italy_secondary is not None)
    fig = plt.figure(figsize=(10, 4.8 * panel_count))
    grid = fig.add_gridspec(panel_count, 1, hspace=0.5)
    row = 0
    panel_ghana(fig, grid[row], ghana)
    row += 1

    # Italy-primary remains the fixed-correction, single-wave benchmark.
    if italy_primary is not None:
        date_range = (
            f"{italy_primary.index.min().date()} to "
            f"{italy_primary.index.max().date()}"
        )
        panel_italy(
            fig,
            grid[row],
            italy_primary,
            f"Italy PRIMARY arm -- single-wave, r=6-valid window ({date_range})",
            "Clean benchmark: Objectives 1 & 2",
            "#2c8a5f",
            "I_proxy (r=6 adjusted)",
        )
        row += 1

    # Secondary is shown only as raw reported incidence after its preprocessing
    # audit and source-versus-proxy positive-day coverage checks both pass.
    if italy_secondary is not None:
        date_range = (
            f"{italy_secondary.index.min().date()} to "
            f"{italy_secondary.index.max().date()}"
        )
        panel_italy(
            fig,
            grid[row],
            italy_secondary,
            f"Italy SECONDARY arm -- full multi-wave range ({date_range})",
            "Raw reported-incidence stress test ONLY -- Objective 3, "
            "NOT a clean data-richness comparison",
            "#8a2c5f",
            "I_proxy (reported, D=5)",
        )
        row += 1

    # Save only after successful validation and plot construction, then report
    # the secondary coverage evidence alongside the generated figure path.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"ok: wrote {OUT}")
    if secondary_coverage is not None:
        print(f"Italy-secondary coverage audit: {secondary_coverage}")


if __name__ == "__main__":
    main()
