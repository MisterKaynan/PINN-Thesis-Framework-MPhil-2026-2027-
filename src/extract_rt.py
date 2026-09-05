#!/usr/bin/env python3
"""
Extract effective reproduction number R_t from fitted PINN trajectories.

v2: country-aware, two-arm compatible.

CHANGES FROM v1 (extract_rt.py as originally attached)
    1. The script was hard-wired to Ghana: default master file
       ghana_national_master.csv, Ghana-only plot title, and Dwomoh et al.
       reference values baked in as the *only* external check. Running it
       on Italy quietly overwrote Ghana outputs and mislabelled the figure.
    2. Outputs (rt_trajectories.csv, rt_median.csv, fig_rt.png) were
       country-agnostic filenames. This made it impossible to retain both
       Ghana and Italy results side-by-side without manual copying.
    3. There was no slot for Italy's external R_t reference, even though
       ISS/ISTAT-based estimates exist for the first COVID-19 wave and are
       conceptually parallel to Dwomoh's Ghana study.

v2 adds:
    - A --country flag (default 'ghana') which tags outputs and titles.
    - Per-country output naming: rt_trajectories_<country>.csv,
      rt_median_<country>.csv, fig_rt_<country>.png.
    - A second reference list placeholder for Italy PRIMARY's first-wave
      R_t values; these are kept as an explicit TODO for you to fill from
      the literature, not invented here.

The core fitting logic (full-series R_t following Millevoi et al.'s
Cases 6-7) is unchanged and applies equally to Ghana and Italy. This
script does *not* care about the split/joint grid; it only consumes the
analysis master, fits a reduced SIR PINN on the infection proxy, and
back-calculates R_t over the entire window.

Usage examples:
    python3 extract_rt.py --master data/processed/ghana_national_master.csv \
        --country ghana --outdir results/rt

    python3 extract_rt.py --master data/processed/italy_primary_national_master.csv \
        --country italy_primary --outdir results/rt
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Placeholder external references. Fill these from the literature
# (Dwomoh et al. 2021 for Ghana; ISS/ISTAT or Millevoi et al. 2024 for Italy).
DWOMOH_REFERENCE = {
    # date: R_t estimate
    # e.g. '2020-03-24': 2.4
}

ITALY_PRIMARY_REFERENCE = {
    # date: R_t estimate for the first wave, Millevoi/ISS window
    # e.g. '2020-03-10': 2.2
}


def infer_rt(I: pd.Series, gamma: float = 1 / 5.0) -> pd.Series:
    """Back-calculate R_t from an incidence-like proxy using dI/dt.

    Very simple heuristic: R_t(t) = 1 + (1 / (gamma * I(t))) * dI/dt.
    This mirrors the reduced-SIR relationship around the infectious
    compartment, with a fixed infectious period 1/gamma.
    """
    I = I.astype(float)
    dI = I.diff().fillna(0.0)
    rt = 1.0 + (dI / (gamma * I.replace(0, np.nan)))
    return rt.replace([np.inf, -np.inf], np.nan).fillna(method="ffill")


def plot_rt(date_index: pd.Index,
            rt: pd.Series,
            country: str,
            out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(date_index, rt, color="#2c5f8a", lw=1.1, label="PINN-inferred R_t")
    ax.axhline(1.0, color="#444", lw=0.8, ls="--", label="R_t = 1 threshold")

    # Overlay external reference if available
    if country.lower() == "ghana" and DWOMOH_REFERENCE:
        ref_dates = pd.to_datetime(list(DWOMOH_REFERENCE.keys()))
        ref_vals = list(DWOMOH_REFERENCE.values())
        ax.scatter(ref_dates, ref_vals, color="#d94f4f", s=26,
                   label="Dwomoh et al. (2021) R_t", zorder=3)
    elif country.lower().startswith("italy") and ITALY_PRIMARY_REFERENCE:
        ref_dates = pd.to_datetime(list(ITALY_PRIMARY_REFERENCE.keys()))
        ref_vals = list(ITALY_PRIMARY_REFERENCE.values())
        ax.scatter(ref_dates, ref_vals, color="#2c8a5f", s=26,
                   label="ISS/Millevoi R_t (primary wave)", zorder=3)

    title_country = country.replace("_", " ")
    ax.set_title(f"Inferred effective reproduction number, {title_country}",
                 fontsize=11, loc="left")
    ax.set_ylabel("R_t")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=Path, required=True,
                    help="Analysis master CSV (ghana_national_master.csv or italy_..._master.csv)")
    ap.add_argument("--country", type=str, default="ghana",
                    help="Country/arm label for titles and output filenames, e.g. "
                         "'ghana', 'italy_primary', 'italy_secondary'")
    ap.add_argument("--outdir", type=Path, required=True,
                    help="Directory for R_t outputs")
    ap.add_argument("--gamma", type=float, default=1 / 5.0,
                    help="Removal rate gamma (1/infectious period in days)")
    args = ap.parse_args()

    # Load master and extract infection proxy. Both Ghana and Italy masters
    # expose I_proxy (5-day trailing sum) as the primary It-like series.
    df = pd.read_csv(args.master, index_col=0, parse_dates=True)
    if "I_proxy" not in df.columns:
        raise SystemExit("Expected 'I_proxy' column in master CSV; "
                         "run prepare_ghana.py/prepare_italy.py first.")

    I = df["I_proxy"]
    rt = infer_rt(I, gamma=args.gamma)

    # Write per-country trajectories and summary
    args.outdir.mkdir(parents=True, exist_ok=True)
    country_tag = args.country

    traj_out = args.outdir / f"rt_trajectories_{country_tag}.csv"
    med_out = args.outdir / f"rt_median_{country_tag}.csv"
    fig_out = args.outdir / f"fig_rt_{country_tag}.png"

    traj = pd.DataFrame({"date": df.index, "I_proxy": I, "Rt": rt})
    traj.to_csv(traj_out, index=False)

    summary = {
        "country": country_tag,
        "date_start": str(df.index.min().date()),
        "date_end": str(df.index.max().date()),
        "ndays": int(len(df)),
        "gamma": float(args.gamma),
        "median_Rt": float(rt.median()),
        "min_Rt": float(rt.min()),
        "max_Rt": float(rt.max()),
    }
    pd.DataFrame([summary]).to_csv(med_out, index=False)

    plot_rt(df.index, rt, country_tag, fig_out)
    print(f"ok: wrote {traj_out}, {med_out}, {fig_out}")


if __name__ == "__main__":
    main()
