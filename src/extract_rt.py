#!/usr/bin/env python3
"""Fit the reduced-SIR PINN and export its learned effective R_t trajectory."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


# External reference overlays stay opt-in and empty until values have been
# verified against their cited source; no comparison values are invented.

# Dwomoh et al. (2021), Section 3.3, journal p. 390.
# Explicitly dated time-dependent estimate; 95% CI: 1.41–2.10.
DWOMOH_REFERENCE: dict[str, float] = {
    "2020-05-13": 1.74,
}

# Millevoi et al. (2024), Figure 11, PDF p. 22.
# APPROXIMATE readings of the BLUE official ISS reference curve,
# NOT the black dashed PINN estimates.
# Dates correspond to labelled ticks; values rounded to one decimal.
ITALY_PRIMARY_REFERENCE: dict[str, float] = {
    "2020-03-04": 2.3,
    "2020-03-29": 0.8,
    "2020-04-23": 0.8,
    "2020-05-18": 0.7,
}


def load_master(path: Path, country: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Validate the daily input series before constructing model tensors."""
    master = pd.read_csv(path, index_col=0, parse_dates=True)
    if master.empty:
        raise ValueError(f"Master file is empty: {path}")
    if master.index.isna().any() or master.index.has_duplicates:
        raise ValueError("Master dates must be valid and unique")
    if not master.index.is_monotonic_increasing:
        raise ValueError("Master dates must be sorted in ascending order")
    expected_dates = pd.date_range(master.index.min(), master.index.max(), freq="D")
    if not master.index.equals(expected_dates):
        raise ValueError("Master must contain one row for each consecutive calendar day")
    if "I_proxy" not in master.columns:
        raise ValueError("Expected 'I_proxy' in the master; prepare the country series first")
    if "country" in master.columns:
        country_values = set(master["country"].dropna().astype(str))
        expected_country = "ghana" if country == "ghana" else country
        if country_values and country_values != {expected_country}:
            raise ValueError(
                f"Master country labels {sorted(country_values)} do not match "
                f"--country {expected_country!r}"
            )

    # The model is fit to a finite, nonnegative observation series. Zero values
    # are valid observations, but an all-zero input cannot identify a useful scale.
    incidence = pd.to_numeric(master["I_proxy"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(incidence).all():
        raise ValueError("I_proxy must contain only finite values")
    if (incidence < 0).any():
        raise ValueError("I_proxy must be non-negative")
    if len(incidence) < 2 or not np.any(incidence > 0):
        raise ValueError("I_proxy needs at least two days and one positive value")
    return master, incidence


def fit_rt(
    incidence: np.ndarray,
    mode: str,
    seed: int,
    epochs_joint: int,
    epochs_split_data: int,
    epochs_split_ode: int,
    n_collocation: int,
    device: str,
    delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Train one full-series PINN and return its fitted I and learned R_net."""
    # Delay the PyTorch import until fitting is requested. Plotting runs in a
    # separate lightweight process so it never initializes Matplotlib after
    # the model's OpenMP runtime in environments with conflicting runtimes.
    import torch

    from pinn import Config, ReducedSIRPINN

    if mode not in {"split", "joint"}:
        raise ValueError(f"Unsupported PINN mode: {mode}")
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be a finite positive removal rate")

    # Scale observations for stable network optimization and use the same
    # normalized time convention as ReducedSIRPINN's training/evaluation code.
    scale = float(incidence.max())
    n_days = len(incidence)
    t_scaled = np.arange(n_days, dtype=np.float32) / n_days
    t_data = torch.tensor(t_scaled, dtype=torch.float32).reshape(-1, 1)
    i_observed = torch.tensor(
        incidence / scale, dtype=torch.float32
    ).reshape(-1, 1)
    config = Config(
        seed=seed,
        device=device,
        delta=delta,
        epochs_joint=epochs_joint,
        epochs_split_data=epochs_split_data,
        epochs_split_ode=epochs_split_ode,
        n_collocation=n_collocation,
    )

    # Both modes train R_net against the reduced-SIR residual. The second
    # prediction returned by predict() is the learned dimensionless R_t curve.
    model = ReducedSIRPINN(config, tf=n_days, scale_I=scale)
    if mode == "split":
        fit_info = model.fit_split(t_data, i_observed)
    else:
        fit_info = model.fit_joint(t_data, i_observed)
    fitted_incidence, rnet = model.predict(t_data)

    # Do not conceal failed or physically inadmissible model output by clipping
    # or forward-filling; fail before any trajectory files are written.
    if not np.isfinite(fitted_incidence).all():
        raise ValueError(f"{mode} seed {seed} produced non-finite fitted incidence")
    if not np.isfinite(rnet).all():
        raise ValueError(f"{mode} seed {seed} produced non-finite R_net values")
    if (rnet < 0).any():
        raise ValueError(f"{mode} seed {seed} produced negative R_net values")
    print(
        f"  {mode} seed={seed}: fit completed in "
        f"{fit_info['seconds']:.1f}s; "
        f"R_net range={float(rnet.min()):.4g}..{float(rnet.max()):.4g}"
    )
    return fitted_incidence, rnet


def summarize_by_date(trajectories: pd.DataFrame) -> pd.DataFrame:
    """Summarize fitted curves across seeds, separately for each training mode."""
    return (
        trajectories.groupby(["country", "mode", "date"], as_index=False)
        .agg(
            median_Rt=("Rt", "median"),
            lower_Rt=("Rt", lambda values: values.quantile(0.1)),
            upper_Rt=("Rt", lambda values: values.quantile(0.9)),
            median_I_pred=("I_pred", "median"),
        )
        .sort_values(["mode", "date"])
    )


def summarize_by_fit(trajectories: pd.DataFrame, delta: float) -> pd.DataFrame:
    """Record full-series fit summaries and provenance for each mode/seed."""
    rows = []
    for (country, mode, seed), group in trajectories.groupby(
        ["country", "mode", "seed"], sort=True
    ):
        rt = group["Rt"].to_numpy(dtype=float)
        rows.append(
            {
                "country": country,
                "mode": mode,
                "seed": seed,
                "date_start": str(pd.Timestamp(group["date"].min()).date()),
                "date_end": str(pd.Timestamp(group["date"].max()).date()),
                "n_days": len(group),
                "delta": delta,
                "median_Rt": float(np.median(rt)),
                "min_Rt": float(np.min(rt)),
                "max_Rt": float(np.max(rt)),
                "estimator": "ReducedSIRPINN.R_net",
            }
        )
    return pd.DataFrame(rows)


def plot_rt(
    daily_summary: pd.DataFrame,
    country: str,
    out_png: Path,
) -> None:
    """Plot median learned R_net by mode with across-seed 10-90% bands."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(9, 4.8))
    colors = {"split": "#2c5f8a", "joint": "#8a5f2c"}
    for mode, group in daily_summary.groupby("mode", sort=True):
        group = group.sort_values("date")
        dates = pd.to_datetime(group["date"])
        color = colors.get(mode, "#555555")
        axis.plot(
            dates,
            group["median_Rt"],
            color=color,
            lw=1.2,
            label=f"{mode} PINN median R_net",
        )
        axis.fill_between(
            dates,
            group["lower_Rt"].to_numpy(dtype=float),
            group["upper_Rt"].to_numpy(dtype=float),
            color=color,
            alpha=0.16,
            linewidth=0,
            label="10th-90th percentile across seeds" if len(daily_summary["mode"].unique()) == 1 else None,
        )

    # Only show literature points for the matching country/arm. Empty reference
    # dictionaries intentionally yield no unsupported external comparison.
    reference = (
        DWOMOH_REFERENCE
        if country == "ghana"
        else ITALY_PRIMARY_REFERENCE
        if country == "italy_primary"
        else {}
    )
    if reference:
        axis.scatter(
            pd.to_datetime(list(reference)),
            list(reference.values()),
            color="#d94f4f",
            s=26,
            label="External R_t reference",
            zorder=3,
        )

    axis.axhline(1.0, color="#444", lw=0.8, ls="--", label="R_t = 1 threshold")
    axis.set_title(
        f"Reduced-SIR PINN-estimated effective reproduction number, "
        f"{country.replace('_', ' ')}",
        fontsize=11,
        loc="left",
    )
    axis.set_ylabel("PINN output R_net(t)")
    axis.set_xlabel("Date")
    axis.legend(fontsize=8, frameon=False, loc="upper right")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    # Collect the country, fit design, and model configuration. Fitting is
    # explicit so the result is not mistaken for a finite-difference proxy.
    parser = argparse.ArgumentParser(
        description="Fit the reduced-SIR PINN and export its learned R_net(t)."
    )
    parser.add_argument("--master", type=Path)
    parser.add_argument(
        "--country",
        choices=["ghana", "italy_primary", "italy_secondary"],
    )
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--modes", nargs="+", choices=["split", "joint"], default=["split", "joint"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--epochs-joint", type=int, default=5000)
    parser.add_argument("--epochs-split-data", type=int, default=3000)
    parser.add_argument("--epochs-split-ode", type=int, default=1000)
    parser.add_argument("--collocation", type=int, default=6000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--delta", type=float, default=0.2,
                        help="Removal rate in days^-1; default corresponds to D=5 days")
    parser.add_argument("--plot-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--summary-csv", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--plot-country", help=argparse.SUPPRESS)
    parser.add_argument("--plot-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # The plotting subprocess reads the already-validated median table and
    # exits without importing or initializing PyTorch.
    if args.plot_only:
        if not args.summary_csv or not args.plot_country or not args.plot_output:
            parser.error("--plot-only requires --summary-csv, --plot-country, and --plot-output")
        plot_rt(pd.read_csv(args.summary_csv), args.plot_country, args.plot_output)
        return

    # The public extraction workflow requires a source master, arm identity,
    # and output directory; these are optional only for the private plot path.
    if not args.master or not args.country or not args.outdir:
        parser.error("--master, --country, and --outdir are required")

    # Validate all requested settings and input data before doing any expensive
    # training or writing output files.
    for name in ("epochs_joint", "epochs_split_data", "epochs_split_ode", "collocation"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain unique seed values")
    if not args.modes or len(set(args.modes)) != len(args.modes):
        parser.error("--modes must contain unique modes")
    if not np.isfinite(args.delta) or args.delta <= 0:
        parser.error("--delta must be finite and positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is not available")
    master, incidence = load_master(args.master, args.country)

    # Refit each requested mode/seed combination on this complete arm series;
    # every exported R_t value is the model's R_net output for the same date.
    records = []
    for mode in args.modes:
        for seed in args.seeds:
            fitted_incidence, rnet = fit_rt(
                incidence,
                mode,
                seed,
                args.epochs_joint,
                args.epochs_split_data,
                args.epochs_split_ode,
                args.collocation,
                args.device,
                args.delta,
            )
            for date, observed, fitted, reproduction in zip(
                master.index, incidence, fitted_incidence, rnet
            ):
                records.append(
                    {
                        "country": args.country,
                        "mode": mode,
                        "seed": seed,
                        "date": date.strftime("%Y-%m-%d"),
                        "I_proxy": observed,
                        "I_pred": fitted,
                        "Rt": reproduction,
                    }
                )

    # Build repeatable daily median/band and per-fit summaries, then persist
    # trajectories and summaries with filenames scoped to the country arm.
    trajectories = pd.DataFrame.from_records(records)
    daily_summary = summarize_by_date(trajectories)
    fit_summary = summarize_by_fit(trajectories, args.delta)
    if daily_summary.empty or fit_summary.empty:
        raise ValueError("No fitted PINN trajectories were produced")

    args.outdir.mkdir(parents=True, exist_ok=True)
    tag = args.country
    traj_out = args.outdir / f"rt_trajectories_{tag}.csv"
    med_out = args.outdir / f"rt_median_{tag}.csv"
    summary_out = args.outdir / f"rt_summary_{tag}.csv"
    fig_out = args.outdir / f"fig_rt_{tag}.png"
    trajectories.to_csv(traj_out, index=False)
    daily_summary.to_csv(med_out, index=False)
    fit_summary.to_csv(summary_out, index=False)
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--plot-only",
            "--summary-csv",
            str(med_out),
            "--plot-country",
            tag,
            "--plot-output",
            str(fig_out),
        ],
        check=True,
    )
    print(f"ok: wrote {traj_out}, {med_out}, {summary_out}, {fig_out}")


if __name__ == "__main__":
    main()
