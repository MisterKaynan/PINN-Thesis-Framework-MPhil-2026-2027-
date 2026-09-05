#!/usr/bin/env python3
"""
Extract the inferred effective reproduction number R_t and compare it against
published estimates for Ghana.

WHY THIS EXISTS
    Forecast error alone cannot tell you whether a model has learned epidemic
    dynamics or merely learned to extrapolate a curve. A model with acceptable
    MASE and an implausible R_t has failed at the task that matters, and that
    failure is invisible in Table 1 of the report.

    This fits on the FULL series rather than at forecast origins, following
    Millevoi et al.'s Cases 6-7, which use all available data to infer past
    dynamics retrospectively. That is the correct setting for validating R_t.

WHAT YOU MUST SUPPLY
    Dwomoh et al. (2021) report reproduction numbers for Ghana's early waves.
    Their values are NOT hard-coded here, because inventing reference numbers
    would defeat the purpose of the check. Fill in DWOMOH_REFERENCE below from
    the paper before running the comparison. Leave it empty and the script still
    produces our trajectory, just without the overlay.

USAGE
    PYTHONPATH=src python src/extract_rt.py --modes split joint --seeds 1 2 3
    PYTHONPATH=src python src/extract_rt.py --modes split --seeds 1 --quick
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# FILL THIS IN from Dwomoh et al. (2021), "Mathematical modeling of COVID-19
# infection dynamics in Ghana", Infectious Disease Modelling 6:381-397.
#
# Format: (date string, R_t estimate). Add as many points as the paper gives.
# Example shape only -- these are NOT real values, replace them:
#     ("2020-03-30", 3.10),
#     ("2020-05-15", 1.40),
#
# If the paper reports a range or a figure rather than a table, take the values
# you can defend and say in the report how you read them off.
# ---------------------------------------------------------------------------
DWOMOH_REFERENCE: list[tuple[str, float]] = [
    # ("YYYY-MM-DD", value),
]


def _fit_one(task):
    """Fit on the full series and return the R_t trajectory."""
    import torch
    torch.set_num_threads(1)
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from pinn import Config, ReducedSIRPINN

    mode, seed, epochs, master_path, win = task
    master = pd.read_csv(master_path, index_col=0, parse_dates=True)
    if win:
        master = master.iloc[win[0]:win[1]]
    I_full = master["I_proxy"].to_numpy(float)
    scale = float(I_full.max())
    n = len(I_full)

    t = np.arange(n, dtype=float) / n
    t_data = torch.tensor(t, dtype=torch.float32).reshape(-1, 1)
    I_obs = torch.tensor(I_full / scale, dtype=torch.float32).reshape(-1, 1)

    cfg = Config(seed=seed, device="cpu",
                 epochs_joint=epochs["joint"],
                 epochs_split_data=epochs["split_data"],
                 epochs_split_ode=epochs["split_ode"],
                 n_collocation=epochs["collocation"])
    m = ReducedSIRPINN(cfg, tf=n, scale_I=scale)
    t0 = time.time()
    if mode == "joint":
        m.fit_joint(t_data, I_obs)
    else:
        m.fit_split(t_data, I_obs)
    I_pred, Rt = m.predict(t_data)

    return dict(mode=mode, seed=seed, seconds=round(time.time() - t0, 1),
                dates=[str(d.date()) for d in master.index],
                Rt=Rt.tolist(), I_pred=I_pred.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=str(ROOT / "data/processed/ghana_national_master.csv"))
    ap.add_argument("--modes", nargs="+", default=["split", "joint"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    ap.add_argument("--window", type=int, default=0,
                    help="Fit on a sub-window of N days instead of the full series. "
                         "Millevoi et al. use 90-120 day windows.")
    ap.add_argument("--start", type=int, default=0,
                    help="Day index at which the sub-window begins.")
    ap.add_argument("--quick", action="store_true",
                    help="Reduced epochs. Pipeline test only, never for the report.")
    a = ap.parse_args()

    epochs = (dict(joint=600, split_data=600, split_ode=200, collocation=2000)
              if a.quick else
              dict(joint=5000, split_data=3000, split_ode=1000, collocation=6000))

    win = (a.start, a.start + a.window) if a.window else None
    tasks = [(m, s, epochs, a.master, win) for m in a.modes for s in a.seeds]
    n_joint = sum(1 for t in tasks if t[0] == "joint")
    print(f"{len(tasks)} full-series fits ({n_joint} joint) on {a.workers} workers")
    print(f"estimated: {(n_joint*40 + (len(tasks)-n_joint)*7)/min(a.workers,len(tasks)):.0f} min")

    results = []
    with ProcessPoolExecutor(max_workers=min(a.workers, len(tasks))) as ex:
        futs = {ex.submit(_fit_one, t): t for t in tasks}
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            print(f"  {r['mode']:5s} seed={r['seed']} done ({r['seconds']}s) "
                  f"mean Rt={np.mean(r['Rt']):.2f}", flush=True)

    outdir = Path(a.outdir)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)

    # Long-format table: one row per (mode, seed, date).
    rows = []
    for r in results:
        for d, rt, ip in zip(r["dates"], r["Rt"], r["I_pred"]):
            rows.append(dict(mode=r["mode"], seed=r["seed"], date=d,
                             Rt=rt, I_pred=ip))
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "tables" / "rt_trajectories.csv", index=False)

    # Median across seeds, per mode.
    med = df.groupby(["mode", "date"]).Rt.median().reset_index()
    med.to_csv(outdir / "tables" / "rt_median.csv", index=False)
    print(f"\nwrote {outdir/'tables'/'rt_trajectories.csv'}")

    plot(df, outdir, a.master)


def plot(df, outdir, master_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    master = pd.read_csv(master_path, index_col=0, parse_dates=True)
    df["date"] = pd.to_datetime(df["date"])

    fig, ax = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1.4], "hspace": 0.12})

    ax[0].fill_between(master.index, master.I_proxy, color="#c8d8e8", lw=0)
    ax[0].set_ylabel("Infection proxy")
    ax[0].set_title("Inferred effective reproduction number, Ghana",
                    fontsize=10, loc="left")

    colours = {"split": "#2c5f8a", "joint": "#c1553b"}
    for mode, g in df.groupby("mode"):
        piv = g.pivot_table(index="date", columns="seed", values="Rt")
        med = piv.median(axis=1)
        ax[1].fill_between(piv.index, piv.min(axis=1), piv.max(axis=1),
                           color=colours.get(mode, "grey"), alpha=0.18, lw=0)
        ax[1].plot(piv.index, med, color=colours.get(mode, "grey"),
                   lw=1.6, label=f"PINN {mode}")

    if DWOMOH_REFERENCE:
        d = [pd.Timestamp(x) for x, _ in DWOMOH_REFERENCE]
        v = [y for _, y in DWOMOH_REFERENCE]
        ax[1].plot(d, v, "k^--", ms=6, lw=1.2, label="Dwomoh et al. (2021)")
    else:
        ax[1].text(0.5, 0.94, "Dwomoh reference values not supplied "
                              "— see DWOMOH_REFERENCE in extract_rt.py",
                   transform=ax[1].transAxes, ha="center", fontsize=7.5,
                   color="#a33", style="italic")

    ax[1].axhline(1.0, color="k", lw=0.9, ls=":")
    ax[1].text(master.index[5], 1.03, r"$\mathcal{R}_t=1$", fontsize=7.5)
    ax[1].set_ylabel(r"$\mathcal{R}_t$")
    ax[1].set_xlabel("Date")
    ax[1].legend(frameon=False, fontsize=8)
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)

    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    dest = outdir / "figures" / "fig_rt.png"
    plt.savefig(dest, dpi=180, bbox_inches="tight")
    print(f"wrote {dest}")

    # Quantitative summary for the report.
    print("\n=== R_t summary by mode ===")
    for mode, g in df.groupby("mode"):
        med = g.groupby("date").Rt.median()
        above = (med > 1).mean() * 100
        print(f"  {mode:5s}: median {med.median():.2f}, "
              f"range {med.min():.2f}-{med.max():.2f}, "
              f"{above:.0f}% of days above 1")


if __name__ == "__main__":
    main()
