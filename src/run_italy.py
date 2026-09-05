#!/usr/bin/env python3
"""
Italy experiments: reduced SIR PINN across daily / weekly / synthetic-sparse
regimes. Same interface and metric conventions as run_ghana.py, so results
merge cleanly with the Ghana grid for the equity-metrics stage.

REGIMES
    daily     Every day is a training point (Millevoi's original regime).
    weekly    Rolling-sum proxy sampled every 7th day, matching Ghana's
              subsampling convention (see run_ghana.py:regime_indices).
    sparseNN  NN% of days dropped at random with a fixed per-regime seed
              (sparsify.py). Interpolates Italy toward Ghana-like sparsity
              under a CONTROLLED, uniform-random missingness mechanism --
              in contrast to Ghana's authentic, clustered/batched gaps.

This lets the equity-metrics stage ask a genuinely new question: does the
model degrade the same way under an equivalent PERCENTAGE of missingness when
that missingness is random (Italy synthetic) vs clustered (Ghana authentic)?
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pinn import Config, ReducedSIRPINN
from run_ghana import mase, mae, rmse, naive_persistence, moving_average, classical_sir


def regime_indices_italy(master: pd.DataFrame, regime: str, upto: int) -> np.ndarray:
    idx = np.arange(upto)
    if regime == "daily":
        return idx
    if regime == "weekly":
        return idx[::7]
    if regime.startswith("sparse"):
        col = f"observed_{regime}"
        if col not in master.columns:
            raise ValueError(f"master file missing column {col}; run sparsify.py first")
        obs = master[col].to_numpy()[:upto].astype(bool)
        return idx[obs]
    raise ValueError(regime)


def run(master: pd.DataFrame, cfg: Config, origins, horizon, regimes, modes,
        seeds, out_path: Path):
    I_full = master["I_proxy"].to_numpy(float)
    scale = float(I_full.max())
    rows = []

    for origin in origins:
        y_train = I_full[:origin]
        y_true = I_full[origin:origin + horizon]
        if len(y_true) < horizon:
            continue

        for name, pred in [
            ("persistence", naive_persistence(y_train, horizon)),
            ("moving_avg_7", moving_average(y_train, horizon)),
            ("classical_sir", classical_sir(y_train, horizon, cfg.delta)[0]),
        ]:
            rows.append(dict(country="italy", origin=origin, regime="-", method=name,
                             seed=-1, mase=mase(y_true, pred, y_train),
                             mae=mae(y_true, pred), rmse=rmse(y_true, pred),
                             n_train_points=len(y_train), seconds=0.0))

        for regime in regimes:
            tr_idx = regime_indices_italy(master, regime, origin)
            if len(tr_idx) < 10:
                continue
            t_data = torch.tensor(tr_idx / origin, dtype=torch.float32).reshape(-1, 1)
            I_obs = torch.tensor(I_full[tr_idx] / scale, dtype=torch.float32).reshape(-1, 1)
            t_fut = torch.tensor(
                np.arange(origin, origin + horizon) / origin, dtype=torch.float32
            ).reshape(-1, 1)

            for mode in modes:
                for seed in seeds:
                    c = Config(**{**cfg.__dict__, "seed": seed})
                    m = ReducedSIRPINN(c, tf=origin, scale_I=scale)
                    if mode == "joint":
                        m.fit_joint(t_data, I_obs)
                    else:
                        m.fit_split(t_data, I_obs)
                    pred, _ = m.predict(t_fut)
                    rows.append(dict(
                        country="italy", origin=origin, regime=regime,
                        method=f"pinn_{mode}", seed=seed,
                        mase=mase(y_true, pred, y_train), mae=mae(y_true, pred),
                        rmse=rmse(y_true, pred), n_train_points=len(tr_idx),
                        seconds=0.0))
                    print(f"  origin={origin:3d} {regime:9s} {mode:5s} seed={seed} "
                          f"MASE={rows[-1]['mase']:8.3f}", flush=True)

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=Path, default=Path("data/processed/italy_national_master.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/tables/italy_grid.csv"))
    ap.add_argument("--origins", type=int, nargs="+", default=[60, 75, 90, 105])
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--regimes", nargs="+",
                    default=["daily", "weekly", "sparse20", "sparse30", "sparse40"])
    ap.add_argument("--modes", nargs="+", default=["split", "joint"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--epochs-joint", type=int, default=5000)
    ap.add_argument("--epochs-split-data", type=int, default=3000)
    ap.add_argument("--epochs-split-ode", type=int, default=1000)
    ap.add_argument("--collocation", type=int, default=6000)
    a = ap.parse_args()

    master = pd.read_csv(a.master, index_col=0, parse_dates=True)
    cfg = Config(epochs_joint=a.epochs_joint, epochs_split_data=a.epochs_split_data,
                 epochs_split_ode=a.epochs_split_ode, n_collocation=a.collocation)

    df = run(master, cfg, a.origins, a.horizon, a.regimes, a.modes, a.seeds, a.out)
    print("\n=== MEDIAN MASE BY METHOD x REGIME (ITALY) ===")
    print(df.pivot_table(index="method", columns="regime", values="mase",
                         aggfunc="median").round(3).to_string())


if __name__ == "__main__":
    main()
