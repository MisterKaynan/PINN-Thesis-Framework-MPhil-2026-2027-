#!/usr/bin/env python3
"""Resumable, checkpointed PINN grid runner.

Run one independent fit at a time and atomically append its result to --out.
If the runtime disconnects, upload/download the CSV and re-run the exact same
command: completed (origin, regime, mode, seed) PINN fits are skipped.

This is deliberately task-level checkpointing, rather than mid-epoch model
checkpointing. The experiment's scientific output is the grid of completed
fits; its tasks are independent, and a partial neural-network fit is not a
valid result. Restarting only the interrupted fit prevents loss of completed
work without mixing incompatible optimizer states.

Requires: pinn.py and run_ghana.py on PYTHONPATH.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pinn import Config, ReducedSIRPINN
from run_ghana import (
    classical_sir,
    mae,
    mase,
    moving_average,
    naive_persistence,
    regime_indices,
    rmse,
)

KEY = ["origin", "regime", "method", "seed"]


def atomic_write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def load_completed(path: Path) -> tuple[pd.DataFrame, set[tuple]]:
    if not path.exists():
        return pd.DataFrame(), set()
    df = pd.read_csv(path)
    missing = set(KEY) - set(df.columns)
    if missing:
        raise SystemExit(
            f"Existing checkpoint {path} lacks required columns {sorted(missing)}. "
            "Use a new --out filename or restore a compatible checkpoint CSV."
        )
    pinn = df[df["method"].astype(str).str.startswith("pinn_")]
    completed = set(map(tuple, pinn[KEY].itertuples(index=False, name=None)))
    return df, completed


def baseline_rows(I_full: np.ndarray, origins: list[int], horizon: int, delta: float) -> list[dict]:
    rows = []
    for origin in origins:
        y_train = I_full[:origin]
        y_true = I_full[origin:origin + horizon]
        if len(y_true) < horizon:
            continue
        for name, pred in [
            ("persistence", naive_persistence(y_train, horizon)),
            ("moving_avg_7", moving_average(y_train, horizon)),
            ("classical_sir", classical_sir(y_train, horizon, delta)[0]),
        ]:
            rows.append(dict(
                origin=origin, regime="-", method=name, seed=-1,
                mase=mase(y_true, pred, y_train), mae=mae(y_true, pred),
                rmse=rmse(y_true, pred), n_train_points=len(y_train),
                seconds=0.0,
            ))
    return rows


def add_baselines_once(df: pd.DataFrame, I_full: np.ndarray, origins: list[int], horizon: int, delta: float, out: Path) -> pd.DataFrame:
    existing = set(map(tuple, df[df["method"].isin(["persistence", "moving_avg_7", "classical_sir"])][KEY].itertuples(index=False, name=None))) if not df.empty else set()
    new = [r for r in baseline_rows(I_full, origins, horizon, delta) if (r["origin"], r["regime"], r["method"], r["seed"]) not in existing]
    if new:
        df = pd.concat([df, pd.DataFrame(new)], ignore_index=True)
        atomic_write(df, out)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="Persistent checkpoint/result CSV. Download or sync this file before a runtime reset.")
    ap.add_argument("--origins", type=int, nargs="+", required=True)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--regimes", nargs="+", required=True)
    ap.add_argument("--modes", nargs="+", default=["split", "joint"], choices=["split", "joint"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--epochs-joint", type=int, default=5000)
    ap.add_argument("--epochs-split-data", type=int, default=3000)
    ap.add_argument("--epochs-split-ode", type=int, default=1000)
    ap.add_argument("--collocation", type=int, default=6000)
    ap.add_argument("--max-fits", type=int, default=None,
                    help="Stop cleanly after this many newly completed PINN fits; useful for planned free-tier sessions.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Discard an existing checkpoint CSV and start this output afresh.")
    args = ap.parse_args()

    if args.overwrite and args.out.exists():
        args.out.unlink()

    master = pd.read_csv(args.master, index_col=0, parse_dates=True)
    if "I_proxy" not in master.columns:
        raise SystemExit("Expected I_proxy in --master. Run preprocessing first.")
    I_full = master["I_proxy"].to_numpy(float)
    scale = float(I_full.max())
    if scale <= 0:
        raise SystemExit("I_proxy has no positive values; cannot fit the PINN.")

    cfg = Config(
        epochs_joint=args.epochs_joint,
        epochs_split_data=args.epochs_split_data,
        epochs_split_ode=args.epochs_split_ode,
        n_collocation=args.collocation,
    )

    df, completed = load_completed(args.out)
    df = add_baselines_once(df, I_full, args.origins, args.horizon, cfg.delta, args.out)

    tasks = [
        (origin, regime, mode, seed)
        for origin in args.origins
        for regime in args.regimes
        for mode in args.modes
        for seed in args.seeds
        if (origin, regime, f"pinn_{mode}", seed) not in completed
    ]
    print(f"Checkpoint: {args.out}")
    print(f"Completed PINN fits: {len(completed)} | Remaining: {len(tasks)}")

    newly_done = 0
    for origin, regime, mode, seed in tasks:
        if args.max_fits is not None and newly_done >= args.max_fits:
            break
        y_train = I_full[:origin]
        y_true = I_full[origin:origin + args.horizon]
        if len(y_true) < args.horizon:
            print(f"SKIP origin={origin}: insufficient future horizon", flush=True)
            continue
        tr_idx = regime_indices(master, regime, origin)
        if len(tr_idx) < 10:
            print(f"SKIP origin={origin} regime={regime}: fewer than 10 training points", flush=True)
            continue

        t_data = torch.tensor(tr_idx / origin, dtype=torch.float32).reshape(-1, 1)
        I_obs = torch.tensor(I_full[tr_idx] / scale, dtype=torch.float32).reshape(-1, 1)
        t_fut = torch.tensor(np.arange(origin, origin + args.horizon) / origin, dtype=torch.float32).reshape(-1, 1)

        run_cfg = Config(**{**cfg.__dict__, "seed": seed})
        model = ReducedSIRPINN(run_cfg, tf=origin, scale_I=scale)
        t0 = time.time()
        if mode == "joint":
            model.fit_joint(t_data, I_obs)
        else:
            model.fit_split(t_data, I_obs)
        pred, _ = model.predict(t_fut)

        row = dict(
            origin=origin, regime=regime, method=f"pinn_{mode}", seed=seed,
            mase=mase(y_true, pred, y_train), mae=mae(y_true, pred),
            rmse=rmse(y_true, pred), n_train_points=len(tr_idx),
            seconds=round(time.time() - t0, 1),
        )
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        atomic_write(df, args.out)
        newly_done += 1
        print(
            f"SAVED {newly_done:3d} | origin={origin:3d} {regime:9s} {mode:5s} "
            f"seed={seed} MASE={row['mase']:.3f} ({row['seconds']}s)",
            flush=True,
        )

    remaining = len(tasks) - newly_done
    print(f"Finished this session: {newly_done} new fit(s); remaining from this invocation: {remaining}")
    print(f"Checkpoint is current: {args.out}")


if __name__ == "__main__":
    main()
