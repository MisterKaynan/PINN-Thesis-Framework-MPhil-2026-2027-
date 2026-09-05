#!/usr/bin/env python3
"""
Parallel grid runner for multi-core CPU machines.

WHY THIS EXISTS
    The workload is embarrassingly parallel: every (origin, regime, mode, seed)
    fit is independent. The networks are also tiny (4x50 and 4x100) with a
    mini-batch of 100, so a single fit cannot saturate even one core's vector
    units, and PyTorch's intra-op threading gives almost nothing.

    Running N independent fits across N cores therefore scales close to
    linearly, which is far more effective here than either intra-op threading
    or a GPU.

    Each worker is pinned to ONE torch thread deliberately. Without this,
    N workers each spawn N threads, oversubscribe the machine and run slower
    than a single process.

Usage
    python src/run_parallel.py --workers 8 --out results/tables/grid.csv
    python src/run_parallel.py --workers 4 --seeds 1 2 3 --quick
"""

from __future__ import annotations

import argparse
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Must be set before torch is imported in the worker processes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd


def _one_fit(task):
    """Run a single (origin, regime, mode, seed) fit. Executed in a subprocess."""
    import torch
    torch.set_num_threads(1)

    from pinn import Config, ReducedSIRPINN
    from run_ghana import regime_indices, mase, mae, rmse

    (origin, regime, mode, seed, horizon, master_path, epochs) = task
    master = pd.read_csv(master_path, index_col=0, parse_dates=True)
    I_full = master["I_proxy"].to_numpy(float)
    scale = float(I_full.max())

    y_train = I_full[:origin]
    y_true = I_full[origin:origin + horizon]
    if len(y_true) < horizon:
        return None

    tr_idx = regime_indices(master, regime, origin)
    if len(tr_idx) < 10:
        return None

    t_data = torch.tensor(tr_idx / origin, dtype=torch.float32).reshape(-1, 1)
    I_obs = torch.tensor(I_full[tr_idx] / scale, dtype=torch.float32).reshape(-1, 1)
    t_fut = torch.tensor(np.arange(origin, origin + horizon) / origin,
                         dtype=torch.float32).reshape(-1, 1)

    cfg = Config(seed=seed, device="cpu",
                 epochs_joint=epochs["joint"],
                 epochs_split_data=epochs["split_data"],
                 epochs_split_ode=epochs["split_ode"],
                 n_collocation=epochs["collocation"])

    m = ReducedSIRPINN(cfg, tf=origin, scale_I=scale)
    t0 = time.time()
    if mode == "joint":
        m.fit_joint(t_data, I_obs)
    else:
        m.fit_split(t_data, I_obs)
    pred, R_pred = m.predict(t_fut)

    return dict(origin=origin, regime=regime, method=f"pinn_{mode}", seed=seed,
                mase=mase(y_true, pred, y_train), mae=mae(y_true, pred),
                rmse=rmse(y_true, pred), n_train_points=len(tr_idx),
                seconds=round(time.time() - t0, 1),
                mean_Rt=float(np.mean(R_pred)))


def baselines(master_path, origins, horizon):
    """Baselines are cheap and deterministic; compute them in the parent."""
    from run_ghana import (naive_persistence, moving_average, classical_sir,
                           mase, mae, rmse)
    master = pd.read_csv(master_path, index_col=0, parse_dates=True)
    I_full = master["I_proxy"].to_numpy(float)
    rows = []
    for origin in origins:
        y_train, y_true = I_full[:origin], I_full[origin:origin + horizon]
        if len(y_true) < horizon:
            continue
        for name, pred in [
            ("persistence", naive_persistence(y_train, horizon)),
            ("moving_avg_7", moving_average(y_train, horizon)),
            ("classical_sir", classical_sir(y_train, horizon)[0]),
        ]:
            rows.append(dict(origin=origin, regime="-", method=name, seed=-1,
                             mase=mase(y_true, pred, y_train),
                             mae=mae(y_true, pred), rmse=rmse(y_true, pred),
                             n_train_points=len(y_train), seconds=0.0,
                             mean_Rt=np.nan))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default="data/processed/ghana_national_master.csv")
    ap.add_argument("--out", default="results/tables/grid.csv")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--origins", type=int, nargs="+",
                    default=[100, 140, 180, 220, 260, 300, 340, 380])
    ap.add_argument("--regimes", nargs="+", default=["daily", "weekly", "mask"])
    ap.add_argument("--modes", nargs="+", default=["split", "joint"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--quick", action="store_true",
                    help="Reduced epochs for a pipeline smoke test. Never for results.")
    ap.add_argument("--shard", type=int, default=0,
                    help="This machine's index when splitting work across PCs (0-based).")
    ap.add_argument("--of", type=int, default=1,
                    help="Total number of machines sharing the grid.")
    a = ap.parse_args()

    epochs = (dict(joint=600, split_data=600, split_ode=200, collocation=2000)
              if a.quick else
              dict(joint=5000, split_data=3000, split_ode=1000, collocation=6000))

    tasks = [(o, g, m, s, a.horizon, a.master, epochs)
             for o, g, m, s in itertools.product(a.origins, a.regimes,
                                                 a.modes, a.seeds)]

    # Shard across machines. Joint fits cost ~8x a split fit, so a naive
    # contiguous split would leave one machine idle for hours. Sorting by cost
    # and dealing round-robin balances the load.
    if a.of > 1:
        tasks.sort(key=lambda t: (t[2] != "joint", t[0]))
        tasks = tasks[a.shard::a.of]
        print(f"SHARD {a.shard + 1} of {a.of}")

    n_joint = sum(1 for t in tasks if t[2] == "joint")
    est = (n_joint * 29 + (len(tasks) - n_joint) * 3.5) / max(a.workers, 1) / 60
    print(f"{len(tasks)} fits ({n_joint} joint) across {a.workers} workers "
          f"({'QUICK' if a.quick else 'FULL'} epochs)")
    print(f"estimated wall-clock: {est:.1f} h")

    rows = baselines(a.master, a.origins, a.horizon)
    done, t0 = 0, time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_one_fit, t): t for t in tasks}
        for f in as_completed(futs):
            r = f.result()
            done += 1
            if r:
                rows.append(r)
                el = time.time() - t0
                eta = el / done * (len(tasks) - done)
                print(f"[{done:4d}/{len(tasks)}] o={r['origin']:3d} "
                      f"{r['regime']:6s} {r['method']:10s} s={r['seed']} "
                      f"MASE={r['mase']:7.2f}  ETA {eta/60:5.1f} min", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"\nWrote {out}  ({len(df)} rows, {(time.time()-t0)/60:.1f} min)")

    pinn = df[df.method.str.startswith("pinn")]
    if len(pinn):
        print("\nMedian MASE by method x regime:")
        print(pinn.pivot_table(index="method", columns="regime",
                               values="mase", aggfunc="median").round(3).to_string())


if __name__ == "__main__":
    main()
