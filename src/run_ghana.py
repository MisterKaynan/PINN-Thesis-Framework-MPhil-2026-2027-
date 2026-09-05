#!/usr/bin/env python3
"""
Ghana experiments: reduced SIR PINN under three observation regimes.

RESEARCH QUESTION
    Millevoi et al. (2024) Case 4 observes that under large data uncertainty the
    SPLIT approach loses reliability, because it leans entirely on the data,
    while the JOINT approach compensates through the ODE residual. They saw this
    under 40% synthetic Gaussian noise and did not pursue it.

    We test that conjecture on authentic low-fidelity surveillance data.

OBSERVATION REGIMES
    daily   Every day is a training point.                     [dense reference]
    weekly  Proxy sampled every 7th day. The proxy is a trailing
            rolling sum, i.e. prevalence-like, so subsampling is
            valid here; raw incidence would have to be summed.
    mask    Training points only where Ghana actually reported.
            117 of 434 days carry no report (27.0%), in 70 separate
            gaps, the longest 14 days. No prior PINN-epidemic paper
            runs this condition: sparsity is inherited, not simulated.

FORECAST PROTOCOL
    Expanding-origin, following the paper's Sec 3.2.2: train on [0, t_k], then
    extrapolate the trained networks H days beyond the window. Origins are fixed
    across every method and regime so comparisons are paired.

METRICS
    MASE is primary: scale-free, and stable across Ghana's zero-heavy troughs
    where MAPE is undefined or explosive. The naive denominator is computed on
    the TRAINING window only, never on data the model has seen at evaluation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pinn import Config, ReducedSIRPINN


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def mase(y_true, y_pred, y_train) -> float:
    """Mean absolute scaled error. Denominator is the in-sample naive forecast."""
    denom = np.mean(np.abs(np.diff(y_train)))
    if denom <= 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / denom)


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #

def naive_persistence(y_train, horizon):
    """Carry the last observed value forward."""
    return np.repeat(y_train[-1], horizon)


def moving_average(y_train, horizon, window=7):
    """Flat forecast at the trailing mean."""
    return np.repeat(np.mean(y_train[-window:]), horizon)


def classical_sir(y_train, horizon, delta=0.2):
    """
    Reduced SIR integrated forward with a single constant R_t, fitted by least
    squares on the training window. This is the classical mechanistic
    comparator: same physics as the PINN, no neural parameterisation, and no
    capacity to let R_t vary in time.
    """
    from scipy.optimize import minimize_scalar

    def simulate(R, n, I0):
        I = np.empty(n)
        I[0] = I0
        for k in range(1, n):
            I[k] = max(I[k - 1] + delta * (R - 1.0) * I[k - 1], 0.0)
        return I

    def sse(R):
        return np.sum((simulate(R, len(y_train), y_train[0]) - y_train) ** 2)

    R_hat = minimize_scalar(sse, bounds=(0.1, 4.0), method="bounded").x
    full = simulate(R_hat, len(y_train) + horizon, y_train[0])
    return full[len(y_train):], float(R_hat)


# --------------------------------------------------------------------------- #
# Regimes
# --------------------------------------------------------------------------- #

def regime_indices(master: pd.DataFrame, regime: str, upto: int) -> np.ndarray:
    """Which day indices in [0, upto) serve as PINN training points."""
    idx = np.arange(upto)
    if regime == "daily":
        return idx
    if regime == "weekly":
        return idx[::7]
    if regime == "mask":
        obs = master["observed_simple"].to_numpy()[:upto].astype(bool)
        return idx[obs]
    raise ValueError(regime)


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #

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

        # --- baselines: regime-independent, computed once per origin --------
        for name, pred in [
            ("persistence", naive_persistence(y_train, horizon)),
            ("moving_avg_7", moving_average(y_train, horizon)),
            ("classical_sir", classical_sir(y_train, horizon, cfg.delta)[0]),
        ]:
            rows.append(dict(origin=origin, regime="-", method=name, seed=-1,
                             mase=mase(y_true, pred, y_train),
                             mae=mae(y_true, pred), rmse=rmse(y_true, pred),
                             n_train_points=len(y_train), seconds=0.0))

        # --- PINN across regimes and modes ----------------------------------
        for regime in regimes:
            tr_idx = regime_indices(master, regime, origin)
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
                    t0 = time.time()
                    if mode == "joint":
                        m.fit_joint(t_data, I_obs)
                    else:
                        m.fit_split(t_data, I_obs)
                    pred, _ = m.predict(t_fut)
                    rows.append(dict(
                        origin=origin, regime=regime, method=f"pinn_{mode}",
                        seed=seed, mase=mase(y_true, pred, y_train),
                        mae=mae(y_true, pred), rmse=rmse(y_true, pred),
                        n_train_points=len(tr_idx),
                        seconds=round(time.time() - t0, 1)))
                    print(f"  origin={origin:3d} {regime:6s} {mode:5s} seed={seed} "
                          f"MASE={rows[-1]['mase']:8.3f} ({rows[-1]['seconds']}s)",
                          flush=True)

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=Path, default=Path("data/processed/ghana_national_master.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/tables/ghana_results.csv"))
    ap.add_argument("--origins", type=int, nargs="+", default=[180])
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--regimes", nargs="+", default=["daily", "weekly", "mask"])
    ap.add_argument("--modes", nargs="+", default=["split", "joint"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[34])
    ap.add_argument("--epochs-joint", type=int, default=5000)
    ap.add_argument("--epochs-split-data", type=int, default=3000)
    ap.add_argument("--epochs-split-ode", type=int, default=1000)
    ap.add_argument("--collocation", type=int, default=6000)
    a = ap.parse_args()

    master = pd.read_csv(a.master, index_col=0, parse_dates=True)
    cfg = Config(epochs_joint=a.epochs_joint,
                 epochs_split_data=a.epochs_split_data,
                 epochs_split_ode=a.epochs_split_ode,
                 n_collocation=a.collocation)

    df = run(master, cfg, a.origins, a.horizon, a.regimes, a.modes, a.seeds, a.out)
    print("\n=== MEDIAN MASE BY METHOD x REGIME ===")
    print(df.pivot_table(index="method", columns="regime",
                         values="mase", aggfunc="median").round(3).to_string())


if __name__ == "__main__":
    main()
