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
import hashlib
import json
import os
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
        if "observed_simple" not in master.columns:
            raise ValueError("master file missing observed_simple; run prepare_ghana.py first")
        obs = master["observed_simple"].to_numpy()[:upto].astype(bool)
        return idx[obs]
    raise ValueError(regime)


def validate_origins(origins: list[int], horizon: int, n_days: int) -> None:
    """Ensure every train/forecast window is complete and unique."""
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if not origins:
        raise ValueError("at least one origin is required")
    if len(set(origins)) != len(origins):
        raise ValueError("origins must be unique")

    # Origin is the first forecast index; horizon days must fit after the prefix.
    invalid = [
        origin for origin in origins
        if origin <= 0 or origin + horizon > n_days
    ]
    if invalid:
        raise ValueError(
            f"origin(s) {invalid} are invalid for {n_days} master days and "
            f"horizon {horizon}; require 1 <= origin <= {n_days - horizon}"
        )


def validate_run_inputs(master: pd.DataFrame, origins: list[int], horizon: int,
                        regimes: list[str], modes: list[str], seeds: list[int]) -> np.ndarray:
    """Check Ghana master data and experiment settings before any model fits."""
    if master.empty:
        raise ValueError("master file contains no rows")
    if "I_proxy" not in master:
        raise ValueError("master file must contain an 'I_proxy' column")
    if not isinstance(master.index, pd.DatetimeIndex):
        raise ValueError("master index must contain parsed dates")
    if not master.index.is_monotonic_increasing or master.index.has_duplicates:
        raise ValueError("master dates must be unique and in increasing order")
    expected_dates = pd.date_range(master.index[0], periods=len(master), freq="D")
    if not master.index.equals(expected_dates):
        raise ValueError("master index must contain a complete, consecutive daily series")

    incidence = master["I_proxy"].to_numpy(dtype=float)
    if not np.isfinite(incidence).all():
        raise ValueError("I_proxy contains missing or non-finite values")
    if (incidence < 0).any() or incidence.max(initial=0) <= 0:
        raise ValueError("I_proxy must be non-negative with at least one positive value")

    validate_origins(origins, horizon, len(master))
    if not regimes or len(set(regimes)) != len(regimes):
        raise ValueError("regimes must be non-empty and unique")
    if not modes or len(set(modes)) != len(modes) or any(
        mode not in {"split", "joint"} for mode in modes
    ):
        raise ValueError("modes must contain only 'split' and/or 'joint'")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    if "mask" in regimes:
        if "observed_simple" not in master:
            raise ValueError("master file missing observed_simple; run prepare_ghana.py first")
        mask = master["observed_simple"].to_numpy()
        if pd.isna(mask).any() or not np.isin(mask, [0, 1]).all():
            raise ValueError("observed_simple must contain only binary 0/1 values")
    if any(regime not in {"daily", "weekly", "mask"} for regime in regimes):
        raise ValueError(f"unsupported regime(s): {regimes}")
    return incidence


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #

def run(master: pd.DataFrame, cfg: Config, origins, horizon, regimes, modes,
        seeds, out_path: Path, master_path: Path | None = None):
    """Run paired Ghana experiments and save metrics plus an internal audit."""
    # Fail before fitting rather than silently dropping incomplete forecast
    # windows or interpreting malformed report masks as valid observations.
    I_full = validate_run_inputs(master, origins, horizon, regimes, modes, seeds)
    scale = float(I_full.max())
    rows = []
    origin_audits = []

    # Fixed origins match Italy primary by default, pairing the forecast dates
    # and train-window lengths across the two countries.
    for origin in origins:
        y_train = I_full[:origin]
        y_true = I_full[origin:origin + horizon]
        regime_audits = []

        # Baselines use the full training prefix and the same held-out horizon
        # as each PINN regime at this origin.
        for name, pred in [
            ("persistence", naive_persistence(y_train, horizon)),
            ("moving_avg_7", moving_average(y_train, horizon)),
            ("classical_sir", classical_sir(y_train, horizon, cfg.delta)[0]),
        ]:
            rows.append(dict(country="ghana", origin=origin, regime="-", method=name, seed=-1,
                             mase=mase(y_true, pred, y_train),
                             mae=mae(y_true, pred), rmse=rmse(y_true, pred),
                             n_train_points=len(y_train), observed_training_pct=100.0,
                             missing_training_pct=0.0, seconds=0.0))

        # PINNs use the selected daily, weekly, or authentic reporting mask.
        for regime in regimes:
            tr_idx = regime_indices(master, regime, origin)
            observed_training_pct = 100.0 * len(tr_idx) / origin
            missing_training_pct = 100.0 - observed_training_pct
            regime_audits.append({
                "regime": regime,
                "observed_training_points": int(len(tr_idx)),
                "training_days": int(origin),
                "observed_training_pct": observed_training_pct,
                "missing_training_pct": missing_training_pct,
                "fit_eligible": bool(len(tr_idx) >= 10),
            })
            if len(tr_idx) < 10:
                rows.append(dict(
                    country="ghana", origin=origin, regime=regime,
                    method="mask_audit", seed=-1, mase=np.nan, mae=np.nan,
                    rmse=np.nan, n_train_points=len(tr_idx),
                    observed_training_pct=observed_training_pct,
                    missing_training_pct=missing_training_pct, seconds=0.0))
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
                    prediction = np.asarray(pred).reshape(-1)
                    if prediction.size != horizon or not np.isfinite(prediction).all():
                        raise ValueError(
                            f"Ghana origin={origin} regime={regime} "
                            f"mode={mode} seed={seed}: expected {horizon} finite "
                            f"predictions, got {prediction.size}"
                        )
                    rows.append(dict(
                        country="ghana", origin=origin, regime=regime,
                        method=f"pinn_{mode}",
                        seed=seed, mase=mase(y_true, prediction, y_train),
                        mae=mae(y_true, prediction), rmse=rmse(y_true, prediction),
                        n_train_points=len(tr_idx),
                        observed_training_pct=observed_training_pct,
                        missing_training_pct=missing_training_pct,
                        seconds=round(time.time() - t0, 1)))
                    print(f"  origin={origin:3d} {regime:6s} {mode:5s} seed={seed} "
                          f"MASE={rows[-1]['mase']:8.3f} ({rows[-1]['seconds']}s)",
                          flush=True)

        origin_audits.append({
            "origin": int(origin),
            "training_days": int(len(y_train)),
            "forecast_days": int(len(y_true)),
            "training_start": str(master.index[0].date()),
            "training_end": str(master.index[origin - 1].date()),
            "forecast_start": str(master.index[origin].date()),
            "forecast_end": str(master.index[origin + horizon - 1].date()),
            "regimes": regime_audits,
        })

    # Check actual output size against the planned baseline, mask-audit, and
    # model rows. A mismatch indicates a silent skip or unexpected duplication.
    df = pd.DataFrame(rows)
    expected_rows = sum(
        3 + sum(
            len(modes) * len(seeds) if item["fit_eligible"] else 1
            for item in origin_audit["regimes"]
        )
        for origin_audit in origin_audits
    )
    if len(df) != expected_rows:
        raise RuntimeError(f"result row audit failed: expected {expected_rows}, got {len(df)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    # Store protocol details separately so metric consumers retain their existing
    # CSV interface while reviewers can inspect inputs and realized regimes.
    audit = {
        "audit_status": "passed",
        "country": "ghana",
        "master_rows": int(len(master)),
        "master_date_start": str(master.index.min().date()),
        "master_date_end": str(master.index.max().date()),
        "I_proxy_min": float(I_full.min()),
        "I_proxy_max": scale,
        "I_proxy_non_finite_values": 0,
        "horizon": int(horizon),
        "origins": origin_audits,
        "regimes": list(regimes),
        "modes": list(modes),
        "seeds": [int(seed) for seed in seeds],
        "metrics_rows": int(len(df)),
        "expected_metrics_rows": int(expected_rows),
        "mask_audit_rows": int((df["method"] == "mask_audit").sum()) if not df.empty else 0,
        "metrics_csv": str(out_path),
    }
    if master_path is not None:
        audit["master_csv"] = str(master_path)
        audit["master_sha256"] = hashlib.sha256(master_path.read_bytes()).hexdigest()

    audit_path = out_path.with_suffix(".audit.json")
    tmp_path = audit_path.with_suffix(audit_path.suffix + ".tmp")
    try:
        tmp_path.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(tmp_path, audit_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    # Console summary makes protocol validation visible during a long run.
    print("\n=== GHANA RUN AUDIT ===")
    print(f"country=ghana  master_days={len(master)}  "
          f"date_range={master.index.min().date()}..{master.index.max().date()}  "
          f"horizon={horizon}")
    for origin_audit in origin_audits:
        print(f"origin={origin_audit['origin']}  "
              f"train={origin_audit['training_start']}..{origin_audit['training_end']}  "
              f"forecast={origin_audit['forecast_start']}..{origin_audit['forecast_end']}")
        for item in origin_audit["regimes"]:
            print(f"  {item['regime']}: observed={item['observed_training_points']}/"
                  f"{item['training_days']} missing={item['missing_training_pct']:.1f}% "
                  f"fit_eligible={item['fit_eligible']}")
    print(f"result_rows={len(df)} (expected={expected_rows})")
    print(f"audit_status=passed  audit_json={audit_path}")
    return df


def main():
    """Parse runner options, validate the master, and execute matched origins."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=Path, default=Path("data/processed/ghana_national_master.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/tables/ghana_results.csv"))
    # These are the same train-window lengths used by Italy primary by default.
    ap.add_argument("--origins", type=int, nargs="+", default=[45, 60, 75])
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--regimes", nargs="+", default=["daily", "weekly", "mask"])
    ap.add_argument("--modes", nargs="+", default=["split", "joint"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[34])
    ap.add_argument("--epochs-joint", type=int, default=5000)
    ap.add_argument("--epochs-split-data", type=int, default=3000)
    ap.add_argument("--epochs-split-ode", type=int, default=1000)
    ap.add_argument("--collocation", type=int, default=6000)
    a = ap.parse_args()

    # Load and check all protocol inputs before creating a model or output file.
    master = pd.read_csv(a.master, index_col=0, parse_dates=True)
    try:
        validate_run_inputs(master, a.origins, a.horizon, a.regimes, a.modes, a.seeds)
    except ValueError as exc:
        ap.error(str(exc))

    cfg = Config(epochs_joint=a.epochs_joint,
                 epochs_split_data=a.epochs_split_data,
                 epochs_split_ode=a.epochs_split_ode,
                 n_collocation=a.collocation)

    df = run(master, cfg, a.origins, a.horizon, a.regimes, a.modes, a.seeds,
             a.out, a.master)
    print("\n=== MEDIAN MASE BY METHOD x REGIME ===")
    print(df.pivot_table(index="method", columns="regime",
                         values="mase", aggfunc="median").round(3).to_string())


if __name__ == "__main__":
    main()
