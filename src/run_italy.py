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
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pinn import Config, ReducedSIRPINN
from run_ghana import mase, mae, rmse, naive_persistence, moving_average, classical_sir


def regime_indices_italy(master: pd.DataFrame, regime: str, upto: int) -> np.ndarray:
    """Return full training indices, subsampled by the requested regime."""
    idx = np.arange(upto)
    if regime == "daily":
        return idx
    if regime == "weekly":
        return idx[::7]
    if regime.startswith("sparse"):
        col = f"observed_{regime}"
        if col not in master.columns:
            raise ValueError(f"master file missing column {col}; run sparsify.py first")
        # Slicing at the origin intentionally permits prefix rates to vary;
        # the exact observed count and realized percentage are recorded below.
        obs = master[col].to_numpy()[:upto].astype(bool)
        return idx[obs]
    raise ValueError(regime)


def validate_origins(origins: list[int], horizon: int, n_days: int) -> None:
    """Reject origins whose full training and forecast windows do not fit."""
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if not origins:
        raise ValueError("at least one origin is required")
    if len(set(origins)) != len(origins):
        raise ValueError("origins must be unique")

    # Origin is the number of training days; forecast indices are
    # [origin, origin + horizon), so the full interval must fit in the master.
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
    """Check the data and run settings before fitting any models."""
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

    # Validate each requested sparse column before model work; only binary
    # indicators make the observed-count and realized-rate audit meaningful.
    for regime in regimes:
        if regime in {"daily", "weekly"}:
            continue
        if not regime.startswith("sparse") or not regime[len("sparse"):].isdigit():
            raise ValueError(f"unsupported regime: {regime}")
        pct = int(regime[len("sparse"):])
        if not 0 < pct < 100:
            raise ValueError(f"sparse percentage must be between 1 and 99: {regime}")
        column = f"observed_{regime}"
        if column not in master:
            raise ValueError(f"master file missing column {column}; run sparsify.py first")
        mask = master[column].to_numpy()
        if pd.isna(mask).any() or not np.isin(mask, [0, 1]).all():
            raise ValueError(f"{column} must contain only binary 0/1 values")
    return incidence


def run(master: pd.DataFrame, cfg: Config, country: str, origins, horizon,
        regimes, modes, seeds, out_path: Path, master_path: Path | None = None):
    """Fit requested Italy regimes and save metrics with per-origin mask density."""
    # Validate the evaluation window once rather than silently skipping origins
    # that cannot produce the requested complete forecast horizon.
    if country not in {"italy_primary", "italy_secondary"}:
        raise ValueError(f"country must be italy_primary or italy_secondary, got {country}")
    I_full = validate_run_inputs(master, origins, horizon, regimes, modes, seeds)
    scale = float(I_full.max())
    rows = []
    origin_audits = []

    # Each origin divides the series into a training prefix and a fixed-length
    # held-out forecast window.
    for origin in origins:
        y_train = I_full[:origin]
        y_true = I_full[origin:origin + horizon]
        regime_audits = []

        # Baselines use the complete prefix, providing the same target window
        # against which every observation regime is compared.
        for name, pred in [
            ("persistence", naive_persistence(y_train, horizon)),
            ("moving_avg_7", moving_average(y_train, horizon)),
            ("classical_sir", classical_sir(y_train, horizon, cfg.delta)[0]),
        ]:
            rows.append(dict(country=country, origin=origin, regime="-", method=name,
                             seed=-1, mase=mase(y_true, pred, y_train),
                             mae=mae(y_true, pred), rmse=rmse(y_true, pred),
                             n_train_points=len(y_train), requested_missing_pct=None,
                             realized_missing_pct=0.0, seconds=0.0))

        for regime in regimes:
            tr_idx = regime_indices_italy(master, regime, origin)
            # Report the actual prefix density: full-arm mask percentages need
            # not equal their realized rates when sliced at this origin.
            requested_missing_pct = (
                int(regime[len("sparse"):]) if regime.startswith("sparse") else None
            )
            realized_missing_pct = 100.0 * (origin - len(tr_idx)) / origin
            regime_audits.append({
                "regime": regime,
                "requested_missing_pct": requested_missing_pct,
                "observed_training_points": int(len(tr_idx)),
                "training_days": int(origin),
                "realized_missing_pct": realized_missing_pct,
                "fit_eligible": bool(len(tr_idx) >= 10),
            })
            if len(tr_idx) < 10:
                # Keep the per-origin mask audit even when too few points remain
                # for a meaningful model fit; metric consumers can ignore NaNs.
                rows.append(dict(
                    country=country, origin=origin, regime=regime,
                    method="mask_audit", seed=-1, mase=np.nan, mae=np.nan,
                    rmse=np.nan, n_train_points=len(tr_idx),
                    requested_missing_pct=requested_missing_pct,
                    realized_missing_pct=realized_missing_pct, seconds=0.0))
                continue
            t_data = torch.tensor(tr_idx / origin, dtype=torch.float32).reshape(-1, 1)
            I_obs = torch.tensor(I_full[tr_idx] / scale, dtype=torch.float32).reshape(-1, 1)
            # Future normalized times are measured on the training-origin scale,
            # matching the model's tf and preserving prior evaluation semantics.
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
                    prediction = np.asarray(pred).reshape(-1)
                    if prediction.size != horizon or not np.isfinite(prediction).all():
                        raise ValueError(
                            f"{country} origin={origin} regime={regime} "
                            f"mode={mode} seed={seed}: expected {horizon} finite "
                            f"predictions, got {prediction.size}"
                        )
                    rows.append(dict(
                        country=country, origin=origin, regime=regime,
                        method=f"pinn_{mode}", seed=seed,
                        mase=mase(y_true, prediction, y_train),
                        mae=mae(y_true, prediction),
                        rmse=rmse(y_true, prediction), n_train_points=len(tr_idx),
                        requested_missing_pct=requested_missing_pct,
                        realized_missing_pct=realized_missing_pct,
                        seconds=0.0))
                    print(f"  origin={origin:3d} {regime:9s} {mode:5s} seed={seed} "
                          f"MASE={rows[-1]['mase']:8.3f}", flush=True)

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

    # Keep output construction centralized so every row shares the same schema.
    df = pd.DataFrame(rows)

    # Verify row counts against the planned baselines, audit-only regimes, and
    # model fits so skipped or duplicated experiment combinations are visible.
    expected_metric_rows = sum(
        3 + sum(
            len(modes) * len(seeds) if item["fit_eligible"] else 1
            for item in origin_audit["regimes"]
        )
        for origin_audit in origin_audits
    )
    if len(df) != expected_metric_rows:
        raise RuntimeError(
            f"result row audit failed: expected {expected_metric_rows}, got {len(df)}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # Save a separate reproducibility record showing the validated data,
    # train/test windows, and realized mask density actually used at each origin.
    audit = {
        "audit_status": "passed",
        "country": country,
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
        "expected_metrics_rows": int(expected_metric_rows),
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
    print("\n=== RUN AUDIT ===")
    print(f"arm={country}  master_days={len(master)}  "
          f"date_range={master.index.min().date()}..{master.index.max().date()}  "
          f"horizon={horizon}")
    for origin_audit in origin_audits:
        print(f"origin={origin_audit['origin']}  "
              f"train={origin_audit['training_start']}..{origin_audit['training_end']}  "
              f"forecast={origin_audit['forecast_start']}..{origin_audit['forecast_end']}")
        for regime_audit in origin_audit["regimes"]:
            print(f"  {regime_audit['regime']}: "
                  f"observed={regime_audit['observed_training_points']}/"
                  f"{regime_audit['training_days']} "
                  f"missing={regime_audit['realized_missing_pct']:.1f}% "
                  f"fit_eligible={regime_audit['fit_eligible']}")
    print(f"result_rows={len(df)} (expected={expected_metric_rows})")
    print(f"audit_status={audit['audit_status']}  audit_json={audit_path}")
    return df


def main():
    """Parse the selected arm and run only valid train/test windows."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", required=True,
                    choices=["italy_primary", "italy_secondary"],
                    help="Italy arm label written to the result rows")
    ap.add_argument("--master", type=Path, default=Path("data/processed/italy_national_master.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/tables/italy_grid.csv"))
    # These origins fit the 91-day primary window with the default 15-day
    # forecast; users must still satisfy the runtime length validation.
    ap.add_argument("--origins", type=int, nargs="+", default=[45, 60, 75])
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

    # Loading errors and invalid windows should stop the run rather than yield
    # partial grids that look like complete experiment results.
    master = pd.read_csv(a.master, index_col=0, parse_dates=True)
    try:
        validate_run_inputs(master, a.origins, a.horizon, a.regimes, a.modes, a.seeds)
    except ValueError as exc:
        ap.error(str(exc))

    # Model settings are shared across all origins and regimes in this run.
    cfg = Config(epochs_joint=a.epochs_joint, epochs_split_data=a.epochs_split_data,
                 epochs_split_ode=a.epochs_split_ode, n_collocation=a.collocation)

    df = run(master, cfg, a.country, a.origins, a.horizon,
             a.regimes, a.modes, a.seeds, a.out, a.master)
    print("\n=== MEDIAN MASE BY METHOD x REGIME (ITALY) ===")
    print(df.pivot_table(index="method", columns="regime", values="mase",
                         aggfunc="median").round(3).to_string())


if __name__ == "__main__":
    main()
