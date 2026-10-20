#!/usr/bin/env python3
"""Resumable, checkpointed PINN grid runner for Ghana and Italy arms.

Run one independent fit at a time and atomically append its result to --out.
If the runtime disconnects, upload/download the CSV and re-run the exact same
command: completed (country, experiment, origin, regime, mode, seed) PINN fits
are skipped. A sidecar identity record prevents resuming with a different arm,
master dataset, or model/experiment configuration.

This is deliberately task-level checkpointing, rather than mid-epoch model
checkpointing. The experiment's scientific output is the grid of completed
fits; its tasks are independent, and a partial neural-network fit is not a
valid result. Restarting only the interrupted fit prevents loss of completed
work without mixing incompatible optimizer states.

Requires: pinn.py, run_ghana.py, and run_italy.py on PYTHONPATH.
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
from run_ghana import (
    classical_sir,
    mae,
    mase,
    moving_average,
    naive_persistence,
    regime_indices as regime_indices_ghana,
    rmse,
)
from run_italy import regime_indices_italy

KEY = ["country", "experiment_id", "origin", "regime", "method", "seed"]


def experiment_identity(args: argparse.Namespace, master_hash: str) -> dict:
    """Create a canonical identity for every setting that affects grid results."""
    return {
        "country": args.country,
        "master_sha256": master_hash,
        "origins": list(args.origins),
        "horizon": args.horizon,
        "regimes": list(args.regimes),
        "modes": list(args.modes),
        "seeds": list(args.seeds),
        "epochs_joint": args.epochs_joint,
        "epochs_split_data": args.epochs_split_data,
        "epochs_split_ode": args.epochs_split_ode,
        "collocation": args.collocation,
        "delta": args.delta,
    }


def identity_id(identity: dict) -> str:
    """Hash normalized identity JSON for a concise, stable row/checkpoint key."""
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_inputs(master: pd.DataFrame, country: str, origins: list[int],
                    horizon: int, regimes: list[str], modes: list[str],
                    seeds: list[int]) -> np.ndarray:
    """Validate selected arm, daily data, windows, and regime-specific columns."""
    if master.empty or "I_proxy" not in master:
        raise ValueError("master must be non-empty and contain I_proxy")
    if not isinstance(master.index, pd.DatetimeIndex):
        raise ValueError("master index must contain parsed dates")
    if not master.index.is_monotonic_increasing or master.index.has_duplicates:
        raise ValueError("master dates must be unique and increasing")
    expected_dates = pd.date_range(master.index[0], periods=len(master), freq="D")
    if not master.index.equals(expected_dates):
        raise ValueError("master must contain a consecutive daily series")

    incidence = master["I_proxy"].to_numpy(dtype=float)
    if not np.isfinite(incidence).all() or (incidence < 0).any() or incidence.max(initial=0) <= 0:
        raise ValueError("I_proxy must be finite, non-negative, and have a positive value")
    if horizon <= 0 or not origins or len(set(origins)) != len(origins):
        raise ValueError("horizon must be positive and origins must be non-empty and unique")
    invalid_origins = [origin for origin in origins if origin <= 0 or origin + horizon > len(master)]
    if invalid_origins:
        raise ValueError(
            f"origin(s) {invalid_origins} do not fit {len(master)} days with horizon {horizon}"
        )
    if not regimes or len(set(regimes)) != len(regimes):
        raise ValueError("regimes must be non-empty and unique")
    if not modes or len(set(modes)) != len(modes) or not set(modes) <= {"split", "joint"}:
        raise ValueError("modes must be unique values from split/joint")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")

    # A country-specific regime selector must recognize every requested regime
    # and validate that its observation data are present and binary.
    selector = select_regime_indices(country)
    for regime in regimes:
        if regime in {"daily", "weekly"}:
            continue
        if country == "ghana" and regime == "mask":
            column = "observed_simple"
        elif country in {"italy_primary", "italy_secondary"} and regime.startswith("sparse"):
            pct_text = regime[len("sparse"):]
            if not pct_text.isdigit() or not 0 < int(pct_text) < 100:
                raise ValueError(f"invalid Italy sparse regime: {regime}")
            column = f"observed_{regime}"
        else:
            raise ValueError(f"unsupported regime {regime!r} for {country}")

        if column not in master:
            raise ValueError(f"master missing {column}, required by regime {regime}")
        values = master[column].to_numpy()
        if pd.isna(values).any() or not np.isin(values, [0, 1]).all():
            raise ValueError(f"{column} must contain only binary 0/1 values")
        # Exercise the selector at each requested prefix before fitting.
        for origin in origins:
            selector(master, regime, origin)
    return incidence


def select_regime_indices(country: str):
    """Return the observation-regime selector for the explicitly selected arm."""
    if country == "ghana":
        return regime_indices_ghana
    if country in {"italy_primary", "italy_secondary"}:
        return regime_indices_italy
    raise ValueError(f"country must be ghana, italy_primary, or italy_secondary; got {country!r}")


def atomic_write(df: pd.DataFrame, path: Path) -> None:
    """Replace the checkpoint CSV atomically after each durable result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def load_completed(path: Path, country: str, experiment: str) -> tuple[pd.DataFrame, set[tuple]]:
    """Load completed PINN keys only when rows match this experiment identity."""
    if not path.exists():
        return pd.DataFrame(), set()
    df = pd.read_csv(path)
    missing = set(KEY) - set(df.columns)
    if missing:
        raise SystemExit(
            f"Existing checkpoint {path} lacks required columns {sorted(missing)}. "
            "Use a new --out filename or restore a compatible checkpoint CSV."
        )
    if not df.empty and (
        not df["country"].eq(country).all()
        or not df["experiment_id"].eq(experiment).all()
    ):
        raise SystemExit(
            f"Checkpoint {path} contains rows from another country or experiment. "
            "Use a new --out path or --overwrite."
        )
    if df.duplicated(KEY).any():
        raise SystemExit(f"Checkpoint {path} contains duplicate result keys.")
    pinn = df[df["method"].astype(str).str.startswith("pinn_")]
    completed = set(map(tuple, pinn[KEY].itertuples(index=False, name=None)))
    return df, completed


def baseline_rows(I_full: np.ndarray, origins: list[int], horizon: int, delta: float,
                  country: str, experiment: str) -> list[dict]:
    """Compute baselines and tag each row with its arm and experiment identity."""
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
                country=country, experiment_id=experiment,
                origin=origin, regime="-", method=name, seed=-1,
                mase=mase(y_true, pred, y_train), mae=mae(y_true, pred),
                rmse=rmse(y_true, pred), n_train_points=len(y_train),
                seconds=0.0,
            ))
    return rows


def add_baselines_once(df: pd.DataFrame, I_full: np.ndarray, origins: list[int],
                       horizon: int, delta: float, out: Path, country: str,
                       experiment: str) -> pd.DataFrame:
    """Add any missing baseline rows without duplicating scoped checkpoint keys."""
    baseline_methods = {"persistence", "moving_avg_7", "classical_sir"}
    existing = (
        set(map(tuple, df[df["method"].isin(baseline_methods)][KEY]
                .itertuples(index=False, name=None)))
        if not df.empty else set()
    )
    new = [
        row for row in baseline_rows(
            I_full, origins, horizon, delta, country, experiment
        )
        if tuple(row[key] for key in KEY) not in existing
    ]
    if new:
        df = pd.concat([df, pd.DataFrame(new)], ignore_index=True)
        atomic_write(df, out)
    return df


def atomic_write_json(data: dict, path: Path) -> None:
    """Write identity and audit sidecars atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> None:
    """Validate a run, resume only its matching checkpoint, and audit progress."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", required=True,
                    choices=["ghana", "italy_primary", "italy_secondary"],
                    help="Country/arm used to select regimes and scope checkpoint rows")
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
    ap.add_argument("--delta", type=float, default=0.2,
                    help="Recovery-rate parameter used by the classical SIR baseline")
    ap.add_argument("--max-fits", type=int, default=None,
                    help="Stop cleanly after this many newly completed PINN fits; useful for planned free-tier sessions.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Discard this output's CSV and sidecars, then start a fresh experiment.")
    args = ap.parse_args()

    # Load and validate all data/protocol options before removing or changing
    # checkpoint files. Country-specific masks use the selected runner's logic.
    master = pd.read_csv(args.master, index_col=0, parse_dates=True)
    try:
        I_full = validate_inputs(
            master, args.country, args.origins, args.horizon,
            args.regimes, args.modes, args.seeds,
        )
    except ValueError as exc:
        ap.error(str(exc))
    scale = float(I_full.max())
    if args.max_fits is not None and args.max_fits <= 0:
        ap.error("--max-fits must be positive")

    # Build a path-independent experiment identity. A same-length/different
    # master or changed model settings therefore cannot reuse old fit rows.
    cfg = Config(
        epochs_joint=args.epochs_joint,
        epochs_split_data=args.epochs_split_data,
        epochs_split_ode=args.epochs_split_ode,
        n_collocation=args.collocation,
        delta=args.delta,
    )
    master_hash = hashlib.sha256(args.master.read_bytes()).hexdigest()
    identity = experiment_identity(args, master_hash)
    identity["model_config"] = {
        key: value for key, value in vars(cfg).items() if key != "seed"
    }
    experiment = identity_id(identity)
    identity_path = args.out.with_suffix(".identity.json")
    audit_path = args.out.with_suffix(".audit.json")

    # Overwrite is the explicit opt-in to discard one output's previous state.
    if args.overwrite:
        for existing_path in (args.out, identity_path, audit_path):
            if existing_path.exists():
                existing_path.unlink()
    elif args.out.exists() and not identity_path.exists():
        raise SystemExit(
            f"Checkpoint {args.out} has no identity sidecar {identity_path}. "
            "Use --overwrite or choose a new --out path."
        )
    elif identity_path.exists():
        recorded_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if recorded_identity != identity:
            raise SystemExit(
                f"Checkpoint identity mismatch in {identity_path}. "
                "Use a new --out path or --overwrite."
            )
    else:
        atomic_write_json(identity, identity_path)

    # The CSV key and sidecar identity are both checked: the sidecar binds the
    # configuration, while each row scopes completion to country + experiment.
    df, completed = load_completed(args.out, args.country, experiment)
    df = add_baselines_once(
        df, I_full, args.origins, args.horizon, cfg.delta, args.out,
        args.country, experiment,
    )

    selector = select_regime_indices(args.country)
    tasks = [
        (args.country, experiment, origin, regime, mode, seed)
        for origin in args.origins
        for regime in args.regimes
        for mode in args.modes
        for seed in args.seeds
        if (args.country, experiment, origin, regime, f"pinn_{mode}", seed) not in completed
    ]
    print(f"Checkpoint: {args.out}")
    print(f"Country: {args.country} | Experiment: {experiment[:12]}")
    print(f"Completed PINN fits: {len(completed)} | Remaining: {len(tasks)}")

    newly_done = 0
    skipped = []
    for country, task_experiment, origin, regime, mode, seed in tasks:
        if args.max_fits is not None and newly_done >= args.max_fits:
            break
        y_train = I_full[:origin]
        y_true = I_full[origin:origin + args.horizon]
        tr_idx = selector(master, regime, origin)
        if len(tr_idx) < 10:
            print(f"SKIP origin={origin} regime={regime}: fewer than 10 training points", flush=True)
            skipped.append({
                "origin": origin, "regime": regime, "mode": mode,
                "seed": seed, "reason": "fewer than 10 training points",
            })
            continue

        # Normalize input coordinates/targets to the same origin/series scales
        # used by the non-checkpointed country runners.
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
        prediction = np.asarray(pred).reshape(-1)
        if prediction.size != args.horizon or not np.isfinite(prediction).all():
            raise ValueError(
                f"{country} origin={origin} regime={regime} mode={mode} seed={seed}: "
                f"expected {args.horizon} finite predictions, got {prediction.size}"
            )

        row = dict(
            country=country, experiment_id=task_experiment,
            origin=origin, regime=regime, method=f"pinn_{mode}", seed=seed,
            mase=mase(y_true, prediction, y_train), mae=mae(y_true, prediction),
            rmse=rmse(y_true, prediction), n_train_points=len(tr_idx),
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

    completed_after = len(load_completed(args.out, args.country, experiment)[1])
    remaining = max(0, len(tasks) - newly_done - len(skipped))
    # Persist a compact run-level audit on every clean exit so a capped session
    # still records its identity, completed count, skipped fits, and remaining work.
    audit = {
        "audit_status": "passed",
        "country": args.country,
        "experiment_id": experiment,
        "identity": identity,
        "planned_pinn_fits": len(args.origins) * len(args.regimes) * len(args.modes) * len(args.seeds),
        "completed_pinn_fits": completed_after,
        "newly_completed_this_session": newly_done,
        "skipped_fits_this_session": skipped,
        "remaining_fits_from_this_invocation": remaining,
        "checkpoint_csv": str(args.out),
        "checkpoint_identity_json": str(identity_path),
    }
    atomic_write_json(audit, audit_path)
    print(f"Finished this session: {newly_done} new fit(s); remaining from this invocation: {remaining}")
    print(f"Checkpoint is current: {args.out}")
    print(f"Run audit: {audit_path} | completed={completed_after}/{audit['planned_pinn_fits']}")


if __name__ == "__main__":
    main()
