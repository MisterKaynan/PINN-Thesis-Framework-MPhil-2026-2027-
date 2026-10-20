#!/usr/bin/env python3
"""Run matched Ghana infection-only versus synthetic-admission PINN fits.

This is an explicitly exploratory ablation: the auxiliary daily admissions
are model-derived, not observed hospital data. Both model variants use the
same infection targets, forecast origins, regimes, modes, and random seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from pinn import Config, ReducedSIRHospitalisationPINN, ReducedSIRPINN
from run_ghana import mae, mase, regime_indices, rmse
from synth_hospitalisation import DWOMOH_SCENARIOS, SCHEMA_VERSION


def _sha256(path: Path) -> str:
    """Return the content hash used to bind synthetic channels to their master."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_inputs(
    master: pd.DataFrame,
    channels: pd.DataFrame,
    metadata: dict[str, Any],
    origins: list[int],
    horizon: int,
    regimes: list[str],
    modes: list[str],
    seeds: list[int],
    scenarios: list[str],
    min_training_points: int,
) -> dict[str, Any]:
    """Audit protocol, source identity, channel coverage, and fit settings."""
    if not isinstance(master.index, pd.DatetimeIndex) or master.empty:
        raise ValueError("Ghana master must have a non-empty DatetimeIndex")
    if master.index.has_duplicates or not master.index.is_monotonic_increasing:
        raise ValueError("Ghana master dates must be unique and increasing")
    if not master.index.equals(
        pd.date_range(master.index[0], master.index[-1], freq="D")
    ):
        raise ValueError("Ghana master dates must be a complete consecutive daily grid")
    if not master.index.equals(channels.index):
        raise ValueError("hospitalisation channel dates must exactly match the master")
    if "I_proxy" not in master:
        raise ValueError("Ghana master is missing I_proxy")
    infection = master["I_proxy"].to_numpy(dtype=float)
    if not np.isfinite(infection).all() or (infection < 0).any():
        raise ValueError("I_proxy must contain finite, nonnegative values")

    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported synthesis audit schema: {metadata.get('schema_version')!r}"
        )
    if metadata.get("audit_status") != "passed":
        raise ValueError("synthesis metadata does not report audit_status='passed'")
    if metadata.get("synthetic_channel") is not True:
        raise ValueError("hospitalisation channel must be explicitly marked synthetic")
    if metadata.get("observed_hospitalisation_data") is not False:
        raise ValueError("hospitalisation channel must not be represented as observed")
    if metadata.get("output_rows") != len(channels):
        raise ValueError("synthesis metadata row count does not match the channel CSV")
    if metadata.get("training_target_column") != "dH_in_synthetic_<scenario>":
        raise ValueError("synthesis metadata does not identify daily admissions as target")
    expected_input = metadata.get("input_definition", {})
    if (
        expected_input.get("daily_incidence_column") != "daily_incidence"
        or expected_input.get("existing_I_proxy_column") != "I_proxy"
        or expected_input.get("existing_I_proxy_definition")
        != "trailing 5-day sum of daily_incidence"
        or expected_input.get("infectious_prevalence_proxy")
        != "trailing 14-day sum of daily_incidence"
    ):
        raise ValueError("synthesis metadata has unexpected incidence/I definitions")
    expected_equations = metadata.get("equations", {})
    if (
        expected_equations.get("quarantine")
        != "dQ/dt = delta_q*I - (xi+n+sigma)*Q"
        or expected_equations.get("daily_admission_inflow")
        != "H_admissions = sigma*Q + psi*I"
        or "no exposed E series" not in expected_equations.get("omitted_term", "")
    ):
        raise ValueError("synthesis metadata does not match the audited Q/admission equations")
    required_checks = (
        "daily_dates_consecutive",
        "daily_incidence_finite_nonnegative",
        "I_proxy_matches_trailing_5_day_incidence",
        "infectious_proxy_is_trailing_14_day_incidence",
        "scenario1_excluded",
        "Q_equation_omits_phi_E_due_to_no_E_series",
        "training_channel_is_daily_admission_inflow",
        "occupancy_retained_as_diagnostic_only",
        "required_scenario_columns_present",
        "admissions_aliases_match",
        "compatibility_alias_matches_default_training_target",
        "all_scenarios_finite_nonnegative",
    )
    checks = metadata.get("audit_checks", {})
    if not all(checks.get(name) is True for name in required_checks):
        raise ValueError("one or more required synthesis audit checks did not pass")
    scenario_metadata = metadata.get("scenarios", {})
    for scenario in scenarios:
        if scenario not in DWOMOH_SCENARIOS:
            raise ValueError(f"unknown synthetic scenario {scenario!r}")
        if scenario not in scenario_metadata:
            raise ValueError(f"synthesis audit is missing scenario {scenario!r}")
        expected_parameters = {
            key: value for key, value in DWOMOH_SCENARIOS[scenario].items()
            if key != "source_scenarios"
        }
        recorded_parameters = scenario_metadata[scenario].get("parameters", {})
        if any(
            key not in recorded_parameters
            or not np.isclose(float(recorded_parameters[key]), float(value))
            for key, value in expected_parameters.items()
        ):
            raise ValueError(f"synthesis metadata has unexpected {scenario} parameters")
        column = f"dH_in_synthetic_{scenario}"
        if column not in channels:
            raise ValueError(f"hospitalisation data are missing {column!r}")
        values = channels[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{column} must contain finite, nonnegative values")
        if not (values > 0).any():
            raise ValueError(f"{column} has no positive admissions values")

    if horizon <= 0 or min_training_points < 1:
        raise ValueError("horizon and minimum training points must be positive")
    if not origins or len(set(origins)) != len(origins):
        raise ValueError("origins must be non-empty and unique")
    if any(origin <= 0 or origin + horizon > len(master) for origin in origins):
        raise ValueError("each origin must leave a complete forecast horizon in the master")
    if not regimes or set(regimes).difference({"daily", "weekly", "mask"}):
        raise ValueError("regimes must be selected from daily, weekly, and mask")
    if "mask" in regimes and "observed_simple" not in master:
        raise ValueError("mask regime requires observed_simple in the Ghana master")
    if not modes or set(modes).difference({"split", "joint"}):
        raise ValueError("modes must be selected from split and joint")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    return {
        "horizon": int(horizon),
        "origins": [int(origin) for origin in origins],
        "regimes": list(regimes),
        "modes": list(modes),
        "seeds": [int(seed) for seed in seeds],
        "scenarios": list(scenarios),
        "min_training_points": int(min_training_points),
    }


def _metric_row(
    y_true: np.ndarray,
    prediction: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, float]:
    """Calculate the same forecast metrics for either model variant."""
    if prediction.size != y_true.size or not np.isfinite(prediction).all():
        raise ValueError(
            f"expected {y_true.size} finite predictions, got {prediction.size}"
        )
    metrics = {
        "mase": mase(y_true, prediction, y_train),
        "mae": mae(y_true, prediction),
        "rmse": rmse(y_true, prediction),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("forecast metrics are undefined or non-finite")
    return metrics


def run(
    master: pd.DataFrame,
    channels: pd.DataFrame,
    metadata: dict[str, Any],
    cfg: Config,
    origins: list[int],
    horizon: int,
    regimes: list[str],
    modes: list[str],
    seeds: list[int],
    scenarios: list[str],
    min_training_points: int,
    out_path: Path,
) -> pd.DataFrame:
    """Fit paired infection-only and auxiliary-admission models and audit rows."""
    protocol = validate_inputs(
        master, channels, metadata, origins, horizon, regimes, modes, seeds,
        scenarios, min_training_points,
    )
    # Check both source identity and artifact integrity before spending compute
    # or creating a partial result file.
    master_path = Path(str(metadata.get("master_csv", "")))
    channels_path = Path(str(metadata.get("output_csv", "")))
    if not master_path.is_file():
        raise ValueError(f"synthesis source master does not exist: {master_path}")
    if _sha256(master_path) != metadata.get("master_sha256"):
        raise ValueError("synthesis channels were generated from a different master file")
    if not channels_path.is_file():
        raise ValueError(f"synthesis CSV does not exist: {channels_path}")
    if _sha256(channels_path) != metadata.get("output_sha256"):
        raise ValueError("synthesis CSV content does not match its audit metadata")

    infection = master["I_proxy"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    origin_audits: list[dict[str, Any]] = []
    baseline_cache: dict[tuple[int, str, str, int], dict[str, Any]] = {}
    fit_count = 0

    # Every comparison shares a country, forecast window, observation regime,
    # optimization mode, and seed; only the auxiliary H loss differs.
    for origin in origins:
        y_train = infection[:origin]
        y_true = infection[origin:origin + horizon]
        regime_audits = []
        for regime in regimes:
            indices = regime_indices(master, regime, origin)
            scale_i_train = (
                float(np.max(infection[indices])) if len(indices) else 0.0
            )
            enough_points = len(indices) >= min_training_points
            regime_audits.append({
                "regime": regime,
                "training_days": int(origin),
                "training_points": int(len(indices)),
                "minimum_points_met": bool(enough_points),
                "positive_infection_scale": bool(scale_i_train > 0),
            })

            for scenario in scenarios:
                h_column = f"dH_in_synthetic_{scenario}"
                h_full = channels[h_column].to_numpy(dtype=float)
                scale_h_train = (
                    float(np.max(h_full[indices])) if len(indices) else 0.0
                )
                scenario_eligible = (
                    enough_points and scale_i_train > 0 and scale_h_train > 0
                )
                for mode in modes:
                    for seed in seeds:
                        common = {
                            "country": "ghana",
                            "experiment": "synthetic_hospitalisation_ablation",
                            "scenario": scenario,
                            "origin": int(origin),
                            "regime": regime,
                            "mode": mode,
                            "seed": int(seed),
                            "n_train_points": int(len(indices)),
                            "training_days": int(origin),
                            "horizon": int(horizon),
                            "h_target_column": h_column,
                            "h_source_infectious_proxy": "I_infectious_proxy_D14",
                            "h_target_provenance": "synthetic_model_derived_not_observed",
                            "i_target_definition": "I_proxy_trailing_5_day_incidence_sum",
                        }
                        if not scenario_eligible:
                            if not enough_points:
                                reason = (
                                    f"requires at least {min_training_points} training "
                                    f"points; regime provides {len(indices)}"
                                )
                            elif scale_i_train <= 0:
                                reason = "selected training dates contain no positive I values"
                            else:
                                reason = (
                                    "no positive synthetic admissions value occurs "
                                    "at the selected training dates"
                                )
                            for variant in ("infection_only", "auxiliary_synthetic_h"):
                                rows.append({
                                    **common,
                                    "model_variant": variant,
                                    "fit_status": "skipped_ineligible_training_window",
                                    "skip_reason": reason,
                                    "mase": np.nan,
                                    "mae": np.nan,
                                    "rmse": np.nan,
                                    "delta_mase_vs_infection_only": np.nan,
                                    "h_training_scale": np.nan,
                                    "h_forecast_max": np.nan,
                                    "seconds": 0.0,
                                })
                            continue

                        cache_key = (origin, regime, mode, seed)
                        if cache_key not in baseline_cache:
                            # Scaling is estimated only from training observations,
                            # avoiding future-data leakage through normalization.
                            scale_i = scale_i_train
                            t_data = torch.tensor(
                                indices / origin, dtype=torch.float32
                            ).reshape(-1, 1)
                            i_obs = torch.tensor(
                                infection[indices] / scale_i,
                                dtype=torch.float32,
                            ).reshape(-1, 1)
                            t_future = torch.tensor(
                                np.arange(origin, origin + horizon) / origin,
                                dtype=torch.float32,
                            ).reshape(-1, 1)

                            baseline = ReducedSIRPINN(
                                Config(**{**cfg.__dict__, "seed": seed}),
                                tf=origin,
                                scale_I=scale_i,
                            )
                            start = time.time()
                            if mode == "joint":
                                baseline.fit_joint(t_data, i_obs)
                            else:
                                baseline.fit_split(t_data, i_obs)
                            base_prediction, _ = baseline.predict(t_future)
                            elapsed = round(time.time() - start, 2)
                            baseline_cache[cache_key] = {
                                **_metric_row(y_true, base_prediction, y_train),
                                "seconds": elapsed,
                            }
                            fit_count += 1

                        baseline_result = baseline_cache[cache_key]
                        rows.append({
                            **common,
                            "model_variant": "infection_only",
                            "fit_status": "completed",
                            "skip_reason": "",
                            **{key: baseline_result[key]
                               for key in ("mase", "mae", "rmse")},
                            "delta_mase_vs_infection_only": 0.0,
                            "h_training_scale": np.nan,
                            "h_forecast_max": np.nan,
                            "seconds": baseline_result["seconds"],
                        })

                        h_train = h_full[indices]
                        scale_h = scale_h_train
                        t_data = torch.tensor(
                            indices / origin, dtype=torch.float32
                        ).reshape(-1, 1)
                        i_obs = torch.tensor(
                            infection[indices] / scale_i_train,
                            dtype=torch.float32,
                        ).reshape(-1, 1)
                        h_obs = torch.tensor(
                            h_train / scale_h, dtype=torch.float32
                        ).reshape(-1, 1)
                        t_future = torch.tensor(
                            np.arange(origin, origin + horizon) / origin,
                            dtype=torch.float32,
                        ).reshape(-1, 1)
                        ablation = ReducedSIRHospitalisationPINN(
                            Config(**{**cfg.__dict__, "seed": seed}),
                            tf=origin,
                            scale_I=scale_i_train,
                            scale_H=scale_h,
                        )
                        start = time.time()
                        if mode == "joint":
                            ablation.fit_joint(t_data, i_obs, h_obs)
                        else:
                            ablation.fit_split(t_data, i_obs, h_obs)
                        ablation_prediction, _, h_prediction = ablation.predict(t_future)
                        h_prediction = np.asarray(h_prediction).reshape(-1)
                        if (
                            h_prediction.size != horizon
                            or not np.isfinite(h_prediction).all()
                            or (h_prediction < 0).any()
                        ):
                            raise ValueError(
                                f"origin={origin} regime={regime} "
                                f"scenario={scenario} mode={mode} seed={seed}: "
                                "auxiliary H forecast must be finite, nonnegative, "
                                f"and have length {horizon}"
                            )
                        ablation_result = _metric_row(y_true, ablation_prediction, y_train)
                        ablation_result["seconds"] = round(time.time() - start, 2)
                        rows.append({
                            **common,
                            "model_variant": "auxiliary_synthetic_h",
                            "fit_status": "completed",
                            "skip_reason": "",
                            **ablation_result,
                            "delta_mase_vs_infection_only": (
                                ablation_result["mase"] - baseline_result["mase"]
                            ),
                            "h_training_scale": scale_h,
                            "h_forecast_max": float(h_prediction.max()),
                        })
                        fit_count += 1
                        print(
                            f"origin={origin:3d} {regime:6s} {scenario:14s} "
                            f"{mode:5s} seed={seed} "
                            f"MASE(I-only/H)={baseline_result['mase']:.3f}/"
                            f"{ablation_result['mase']:.3f}",
                            flush=True,
                        )

        origin_audits.append({
            "origin": int(origin),
            "training_start": str(master.index[0].date()),
            "training_end": str(master.index[origin - 1].date()),
            "forecast_start": str(master.index[origin].date()),
            "forecast_end": str(master.index[origin + horizon - 1].date()),
            "regimes": regime_audits,
        })

    result = pd.DataFrame(rows)
    expected_rows = (
        len(scenarios) * len(origins) * len(regimes) * len(modes)
        * len(seeds) * 2
    )
    if len(result) != expected_rows:
        raise RuntimeError(
            f"output row audit failed: expected {expected_rows}, got {len(result)}"
        )
    successful_h = result[
        (result["model_variant"] == "auxiliary_synthetic_h")
        & (result["fit_status"] == "completed")
    ]
    paired_status = result.pivot(
        index=["scenario", "origin", "regime", "mode", "seed"],
        columns="model_variant",
        values="fit_status",
    )
    if not np.array_equal(
        paired_status["infection_only"].to_numpy(),
        paired_status["auxiliary_synthetic_h"].to_numpy(),
    ):
        raise RuntimeError("infection-only and auxiliary fit statuses are not paired")
    if successful_h.empty:
        raise RuntimeError("no auxiliary hospitalisation ablation fits completed")
    if successful_h["delta_mase_vs_infection_only"].isna().any():
        raise RuntimeError("a completed auxiliary fit lacks a paired MASE difference")
    if (
        not np.isfinite(successful_h["h_forecast_max"].to_numpy(dtype=float)).all()
        or (successful_h["h_forecast_max"] < 0).any()
    ):
        raise RuntimeError("a completed auxiliary fit has an invalid H forecast audit")
    if not set(result["h_target_provenance"]).issubset(
        {"synthetic_model_derived_not_observed"}
    ):
        raise RuntimeError("synthetic-channel provenance label was lost")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    source_hash = _sha256(master_path)

    audit = {
        "audit_status": "passed",
        "experiment": "ghana_synthetic_hospitalisation_ablation",
        "country": "ghana",
        "scientific_status": (
            "exploratory only; synthetic model-derived admission channel, "
            "not observed Ghana hospitalisation data"
        ),
        "master_csv": str(master_path),
        "master_sha256": source_hash,
        "hospitalisation_csv": metadata.get("output_csv"),
        "hospitalisation_audit_status": metadata.get("audit_status"),
        "input_definition": metadata.get("input_definition"),
        "infection_target_definition": (
            "I_proxy, trailing 5-day sum of daily_incidence"
        ),
        "hospitalisation_source_I_definition": (
            "I_infectious_proxy_D14, trailing 14-day sum of daily_incidence"
        ),
        "distinct_I_approximations_disclosed": True,
        "equations": metadata.get("equations"),
        "scenario_definitions": {
            scenario: metadata["scenarios"][scenario] for scenario in scenarios
        },
        "protocol": protocol,
        "origins": origin_audits,
        "model_variants": ["infection_only", "auxiliary_synthetic_h"],
        "h_channel_training_loss": (
            "normalized supervised loss at the same selected dates as I; "
            "no H physics residual"
        ),
        "normalization": (
            "I and H scales estimated from training observations only, "
            "separately at each origin/regime"
        ),
        "result_rows": int(len(result)),
        "expected_result_rows": int(expected_rows),
        "completed_infection_only_rows": int(
            ((result["model_variant"] == "infection_only")
             & (result["fit_status"] == "completed")).sum()
        ),
        "completed_auxiliary_rows": int(len(successful_h)),
        "skipped_rows": int((result["fit_status"] != "completed").sum()),
        "actual_model_fits": int(fit_count),
        "metrics_csv": str(out_path),
        "audit_checks": {
            "input_dates_exactly_aligned": True,
            "synthesis_metadata_passed": True,
            "source_master_hash_matches_synthesis": True,
            "synthesis_output_hash_matches_metadata": True,
            "all_requested_variants_accounted_for": len(result) == expected_rows,
            "auxiliary_results_have_matched_infection_only_metrics": True,
            "auxiliary_h_forecasts_finite_nonnegative_and_horizon_aligned": True,
            "synthetic_provenance_label_present": True,
            "h_target_is_daily_admission_inflow": True,
            "distinct_I_approximations_disclosed": True,
            "occupancy_not_used_as_training_target": True,
        },
    }
    audit_path = out_path.with_suffix(".audit.json")
    temporary = audit_path.with_suffix(audit_path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(audit, indent=2, allow_nan=False),
                             encoding="utf-8")
        os.replace(temporary, audit_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(
        f"ablation audit passed: rows={len(result)} fits={fit_count} "
        f"audit={audit_path}"
    )
    return result


def main() -> None:
    """CLI entry point for synthesis-validated paired ablation experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--hospitalisation", type=Path, required=True,
                        help="CSV produced by synth_hospitalisation.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--origins", type=int, nargs="+", default=[45, 60, 75])
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--regimes", nargs="+", default=["daily", "weekly", "mask"])
    parser.add_argument("--modes", nargs="+", default=["split", "joint"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[34])
    parser.add_argument("--scenarios", nargs="+", choices=list(DWOMOH_SCENARIOS),
                        default=list(DWOMOH_SCENARIOS))
    parser.add_argument("--min-training-points", type=int, default=10)
    parser.add_argument("--epochs-joint", type=int, default=5000)
    parser.add_argument("--epochs-split-data", type=int, default=3000)
    parser.add_argument("--epochs-split-ode", type=int, default=1000)
    parser.add_argument("--collocation", type=int, default=6000)
    args = parser.parse_args()

    master = pd.read_csv(args.master, index_col=0, parse_dates=True)
    channels = pd.read_csv(args.hospitalisation, index_col=0, parse_dates=True)
    metadata_path = args.hospitalisation.with_suffix(".meta.json")
    if not metadata_path.is_file():
        parser.error(f"missing synthesis audit metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    # Ensure the metadata refers to the exact master supplied to this runner,
    # rather than relying only on the row dates or similar-looking filenames.
    if metadata.get("master_sha256") != _sha256(args.master):
        parser.error("hospitalisation artifact was generated from a different master")
    if metadata.get("output_sha256") != _sha256(args.hospitalisation):
        parser.error("hospitalisation CSV does not match its synthesis audit metadata")
    metadata["master_csv"] = str(args.master)
    metadata["output_csv"] = str(args.hospitalisation)

    cfg = Config(
        epochs_joint=args.epochs_joint,
        epochs_split_data=args.epochs_split_data,
        epochs_split_ode=args.epochs_split_ode,
        n_collocation=args.collocation,
    )
    try:
        run(
            master, channels, metadata, cfg, args.origins, args.horizon,
            args.regimes, args.modes, args.seeds, args.scenarios,
            args.min_training_points, args.out,
        )
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
