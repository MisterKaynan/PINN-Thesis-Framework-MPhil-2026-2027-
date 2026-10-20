#!/usr/bin/env python3
"""Generate auditable, model-derived Ghana hospitalisation-channel scenarios.

The resulting admission series is synthetic and is intended only for the
auxiliary-channel ablation. It is not Ghana hospital surveillance data and
must not be described as observed admissions.

The reduced construction follows the definitions supplied for Dwomoh et al.:
daily admissions are the inflow 
``sigma * Q + psi * I``; 
Q follows its I-driven equation with no ``phi * E`` term because no exposed series is
available. Daily incidence is converted to an approximate infectious
prevalence using a trailing 14-day sum. Occupancy is retained as a diagnostic,
but is not the PINN training target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = 2
INFECTIOUS_WINDOW_DAYS = 14
DAILY_INCIDENT_COLUMN = "daily_incidence"
LEGACY_INFECTION_PROXY_COLUMN = "I_proxy"

# The attachment groups scenarios 2/3 and 4/5 because the reduced equations
# cannot distinguish the paired scenarios' additional intervention terms.
# Scenario 1 is excluded because its zero hospitalisation parameters produce
# a degenerate channel rather than a useful sensitivity case.
DWOMOH_SCENARIOS: dict[str, dict[str, float | str]] = {
    "dwomoh_s2_s3": {
        "sigma": 1 / 10,
        "psi": 1 / 120,
        "tau": 0.0,
        "alpha": 1 / 50,
        "xi": 1.978e-5,
        "delta_q": 1 / 120,
        "n": 0.0,
        "source_scenarios": "2 and 3",
    },
    "dwomoh_s4_s5": {
        "sigma": 1 / 10,
        "psi": 1 / 120,
        "tau": 1 / 120,
        "alpha": 1 / 50,
        "xi": 1.978e-5,
        "delta_q": 1 / 120,
        "n": 1 / 120,
        "source_scenarios": "4 and 5",
    },
}


def _sha256(path: Path) -> str:
    """Hash the exact source file so the generated channels can be traced."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_master(master: pd.DataFrame) -> None:
    """Reject malformed inputs before any rolling or differential updates."""
    required = {DAILY_INCIDENT_COLUMN, LEGACY_INFECTION_PROXY_COLUMN}
    missing = required.difference(master.columns)
    if missing:
        raise ValueError(f"Ghana master is missing required columns: {sorted(missing)}")
    if not isinstance(master.index, pd.DatetimeIndex):
        raise ValueError("Ghana master must have a parsed DatetimeIndex")
    if master.empty or master.index.has_duplicates:
        raise ValueError("Ghana master must be non-empty and have unique dates")
    if not master.index.is_monotonic_increasing:
        raise ValueError("Ghana master dates must be strictly increasing")
    expected = pd.date_range(master.index[0], master.index[-1], freq="D")
    if not master.index.equals(expected):
        raise ValueError("Ghana master dates must be a complete consecutive daily grid")

    incidence = master[DAILY_INCIDENT_COLUMN].to_numpy(dtype=float)
    proxy = master[LEGACY_INFECTION_PROXY_COLUMN].to_numpy(dtype=float)
    for name, values in ((DAILY_INCIDENT_COLUMN, incidence),
                         (LEGACY_INFECTION_PROXY_COLUMN, proxy)):
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{name} must contain finite, nonnegative values")

    # This confirms what the existing master calls I_proxy: a five-day rolling
    # sum of daily cases. It is not silently re-used as infectious prevalence.
    expected_proxy = (
        master[DAILY_INCIDENT_COLUMN]
        .rolling(5, min_periods=1)
        .sum()
        .to_numpy(dtype=float)
    )
    if not np.allclose(proxy, expected_proxy, rtol=0, atol=1e-8):
        raise ValueError(
            "I_proxy does not match the documented trailing five-day "
            "daily_incidence sum; inspect the master provenance"
        )


def infectious_prevalence_proxy(master: pd.DataFrame) -> np.ndarray:
    """Approximate currently infectious people with a trailing 14-day sum."""
    return (
        master[DAILY_INCIDENT_COLUMN]
        .rolling(INFECTIOUS_WINDOW_DAYS, min_periods=1)
        .sum()
        .to_numpy(dtype=float)
    )


def simulate_scenario(
    infectious: np.ndarray,
    params: dict[str, float | str],
    *,
    dt: float = 1.0,
) -> dict[str, np.ndarray]:
    """Integrate Q and occupancy while returning the daily admission inflow.

    Q is advanced by explicit Euler using the preceding day's infectious
    prevalence. The admissions column is the current-day inflow
    ``sigma*Q + psi*I``; occupancy H separately accumulates inflow and loses
    people through ``tau + xi + alpha``. ``n`` affects Q only, as in the
    supplied equation. No exposed-compartment term is available or inferred.
    """
    if infectious.ndim != 1 or infectious.size == 0:
        raise ValueError("infectious prevalence must be a non-empty 1-D series")
    if not np.isfinite(infectious).all() or (infectious < 0).any():
        raise ValueError("infectious prevalence must be finite and nonnegative")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"dt must be finite and positive, got {dt}")

    sigma = float(params["sigma"])
    psi = float(params["psi"])
    tau = float(params["tau"])
    alpha = float(params["alpha"])
    xi = float(params["xi"])
    delta_q = float(params["delta_q"])
    n = float(params["n"])

    q = np.zeros_like(infectious, dtype=float)
    h_occupancy = np.zeros_like(infectious, dtype=float)
    admissions = np.zeros_like(infectious, dtype=float)

    # The first-day state starts at zero; subsequent Q values follow dQ/dt.
    for day in range(infectious.size):
        admissions[day] = sigma * q[day] + psi * infectious[day]
        if day + 1 < infectious.size:
            dq = delta_q * infectious[day] - (xi + n + sigma) * q[day]
            q[day + 1] = max(q[day] + dt * dq, 0.0)
            dh = admissions[day] - (tau + xi + alpha) * h_occupancy[day]
            h_occupancy[day + 1] = max(
                h_occupancy[day] + dt * dh, 0.0
            )

    arrays = {
        "Q": q,
        "H_occupancy": h_occupancy,
        "H_admissions": admissions,
    }
    for name, values in arrays.items():
        if not np.isfinite(values).all() or (values < 0).any():
            raise RuntimeError(f"scenario produced invalid {name} values")
    return arrays


def build_channels(master: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build both non-degenerate scenarios and an internal method audit."""
    validate_master(master)
    infectious = infectious_prevalence_proxy(master)
    output = pd.DataFrame(
        {"I_infectious_proxy_D14": infectious},
        index=master.index.copy(),
    )
    scenario_audits: dict[str, Any] = {}

    for name, params in DWOMOH_SCENARIOS.items():
        trajectories = simulate_scenario(infectious, params)
        output[f"Q_synthetic_{name}"] = trajectories["Q"]
        output[f"H_occupancy_synthetic_{name}"] = trajectories["H_occupancy"]
        # ``dH_in`` is the paper-aligned training name; the H_admissions
        # alias makes the flow-versus-occupancy distinction obvious in tables.
        output[f"dH_in_synthetic_{name}"] = trajectories["H_admissions"]
        output[f"H_admissions_synthetic_{name}"] = trajectories["H_admissions"]
        scenario_audits[name] = {
            "source_scenarios": params["source_scenarios"],
            "parameters": {key: value for key, value in params.items()
                           if key != "source_scenarios"},
            "admissions_finite": bool(np.isfinite(trajectories["H_admissions"]).all()),
            "admissions_nonnegative": bool((trajectories["H_admissions"] >= 0).all()),
            "admissions_positive_days": int(
                np.count_nonzero(trajectories["H_admissions"] > 0)
            ),
            "admissions_total": float(trajectories["H_admissions"].sum()),
            "admissions_max": float(trajectories["H_admissions"].max()),
            "occupancy_finite": bool(np.isfinite(trajectories["H_occupancy"]).all()),
            "occupancy_nonnegative": bool((trajectories["H_occupancy"] >= 0).all()),
            "occupancy_max": float(trajectories["H_occupancy"].max()),
        }

    # Keep the existing generic name as a compatibility alias, but define it
    # unambiguously as daily admissions from the selected default scenario.
    selected = "dwomoh_s4_s5"
    output["H_synthetic_seiqhrs"] = output[
        f"dH_in_synthetic_{selected}"
    ]
    output.index.name = "date"

    checks = {
        "daily_dates_consecutive": True,
        "daily_incidence_finite_nonnegative": True,
        "I_proxy_matches_trailing_5_day_incidence": True,
        "infectious_proxy_is_trailing_14_day_incidence": True,
        "scenario1_excluded": "dwomoh_scenario1" not in DWOMOH_SCENARIOS,
        "Q_equation_omits_phi_E_due_to_no_E_series": True,
        "training_channel_is_daily_admission_inflow": True,
        "occupancy_retained_as_diagnostic_only": True,
        "required_scenario_columns_present": all(
            f"dH_in_synthetic_{name}" in output
            and f"H_admissions_synthetic_{name}" in output
            and f"H_occupancy_synthetic_{name}" in output
            for name in DWOMOH_SCENARIOS
        ),
        "admissions_aliases_match": all(
            np.array_equal(
                output[f"dH_in_synthetic_{name}"].to_numpy(),
                output[f"H_admissions_synthetic_{name}"].to_numpy(),
            )
            for name in DWOMOH_SCENARIOS
        ),
        "compatibility_alias_matches_default_training_target": np.array_equal(
            output["H_synthetic_seiqhrs"].to_numpy(),
            output["dH_in_synthetic_dwomoh_s4_s5"].to_numpy(),
        ),
        "all_scenarios_finite_nonnegative": all(
            audit["admissions_finite"]
            and audit["admissions_nonnegative"]
            and audit["occupancy_finite"]
            and audit["occupancy_nonnegative"]
            and audit["admissions_positive_days"] > 0
            for audit in scenario_audits.values()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"hospitalisation synthesis audit failed: {checks}")

    # This is only a dimensional comparison under the supplied assumptions.
    # It is not a fitted conversion between the two papers' model parameters.
    direct_rate = float(DWOMOH_SCENARIOS["dwomoh_s2_s3"]["psi"])
    millevoi_delta = 1 / 5
    equivalent_sigma = direct_rate / millevoi_delta
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "audit_status": "passed",
        "country": "ghana",
        "synthetic_channel": True,
        "observed_hospitalisation_data": False,
        "training_target_column": "dH_in_synthetic_<scenario>",
        "training_target_definition": "daily admissions = sigma*Q + psi*I",
        "compatibility_alias": {
            "column": "H_synthetic_seiqhrs",
            "scenario": selected,
            "meaning": "daily synthetic admission inflow, not occupancy",
        },
        "admissions_aliases": (
            "H_admissions_synthetic_<scenario> is numerically identical to "
            "dH_in_synthetic_<scenario>; use dH_in as the PINN training target."
        ),
        "input_definition": {
            "daily_incidence_column": DAILY_INCIDENT_COLUMN,
            "existing_I_proxy_column": LEGACY_INFECTION_PROXY_COLUMN,
            "existing_I_proxy_definition": "trailing 5-day sum of daily_incidence",
            "infectious_prevalence_proxy": (
                f"trailing {INFECTIOUS_WINDOW_DAYS}-day sum of daily_incidence"
            ),
            "assumption": (
                "A 14-day infectious period is used to approximate prevalent I; "
                "this is model-derived from reported incidence, not observed I."
            ),
            "incidence_reporting_gaps": (
                "The Ghana source master encodes missing reports as zero. The "
                "synthetic channel inherits this limitation and does not impute "
                "unreported cases."
            ),
        },
        "equations": {
            "quarantine": "dQ/dt = delta_q*I - (xi+n+sigma)*Q",
            "daily_admission_inflow": "H_admissions = sigma*Q + psi*I",
            "occupancy_diagnostic": (
                "dH/dt = sigma*Q + psi*I - (tau+xi+alpha)*H"
            ),
            "integration": "explicit Euler, dt=1 day, Q(0)=H(0)=0",
            "omitted_term": (
                "phi*E omitted because the source master has no exposed E series"
            ),
            "training_semantics": (
                "The model's auxiliary H output is trained on daily admission "
                "inflow; H occupancy is diagnostic only."
            ),
        },
        "scenarios": scenario_audits,
        "cross_paper_parameter_comparison": {
            "Dwomoh_direct_I_to_H_rate_psi_per_day": direct_rate,
            "assumed_Millevoi_delta_per_day": millevoi_delta,
            "dimensional_equivalent_sigma": equivalent_sigma,
            "reported_Millevoi_sigma_range": [0.03, 0.23],
            "within_reported_range": 0.03 <= equivalent_sigma <= 0.23,
            "interpretation": (
                "Dimensional comparison under assumptions only; not a fitted "
                "equivalence or validation of the synthetic Ghana series."
            ),
        },
        "paper_reference_diagnostics_not_targets": {
            "Dwomoh": {
                "scenario_1_H_max": 0.0,
                "scenario_2_H_day_365_approx": 900.0,
                "scenario_3_H_day_365_approx": 820.0,
                "scenario_4_H_day_365_approx": 420.0,
                "scenario_5_H_peak_approx": 7.0,
                "scenario_5_peak_day_approx": 200,
                "interpretation": (
                    "Paper model outputs only; different from this reduced "
                    "incidence-driven construction and not observed Ghana "
                    "admissions or acceptance thresholds."
                ),
            },
            "Millevoi_fig11": {
                "visual_point_readings": True,
                "approximate_relative_uncertainty": "10-15%",
                "use": (
                    "Do not use the visual readings as training targets; use "
                    "actual ISS data for quantitative estimation."
                ),
            },
        },
        "audit_checks": checks,
    }
    return output, metadata


def main() -> None:
    """CLI: validate the master, generate scenario channels, and persist audit."""
    parser = argparse.ArgumentParser(
        description="Generate audited synthetic Ghana hospital-admission channels."
    )
    parser.add_argument("--master", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    master = pd.read_csv(args.master, index_col=0, parse_dates=True)
    channels, metadata = build_channels(master)
    metadata["master_csv"] = str(args.master)
    metadata["master_sha256"] = _sha256(args.master)
    metadata["output_csv"] = str(args.out)
    metadata["output_rows"] = int(len(channels))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    channels.to_csv(args.out)
    metadata["output_sha256"] = _sha256(args.out)
    audit_path = args.out.with_suffix(".meta.json")
    temporary = audit_path.with_suffix(audit_path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False),
                             encoding="utf-8")
        os.replace(temporary, audit_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps(metadata, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
