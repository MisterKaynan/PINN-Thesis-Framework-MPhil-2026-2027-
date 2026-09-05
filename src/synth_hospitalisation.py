#!/usr/bin/env python3
"""
Synthetic hospitalisation channel for Ghana, derived from the SEIQHRS ODE
system in Dwomoh et al. (2021), Infectious Disease Modelling 6:381-397.

PURPOSE
    Millevoi et al.'s richest real-data cases (6-7) use TWO observed channels
    for Italy: infections and hospitalisations. HERA's Ghana series has no
    hospitalisation field, so this script produces a MODEL-DERIVED proxy that
    lets us test (as an ablation, not a primary claim) whether a second
    observed channel changes PINN behaviour for Ghana the way it does for
    Italy.

WARNING -- READ BEFORE USING DOWNSTREAM
    This is NOT observed data. Dwomoh et al.'s rate parameters (sigma, psi,
    tau, alpha) are explicitly flagged in their own Table 1 as "assumed" or
    "extreme case scenario" values, not fitted to real Ghanaian hospital
    admission records. Any result built on H_synthetic_seiqhrs must be
    reported as a sensitivity/ablation finding, and the column must never be
    relabelled as if it were real surveillance data.

METHOD
    dH/dt = sigma*Q + psi*I - (tau + xi + alpha) H          (Dwomoh Eq 1.5a)

    Q (quarantine compartment) has no real Ghanaian series either. We
    approximate Q_t as a lagged fraction of observed I_t:
        Q_t = q_frac * I_{t-lag}
    with q_frac and lag treated as declared assumptions (defaults below),
    swept in a sensitivity table alongside the Dwomoh parameter scenarios.

Usage:
    python3 synth_hospitalisation.py --master ghana_national_master.csv \
        --params dwomoh_scenario5 --out ghana_hospitalisation_synth.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Dwomoh et al. (2021) Table 1 -- rate parameters by scenario (day^-1 unless noted)
DWOMOH_SCENARIOS = {
    "dwomoh_scenario1": dict(sigma=0.0, psi=0.0, tau=0.0, alpha=1/50, xi=1.978e-5),
    "dwomoh_scenario3": dict(sigma=1/10, psi=1/120, tau=0.0, alpha=1/50, xi=1.978e-5),
    "dwomoh_scenario5": dict(sigma=1/10, psi=1/120, tau=1/120, alpha=1/50, xi=1.978e-5),
}
Q_FRAC_DEFAULT = 0.15   # declared assumption: fraction of I routed through quarantine
Q_LAG_DEFAULT = 3       # declared assumption: days between infection and quarantine entry


def simulate_H(I: np.ndarray, q_frac: float, q_lag: int, params: dict) -> np.ndarray:
    Q = np.zeros_like(I)
    if q_lag < len(I):
        Q[q_lag:] = q_frac * I[:-q_lag]
    H = np.zeros_like(I)
    out_rate = params["tau"] + params["xi"] + params["alpha"]
    for t in range(1, len(I)):
        dH = params["sigma"] * Q[t - 1] + params["psi"] * I[t - 1] - out_rate * H[t - 1]
        H[t] = max(H[t - 1] + dH, 0.0)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True, type=Path)
    ap.add_argument("--params", default="dwomoh_scenario5", choices=list(DWOMOH_SCENARIOS))
    ap.add_argument("--q-frac", type=float, default=Q_FRAC_DEFAULT)
    ap.add_argument("--q-lag", type=int, default=Q_LAG_DEFAULT)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sensitivity", action="store_true",
                    help="Also write H under all three Dwomoh scenarios for a range check.")
    a = ap.parse_args()

    master = pd.read_csv(a.master, index_col=0, parse_dates=True)
    I = master["I_proxy"].to_numpy(float)

    params = DWOMOH_SCENARIOS[a.params]
    H = simulate_H(I, a.q_frac, a.q_lag, params)

    out = pd.DataFrame({"H_synthetic_seiqhrs": H}, index=master.index)
    out.index.name = "date"

    if a.sensitivity:
        for name, p in DWOMOH_SCENARIOS.items():
            out[f"H_synthetic_{name}"] = simulate_H(I, a.q_frac, a.q_lag, p)
        rng = out[[c for c in out.columns if c.startswith("H_synthetic_dwomoh")]]
        print("Sensitivity range across Dwomoh scenarios (max H):")
        print(rng.max().to_string())

    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out)

    meta = dict(params_used=a.params, q_frac=a.q_frac, q_lag=a.q_lag,
                max_H=float(H.max()), mean_H=float(H.mean()),
                warning="MODEL-DERIVED, NOT OBSERVED DATA. Ablation use only.")
    (a.out.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
