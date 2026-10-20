#!/usr/bin/env python3
"""
Adds a two-arm Italy stage wrapper on top of run_stages.py's resume-safe
staging pattern, without modifying run_stages.py itself.

Calls prepare_italy.py twice (primary, secondary) then run_italy.py twice
(once per arm), adding each arm's sparse masks before fitting, and tagging
results so they never get merged by accident.

The prepare stage is expected to leave both master CSVs with binary,
deterministic observed_sparse20/30/40 columns. It prints a coverage check and
fails if those columns are missing, malformed, non-nested, or have incorrect
full-arm dropout counts.

USAGE
    python3 run_stages_italy_arms.py --stage prepare
    python3 run_stages_italy_arms.py --stage split
    python3 run_stages_italy_arms.py --stage joint
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).parent
RAW = Path("data/raw/italy_case_hosp.csv")
PROC = Path("data/processed")
RESULTS = Path("results/tables")
SPARSE_PCTS = (20, 30, 40)


def run(cmd: list[str]) -> None:
    """Run a child stage and stop the pipeline if it exits unsuccessfully."""
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def audit_sparse_master(master_path: Path, country: str) -> None:
    """Verify mask columns on disk before this arm is used by the model runner."""
    if not master_path.exists():
        raise FileNotFoundError(f"{country} master not found: {master_path}")
    master = pd.read_csv(master_path, index_col=0, parse_dates=True)
    if master.empty:
        raise ValueError(f"{country} master is empty: {master_path}")

    masks = {}
    print(f"\nMask audit: {country} ({master_path})")
    for pct in SPARSE_PCTS:
        column = f"observed_sparse{pct}"
        if column not in master:
            raise ValueError(f"{master_path} is missing required column {column}")
        values = master[column]
        if values.isna().any() or not values.isin([0, 1]).all():
            raise ValueError(f"{master_path}:{column} must contain only binary 0/1 values")

        mask = values.to_numpy(dtype="int8")
        expected_dropped = round(len(master) * pct / 100)
        dropped = int((mask == 0).sum())
        if dropped != expected_dropped:
            raise ValueError(
                f"{master_path}:{column} dropped {dropped}/{len(master)} days; "
                f"expected {expected_dropped}"
            )
        masks[pct] = mask
        print(f"  {column}: dropped={dropped}/{len(mask)} "
              f"({100 * dropped / len(mask):.1f}%), observed={int(mask.sum())}")

    # A larger requested missing percentage must only remove observations;
    # this confirms the configured masks are nested across the arm.
    for lower_pct, higher_pct in zip(SPARSE_PCTS, SPARSE_PCTS[1:]):
        if (masks[higher_pct] > masks[lower_pct]).any():
            raise ValueError(
                f"{master_path}: sparse{higher_pct} is not nested within sparse{lower_pct}"
            )
    print(f"  PASS: all masks are present, binary, count-correct, and nested")


def stage_prepare(overwrite: bool = False):
    """Prepare both arm masters, add masks, and audit the resulting files."""
    # Build the two distinct target series first; sparsification must operate
    # on these final masters so run_italy.py reads masks alongside I_proxy.
    overwrite_args = ["--overwrite"] if overwrite else []
    run([sys.executable, str(SRC / "prepare_italy.py"), "--arm", "primary",
         "--input", str(RAW), "--outdir", str(PROC), *overwrite_args])
    run([sys.executable, str(SRC / "prepare_italy.py"), "--arm", "secondary",
         "--ascertainment", "restrict", "--input", str(RAW), "--outdir", str(PROC),
         *overwrite_args])

    # Generate nested 20/30/40% masks independently per arm. --force permits
    # a deliberate prepare-stage rerun to refresh masks on regenerated masters.
    for country in ("italy_primary", "italy_secondary"):
        master_path = PROC / f"{country}_national_master.csv"
        run([sys.executable, str(SRC / "sparsify.py"),
             "--master", str(master_path),
             "--country", country,
             "--pct", *(str(pct) for pct in SPARSE_PCTS),
             "--inplace", "--force"])

    # Audit saved files, not in-memory mask arrays, so downstream CLI inputs are
    # confirmed to contain the intended columns and full-arm dropout rates.
    for country in ("italy_primary", "italy_secondary"):
        audit_sparse_master(PROC / f"{country}_national_master.csv", country)


def stage_grid(mode: str):
    """Run a split or joint grid after confirming each input has its masks."""
    RESULTS.mkdir(parents=True, exist_ok=True)

    # A user may invoke split/joint without prepare in this process; fail early
    # with a clear audit instead of launching models on stale or incomplete files.
    for country in ("italy_primary", "italy_secondary"):
        audit_sparse_master(PROC / f"{country}_national_master.csv", country)

    # PRIMARY arm: single 91-day window -> origins must fit inside it
    run([sys.executable, str(SRC / "run_italy.py"),
         "--country", "italy_primary",
         "--master", str(PROC / "italy_primary_national_master.csv"),
         "--out", str(RESULTS / f"italy_primary_grid_{mode}.csv"),
         "--origins", "45", "60", "75",
         "--horizon", "15",
         "--regimes", "daily", "weekly", "sparse20", "sparse30", "sparse40",
         "--modes", mode, "--seeds", "1", "2", "3"])
    # SECONDARY arm: full multi-wave range -> wider origin spread across waves
    run([sys.executable, str(SRC / "run_italy.py"),
         "--country", "italy_secondary",
         "--master", str(PROC / "italy_secondary_national_master.csv"),
         "--out", str(RESULTS / f"italy_secondary_grid_{mode}.csv"),
         "--origins", "100", "300", "500", "700", "900",
         "--horizon", "15",
         "--regimes", "daily", "weekly", "sparse20", "sparse30", "sparse40",
         "--modes", mode, "--seeds", "1", "2", "3"])


def main():
    """Dispatch the requested pipeline stage."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prepare", "split", "joint"], required=True)
    ap.add_argument("--overwrite", action="store_true",
                    help="For prepare only: replace existing arm outputs; back them up first")
    a = ap.parse_args()
    if a.stage == "prepare":
        stage_prepare(overwrite=a.overwrite)
    else:
        if a.overwrite:
            ap.error("--overwrite is only valid with --stage prepare")
        stage_grid(a.stage)


if __name__ == "__main__":
    main()
