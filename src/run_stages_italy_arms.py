#!/usr/bin/env python3
"""
Adds a two-arm Italy stage wrapper on top of run_stages.py's resume-safe
staging pattern, without modifying run_stages.py itself.

Calls prepare_italy.py twice (primary, secondary) then run_italy.py twice
(once per arm), tagging results so they never get merged by accident.

USAGE
    python3 run_stages_italy_arms.py --stage prepare
    python3 run_stages_italy_arms.py --stage split
    python3 run_stages_italy_arms.py --stage joint
"""
import argparse
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent
RAW = Path("data/raw/italy_case_hosp.csv")
PROC = Path("data/processed")
RESULTS = Path("results/tables")


def run(cmd: list[str]) -> None:
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def stage_prepare():
    run([sys.executable, str(SRC / "prepare_italy.py"), "--arm", "primary",
         "--input", str(RAW), "--outdir", str(PROC)])
    run([sys.executable, str(SRC / "prepare_italy.py"), "--arm", "secondary",
         "--ascertainment", "restrict", "--input", str(RAW), "--outdir", str(PROC)])


def stage_grid(mode: str):
    RESULTS.mkdir(parents=True, exist_ok=True)
    # PRIMARY arm: single 91-day window -> origins must fit inside it
    run([sys.executable, str(SRC / "run_italy.py"),
         "--master", str(PROC / "italy_primary_national_master.csv"),
         "--out", str(RESULTS / f"italy_primary_grid_{mode}.csv"),
         "--origins", "45", "60", "75",
         "--horizon", "15",
         "--regimes", "daily", "weekly", "sparse20", "sparse30", "sparse40",
         "--modes", mode, "--seeds", "1", "2", "3"])
    # SECONDARY arm: full multi-wave range -> wider origin spread across waves
    run([sys.executable, str(SRC / "run_italy.py"),
         "--master", str(PROC / "italy_secondary_national_master.csv"),
         "--out", str(RESULTS / f"italy_secondary_grid_{mode}.csv"),
         "--origins", "100", "300", "500", "700", "900",
         "--horizon", "15",
         "--regimes", "daily", "weekly", "sparse20", "sparse30", "sparse40",
         "--modes", mode, "--seeds", "1", "2", "3"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prepare", "split", "joint"], required=True)
    a = ap.parse_args()
    if a.stage == "prepare":
        stage_prepare()
    else:
        stage_grid(a.stage)


if __name__ == "__main__":
    main()
