#!/usr/bin/env python3
"""
Staged local runner. Cross-platform (Windows, macOS, Linux).

WHY STAGES
    The grid is not uniform: a joint fit costs about 8x a split fit. Running
    everything as one job means nothing is on disk until the very end, so an
    interrupted run leaves you with nothing.

    This runs cheapest-first and writes a CSV after every stage. Stop whenever
    you like; whatever finished is already saved and usable.

    Stage 1 (validation)  ~35 min   confirms the implementation is correct
    Stage 2 (split)       ~50 min   full split results across all regimes
    Stage 3 (joint)       ~6.5 h    the expensive half

    On 4 cores you have publishable split results inside 1.5 hours, and can
    leave stage 3 running overnight.

USAGE
    python src/run_stages.py                    # all stages, 4 workers
    python src/run_stages.py --workers 8
    python src/run_stages.py --skip-validation  # if already validated
    python src/run_stages.py --stage split      # one stage only

    Resume after an interruption: rerun the same command. Completed stages are
    detected from their output files and skipped.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"

ORIGINS = ["100", "140", "180", "220", "260", "300"]
REGIMES = ["daily", "weekly", "mask"]
SEEDS = ["1", "2", "3"]


def sh(cmd: list[str], log: Path) -> bool:
    """Run a command, streaming output to console and a log file."""
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"),
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    with log.open("w") as fh:
        p = subprocess.Popen(cmd, env=env, cwd=ROOT, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line)
            fh.write(line)
        p.wait()
    return p.returncode == 0


def stage_validation(args) -> bool:
    out = TABLES / "validation_case4.json"
    if out.exists() and not args.force:
        print(f"[skip] validation already done -> {out.name}")
        return True
    print("=" * 66)
    print("STAGE 1/3  VALIDATION  (~35 min)")
    print("  Confirms our code reproduces Millevoi et al. Table 4.")
    print("  Targets: split err_I 0.1331, err_Rt 0.4744")
    print("=" * 66)
    ok = sh([sys.executable, "src/pinn.py", "--repo", "reference",
             "--mode", "both", "--out", str(out)],
            ROOT / "results" / "logs" / "validation.log")
    if ok:
        import json
        r = json.load(open(out))["case4"]
        e = r.get("split", {}).get("err_I")
        if e is not None and not (0.10 <= e <= 0.22):
            print(f"\n!! WARNING: split err_I = {e:.4f}, expected ~0.13.")
            print("!! Do not trust downstream results. Check the log.")
    return ok


def stage_grid(name: str, mode: str, args) -> bool:
    out = TABLES / f"grid_{name}.csv"
    if out.exists() and not args.force:
        print(f"[skip] {name} already done -> {out.name}")
        return True
    n = len(ORIGINS) * len(REGIMES) * len(SEEDS)
    per = 29 if mode == "joint" else 3.5
    print("=" * 66)
    print(f"STAGE {'3/3' if mode=='joint' else '2/3'}  {mode.upper()} GRID  "
          f"({n} fits, ~{n*per/args.workers/60:.1f} h on {args.workers} workers)")
    print("=" * 66)
    return sh([sys.executable, "src/run_parallel.py",
               "--workers", str(args.workers),
               "--origins", *ORIGINS, "--regimes", *REGIMES,
               "--modes", mode, "--seeds", *SEEDS,
               "--out", str(out)],
              ROOT / "results" / "logs" / f"grid_{name}.log")


def merge():
    """Combine whatever stage outputs exist into one grid.csv."""
    import pandas as pd
    parts = sorted(TABLES.glob("grid_*.csv"))
    parts = [p for p in parts if p.name != "grid.csv"]
    if not parts:
        return
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    # Baselines are recomputed identically in each stage; keep one copy.
    base = df[~df.method.str.startswith("pinn")].drop_duplicates(
        subset=["origin", "method"])
    df = pd.concat([df[df.method.str.startswith("pinn")], base],
                   ignore_index=True)
    dest = TABLES / "grid.csv"
    df.to_csv(dest, index=False)
    print(f"\nMerged {len(parts)} stage file(s) -> {dest}  ({len(df)} rows)")
    pinn = df[df.method.str.startswith("pinn")]
    if len(pinn):
        print("\nMedian MASE by method x regime:")
        print(pinn.pivot_table(index="method", columns="regime",
                               values="mase", aggfunc="median").round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--skip-validation", action="store_true")
    ap.add_argument("--stage", choices=["validation", "split", "joint", "all"],
                    default="all")
    ap.add_argument("--force", action="store_true",
                    help="Rerun stages even if their output already exists.")
    args = ap.parse_args()

    t0 = time.time()
    print(f"root    : {ROOT}")
    print(f"workers : {args.workers}")

    if args.stage in ("validation", "all") and not args.skip_validation:
        if not stage_validation(args):
            print("\n!! Validation failed. Stopping.")
            return 1

    if args.stage in ("split", "all"):
        stage_grid("split", "split", args)
        merge()
        print("\n>>> Split results are saved. Safe to stop here if needed.")

    if args.stage in ("joint", "all"):
        stage_grid("joint", "joint", args)
        merge()

    print(f"\nTotal elapsed: {(time.time()-t0)/3600:.2f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
