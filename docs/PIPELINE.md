# MPhil Thesis Pipeline: Split/Joint Reduced-SIR PINN, Italy Benchmark vs Ghana Focal
## Extending the CSCD618/DSCD604 course project codebase

This document specifies the full experimental pipeline built on top of the
existing course-project code (`pinn.py`, `prepare_ghana.py`, `run_ghana.py`,
`run_stages.py`, `run_parallel.py`, `extract_rt.py`, `fig_dataquality.py`).
Nothing below throws away that code — it is reused as the Ghana arm and as the
validated PINN engine. Three things are added: (1) an Italy benchmark arm using
the same engine, (2) a synthetic SEIQHRS-derived hospitalisation channel for
Ghana, and (3) an equity/statistical-testing layer that the course project did
not need but the thesis's research questions require.

--------------------------------------------------------------------------
## 0. Objectives mapped to pipeline stages

| RQ / Hypothesis | Pipeline stage that answers it |
|---|---|
| R1 data-quality characterisation | Stage 1 (audit) + `fig_dataquality.py` extended to Italy |
| R2 / H1 sparsity robustness, MASE | Stage 3 (grid) across daily/weekly/mask/synthetic-dropout regimes, Italy + Ghana |
| R3 / H2 equity gaps | Stage 4 (equity metrics): cross-regime, cross-country MASE gap, ECE/CRPS |
| R4 / H3 decision utility | Stage 5 (Rt validity + peak-timing) vs Dwomoh et al. and ISS Rt |
| Statistical tests | Stage 6 (significance testing on paired grid results) |

--------------------------------------------------------------------------
## 1. Directory layout (extends the course project's `src/`, `data/`, `results/`)

```
proj/
  data/
    raw/
      ghana_hera.csv                 # existing
      italy_iss_rt_<region>.csv      # existing (regional Rt) + national aggregate needed
      italy_case_hosp.csv            # NEW: ISS national daily infections + hospitalisations
    processed/
      ghana_national_master.csv      # existing (prepare_ghana.py output)
      ghana_weekly.csv               # existing
      ghana_millevoi_format.csv      # existing
      ghana_hospitalisation_synth.csv # NEW (synthetic daily admissions + occupancy diagnostics)
      italy_national_master.csv      # NEW (prepare_italy.py output, mirrors Ghana schema)
      italy_synth_sparse/            # NEW: Italy under imposed 20/30/40% dropout + weekly agg
  src/
    pinn.py                         # existing, UNCHANGED (engine)
    prepare_ghana.py                # existing, UNCHANGED
    prepare_italy.py                # NEW
    synth_hospitalisation.py        # NEW (SEIQHRS ODE -> H proxy)
    sparsify.py                     # NEW (imposes missingness on Italy for the sparsity axis)
    run_ghana.py                    # existing, UNCHANGED (Ghana grid engine)
    run_ghana_hospitalisation_ablation.py # NEW (matched infection-only vs auxiliary-H fits)
    run_italy.py                    # NEW (Italy grid engine, same interface as run_ghana.py)
    run_parallel.py                 # existing, EXTENDED (--country flag)
    run_stages.py                   # existing, EXTENDED (adds italy stage)
    extract_rt.py                   # fits the reduced-SIR PINN and exports R_net(t)
    equity_metrics.py               # NEW (Stage 4)
    stat_tests.py                   # NEW (Stage 6)
    fig_dataquality.py              # existing, EXTENDED to two-country panel
  results/
    tables/  results/figures/  results/logs/
  reference/
    Case4.txt                        # existing (Millevoi validation fixture)
```

--------------------------------------------------------------------------
## 2. Stage-by-stage pipeline

### STAGE 0 — Environment setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU wheel, ~200MB
pip install numpy pandas scipy matplotlib
```
No GPU dependency is required for the base pipeline (see Sec 4). CUDA wheel
optional, see Sec 4.2.

### STAGE 1 — Preprocessing & data-quality audit (both countries)

Ghana: unchanged, run exactly as the course project did.
```bash
python src/prepare_ghana.py --input data/raw/ghana_hera.csv --outdir data/processed
```

Italy (NEW — `prepare_italy.py`, mirrors `prepare_ghana.py`'s design-decision
docstring pattern):
```bash
python src/prepare_italy.py --input data/raw/italy_case_hosp.csv --outdir data/processed
```
Design decisions to declare explicitly (mirroring Ghana's numbered list):
1. ISS series is already national daily incidence (not cumulative) — no
   monotonicity correction needed, unlike a naive DPC aggregation.
2. Infection series scaled by ascertainment ratio r=6 per Millevoi et al.
   (Italian ISTAT serological estimate), applied BEFORE building I_proxy.
3. Hospitalisation series retained natively (Italy has it; Ghana does not) —
   this is the key asymmetry the SEIQHRS synthesis (Stage 2) is designed to
   address for comparability.
4. Rolling D=5 day infection proxy, identical window to Ghana, for
   architectural parity across countries.
5. Millevoi-format export written for both real (dense) and later
   sparsified/weekly variants.

Data-quality figure (extend `fig_dataquality.py` to a two-panel Italy/Ghana
comparison — same gap-length logic, run on both master files):
```bash
python src/fig_dataquality.py --countries ghana italy
```

Expected output: `results/figures/fig_data_quality_ghana_italy.png`,
`audit_summary.json` for each country (completeness %, longest gap, n_gaps).

### STAGE 2 — Synthetic hospitalisation channel for Ghana (NEW)

`synth_hospitalisation.py` produces explicitly model-derived sensitivity
channels from Ghana's reported daily incidence. It does not produce observed
hospitalisation data or replicate the Italy two-observed-channel experiment.
The legacy `I_proxy` is verified as a trailing five-day incidence sum; the
Dwomoh-style I input is instead approximated as a trailing 14-day sum to
represent currently infectious people. This is an assumption, and it inherits
the Ghana master’s zero-encoded reporting gaps.

```bash
python src/synth_hospitalisation.py \
    --master data/processed/ghana_national_master.csv \
    --out data/processed/ghana_hospitalisation_synth.csv
```

The script requires a consecutive, finite, nonnegative master and verifies
that `I_proxy` matches the documented five-day sum. It writes both
`dH_in_synthetic_dwomoh_s2_s3` and
`dH_in_synthetic_dwomoh_s4_s5` as training targets. The clearer
`H_admissions_synthetic_*` aliases are numerically identical; occupancy is
retained separately as `H_occupancy_synthetic_*` for diagnostics only.
Scenario 1 is excluded because it yields a degenerate zero channel. Paired
scenarios 2/3 and 4/5 are grouped where the reduced equations cannot represent
the additional intervention distinction.

The Q equation is `dQ/dt = delta_q*I - (xi+n+sigma)*Q`; the `phi*E` term is
omitted because no exposed E series exists. The JSON sidecar records these
assumptions, input/master hash, scenario parameters, finite/nonnegative and
coverage checks, and the cautious dimensional comparison to Millevoi's
reported sigma range. `H_synthetic_seiqhrs` remains as a compatibility alias
for the scenario 4/5 daily-admissions channel; it is not occupancy.
Approximate paper-reported
scenario diagnostics and visual readings from Millevoi's figure are recorded
in the metadata as context only, never as fit targets or acceptance thresholds.

### STAGE 3 — Main experimental grid (extends `run_ghana.py`/`run_parallel.py`)

The primary `ReducedSIRPINN` engine and its existing runners remain unchanged;
the separate auxiliary-output ablation extends `pinn.py` without changing
that baseline model's API:

**3a. Ghana arm** — reuse `run_ghana.py` / `run_stages.py` / `run_parallel.py`
exactly as in the course project (daily / weekly / authentic mask regimes).

**3b. Italy arm** — NEW `run_italy.py`, identical interface to `run_ghana.py`,
but with an added synthetic-sparsification regime axis so Italy can be
degraded step-wise from dense to Ghana-like:
```bash
python src/run_italy.py \
    --country italy_primary \
    --master data/processed/italy_national_master.csv \
    --regimes daily weekly sparse20 sparse30 sparse40 \
    --modes split joint --seeds 1 2 3 \
    --origins 45 60 75 \
    --out results/tables/italy_grid.csv
```
Use the same origins and horizon for Ghana when making the primary
Ghana–Italy comparison; `run_ghana.py` defaults to `45 60 75` with horizon 15.
`sparseNN` regimes are produced by `sparsify.py`, which drops NN% of days at
random (fixed seed per regime) from the dense Italian series — this is the
controlled variable that interpolates between "Italy dense" and "Ghana-like
sparse", answering R2 directly.

**3c. Hospitalisation ablation arm** — use the separate
`ReducedSIRHospitalisationPINN` class, leaving the primary infection-only
`ReducedSIRPINN` unchanged. The ablation adds a non-negative H output head
that shares the infection-time feature trunk and an equally weighted,
channel-normalized supervised daily-admissions loss at the same training
dates as I. The published reduced-SIR residual still constrains only I and
`R_net`; there is no H ODE residual. The dedicated runner consumes the
audited synthesis file and verifies that it was generated from the exact
master supplied to the run:

```bash
python src/run_ghana_hospitalisation_ablation.py \
    --master data/processed/ghana_national_master.csv \
    --hospitalisation data/processed/ghana_hospitalisation_synth.csv \
    --out results/tables/ghana_hospitalisation_ablation.csv \
    --origins 45 60 75 --horizon 15 \
    --regimes daily weekly mask \
    --modes split joint --seeds 34 \
    --scenarios dwomoh_s2_s3 dwomoh_s4_s5
```

The runner pairs infection-only and auxiliary-H models at each origin,
regime, mode, and seed. Infection and admissions normalization scales are
estimated from training observations only. CSV results include paired MASE
deltas and explicitly mark the H target as synthetic and model-derived; the
JSON audit records protocol, source identity, fit/skip counts, equations, and
scenario provenance. The output is an exploratory sensitivity ablation, not
a reproduction of Millevoi's two-observed-channel model or a primary result.

Staged execution (extends `run_stages.py` with an `italy` stage so nothing is
lost on interruption, following the exact resume-safe pattern already built):
```bash
python src/run_stages.py --stage validation
python src/run_stages.py --stage split          # Ghana, ~50 min (as before)
python src/run_stages.py --stage italy_split     # NEW, ~1h on Italy's shorter series
python src/run_stages.py --stage joint           # Ghana, ~6.5h (as before)
python src/run_stages.py --stage italy_joint     # NEW, ~1.5h
```

### STAGE 4 — Equity metrics (NEW, `equity_metrics.py`)

Computes, per (method, regime) pair, across the combined Italy+Ghana grid:
- MASE gap = MASE(low-data regime) − MASE(high-data regime), paired by origin
- Peak-timing error: |argmax(pred window) − argmax(true window)| in days
- ECE / CRPS: requires predictive uncertainty; since the base PINN is
  deterministic, uncertainty is obtained from the across-seed ensemble spread
  (already logged by `run_parallel.py`'s multi-seed grid) rather than adding a
  new probabilistic head — declare this as the uncertainty proxy used.

```bash
python src/equity_metrics.py --grid results/tables/grid.csv \
    --italy-grid results/tables/italy_grid.csv \
    --out results/tables/equity_summary.csv
```

### STAGE 5 — PINN-derived R_t trajectories (`extract_rt.py`)

`ReducedSIRPINN.predict()` returns both fitted incidence and its learned
`R_net(t)`. The extractor fits that model to the selected arm's full `I_proxy`
series and writes the actual network output; it does not apply a finite-
difference approximation to the input incidence. Since `I_proxy` is a
preprocessed five-day target, the result is conditional on that target and
model configuration, not an independent official estimate.

Run separate fits for each country arm:
```bash
python src/extract_rt.py --master data/processed/ghana_national_master.csv \
    --country ghana --outdir results/rt --modes split joint --seeds 1 2 3
python src/extract_rt.py --master data/processed/italy_primary_national_master.csv \
    --country italy_primary --outdir results/rt --modes split joint --seeds 1 2 3
```

Italy-secondary is optional and should remain a separately labelled
multi-wave robustness arm. The output includes per-seed trajectories, daily
median/10th-90th percentile across seeds, per-fit summaries, and one figure
per country label. Input dates and I_proxy values, as well as fitted R_net
outputs, are validated before writing. Default PINN training can be
computationally expensive; use the epoch, collocation, device, and delta
options to document any deliberate configuration change. External references
remain absent until verified literature values are supplied.

### STAGE 6 — Origin-aware statistical comparisons (`stat_tests.py`)

- Wilcoxon signed-rank compares split and joint MASE after pairing seed fits
  within each origin and averaging the paired scores across seeds. The
  independent observation is the forecast origin, not each seed-origin fit.
- Friedman compares regimes within each country/arm and method. Seed scores
  are averaged within each origin and regime, and only complete origins enter
  the repeated-measures test.
- Mann–Whitney compares Ghana mask and the selected Italy sparse regime after
  averaging seeds within each origin. Country/arm samples are kept distinct;
  cross-country results are exploratory because country and source differences
  limit causal interpretation.
- `--italy-arm both` computes country-specific Wilcoxon/Friedman results once
  and produces separate Ghana-versus-Italy comparisons for each available
  Italy arm. It does not duplicate Ghana-only test rows or pool countries.
- Results with fewer than 10 independent origins in any tested group are
  labelled `exploratory_insufficient_independent_origins`. Ten is this
  pipeline's reporting threshold, not a universal guarantee of statistical
  adequacy. Expanding forecast windows can overlap across origins, so adequate
  origin counts alone do not prove statistical independence.

```bash
python src/stat_tests.py --grid results/tables/grid.csv \
    --italy-grid results/tables/italy_grid.csv \
    --out results/tables/significance.csv \
    --italy-arm primary
```

--------------------------------------------------------------------------
## 3. Benchmarks and expected numbers (carried over + extrapolated)

| Check | Expected value | Source |
|---|---|---|
| Case4 replication, split err_I | ~0.133 (within 6% of 0.1331 published) | course project Table 1 |
| Case4 replication, split err_Rt | ~0.49 (within 6% of 0.4744 published) | course project Table 1 |
| Ghana daily split MASE | ~2.07 | course project Table 2 |
| Ghana weekly split MASE | ~3.05 | course project Table 2 |
| Ghana mask split MASE | ~2.30 | course project Table 2 |
| Split faster than joint | ~5.7x on Ghana daily | course project Sec 4.4 |
| Split accuracy advantage | 30.6% (daily) narrowing to 5.7% (weekly) | course project Sec 4.5 |
| Split robustness loss | 47.5% under weekly vs 8.6% for joint | course project Sec 4.5 |
| **Italy dense split MASE (NEW, expected)** | should approach published Millevoi Case 6/7 accuracy, i.e. lower than any Ghana regime, since ISS data is dense and complete | to be measured, Stage 3b |
| **Italy sparse40 vs Ghana mask MASE (NEW, expected)** | if the "authentic mask is worse than any synthetic dropout of similar %" finding from the course project generalises, Italy-sparse40 should still slightly outperform Ghana-mask at equivalent nominal missingness, because Ghana's gaps are batched/clustered rather than uniformly random | to be measured, Stage 4 |
| **Synthetic H ablation (NEW, expected)** | adding H may reduce Rt error at flat-trend origins but add noise at turning points, mirroring Millevoi's own finding that the H channel helps most when I is noisy | to be measured, Stage 3c |

--------------------------------------------------------------------------
## 4. Hardware requirements and scaling guidance

### 4.1 Minimum baseline (what the course project ran on)
- 4-core CPU, 8 GB RAM, no GPU.
- Validation stage: ~35 min.
- Ghana split grid (6 origins x 3 regimes x 3 seeds = 54 fits): ~50 min.
- Ghana joint grid (same 54 fits, ~8x cost/fit): ~6.5 h.
- Networks are tiny (4x50, 4x100 units); a single fit cannot saturate a CPU
  core's vector units, so `run_parallel.py`'s one-torch-thread-per-worker
  design is the correct strategy — do NOT rely on intra-op threading or GPU
  for this workload size.

### 4.2 Extended grid cost (this thesis adds Italy + ablation + sparsification)
Rough multiplier on the course project's baseline grid:
- Italy arm: similar per-fit cost (comparable series length, ~90-120 days vs
  Ghana's 434 days — actually cheaper per fit since collocation count is
  fixed at 6000 regardless of series length).
- 5 Italy regimes (daily/weekly/sparse20/30/40) vs Ghana's 3 → roughly 1.7x
  more fits for the Italy arm alone.
- Hospitalisation ablation doubles the Ghana fit count (with-H vs without-H).
- Combined: expect roughly 3-4x the course project's total fit count.
- On the same 4-core/8GB baseline: validation unchanged (~35 min); full split
  grid (Ghana+Italy+ablation) ~3 h; full joint grid ~20-24 h. Use
  `run_stages.py`'s resume-safe staging exactly as before — split first,
  joint overnight, do not run both simultaneously without more cores.

### 4.3 If more cores are available (8-16 core workstation / departmental server)
- Increase `--workers` in `run_parallel.py` proportionally; scaling is close
  to linear because fits are embarrassingly parallel and each is
  single-threaded (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` already enforced).
- 8 cores: joint grid drops to ~10-12 h; 16 cores: ~5-6 h.
- No code changes needed — only `--workers N`.

### 4.4 If a GPU is available (optional, not required)
- Set `device: "cuda"` in `Config` (already supported via `cfg.device="auto"`
  auto-detecting CUDA in `pinn.py`).
- GPU benefit is LIMITED for this workload: networks are tiny (4x50/4x100),
  batch sizes are small (10-100), so kernel-launch overhead can make a single
  GPU fit slower than CPU. GPU becomes worthwhile only if the sparsification
  grid is expanded to larger networks or larger batches, OR if `run_parallel`
  is replaced by batching multiple independent fits onto ONE GPU via `vmap`/
  manual parameter batching — this is an optional Sec 4.5 extension, not a
  requirement.
- If a GPU IS used opportunistically (e.g., free Colab/Kaggle GPU), route only
  the joint-mode Italy sparse grid to it (the single most expensive sub-grid)
  while the CPU workstation handles Ghana + split grids in parallel — this
  simple two-machine split roughly halves wall-clock without any code change
  beyond `--device cuda` and sharding via `run_parallel.py`'s existing
  `--shard`/`--of` flags.

### 4.5 Multi-machine sharding (already supported, reuse as-is)
`run_parallel.py`'s `--shard`/`--of` flags let the grid be split across
multiple machines (e.g., personal laptop + lab PC + a Colab session):
```bash
# Machine 1 of 2
python src/run_parallel.py --workers 4 --shard 0 --of 2 --out results/tables/grid_shard0.csv
# Machine 2 of 2
python src/run_parallel.py --workers 4 --shard 1 --of 2 --out results/tables/grid_shard1.csv
```
Concatenate shard outputs before Stage 4/6. Cost-sorted round-robin dealing
(joint fits first) is already implemented, so no machine sits idle at the end.

### 4.6 Practical recommendation for this thesis
Given typical MPhil hardware access (a personal laptop, occasional access to
a shared lab machine, and free-tier Colab):
1. Run Stage 1-2 (preprocessing, SEIQHRS synthesis) locally — trivially cheap.
2. Run Stage 3 split grids (Ghana + Italy) overnight on a 4-8 core laptop.
3. Route the Stage 3 joint grids — the expensive half — to whatever machine
   has the most cores available for a multi-hour unattended run, sharded
   across 2 machines if accessible.
4. Stage 4-6 (equity metrics, Rt extraction, statistical tests) are cheap
   post-processing on the merged CSVs — no special hardware needed.
