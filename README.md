# Resumable PINN Experiments for Sparse COVID-19 Surveillance Data (Thesis Title: Physics-Informed Neural Networks for Infectious-Disease Modelling under Ghanaian Surveillance Constraints: An Adaptation of Millevoi et al’s Compartmental Model by Atua-Ntow, 2026/2027)

A reproducible experiment pipeline for evaluating a reduced SIR Physics-Informed Neural Network (PINN) under realistic surveillance-data limitations. The project compares Ghana's authentic reporting gaps with controlled sparsification of Italian surveillance data, using a split-versus-joint PINN design adapted from Millevoi, Pasetto, and Ferronato (2024).

This repository is designed for an MPhil Data Science thesis workflow: it separates the core Ghana–Italy-primary comparison from an optional Italy-secondary, multi-wave robustness arm; supports free-tier notebook constraints; and preserves completed experiment results through checkpointed CSV outputs.

> **Research use notice.** This repository is primarily for research and educational use. It does not provide clinical forecasts, public-health guidance, or operational estimates of disease transmission. Any inferred reproduction-number results are model-derived and require epidemiological interpretation.

## Research question

The primary question is whether a reduced SIR PINN maintains forecasting accuracy and stability when surveillance observations are sparse, delayed, or clustered into reporting gaps.

The central comparison is:

- **Ghana:** authentic surveillance incompleteness, represented by observed/unobserved reporting days in the HERA-derived national series.
- **Italy-primary:** a dense, single-wave comparator processed with the same infection-proxy construction and evaluated under controlled synthetic sparsity.

The optional Italy-secondary arm covers a longer, multi-wave series. It is useful for regime-shift robustness analysis, but must not be pooled with Italy-primary or interpreted as a clean data-richness comparator.

## Design overview

| Component | Ghana | Italy-primary | Italy-secondary |
|---|---|---|---|
| Intended role | Core low-fidelity surveillance arm | Core dense single-wave comparator | Optional Objective-3 robustness arm |
| Observation issue | Authentic clustered reporting gaps | Controlled random synthetic dropout | Controlled dropout plus multi-wave regime shifts |
| Main label | `ghana` | `italy_primary` | `italy_secondary` |
| Main use | Primary equity comparison | Primary equity comparison | Separate, caveated robustness analysis |
| Default priority | High | High | Last |

All primary analyses should compare Ghana with `italy_primary`. Do not relabel both Italy arms as simply `italy`, and do not concatenate them into a single result set for equity metrics or formal tests.

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── pinn.py
│   ├── prepare_ghana.py
│   ├── prepare_italy.py
│   ├── run_ghana.py
│   ├── run_italy.py
│   ├── run_parallel.py
│   ├── run_dual_gpu_shards.py
│   ├── checkpoint_grid_runner.py
│   ├── extract_rt.py
│   ├── fig_dataquality.py
│   ├── equity_metrics.py
│   └── stat_tests.py
├── data/
│   ├── raw/                 # Not versioned unless redistribution is permitted
│   └── processed/           # Generated files; usually excluded from Git
├── reference/
│   └── Case4.txt            # Millevoi Case 4 validation input, if redistributable
├── results/
│   ├── figures/
│   ├── rt/
│   └── tables/
└── notebooks/
    └── experiment_pipeline.ipynb
```

Recommended `.gitignore` entries are included later in this README.

## Installation

Create and activate an isolated Python environment, then install the scientific stack:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows PowerShell

pip install --upgrade pip
pip install numpy pandas scipy matplotlib torch
```

For notebook execution, additionally install Jupyter:

```bash
pip install jupyter ipykernel
```

Verify that PyTorch can see your intended accelerator:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPUs:', torch.cuda.device_count())"
```

## Data preparation

### Ghana

Prepare the national Ghana analysis master from the HERA-format source data:

```bash
PYTHONPATH=src python src/prepare_ghana.py \
  --input data/raw/gha_subnational_covid19_hera.csv \
  --outdir data/processed
```

Expected key outputs include:

```text
data/processed/ghana_national_master.csv
data/processed/ghana_weekly.csv
data/processed/ghana_millevoi_format.csv
data/processed/revision_log.csv
data/processed/audit_summary.json
```

The Ghana master contains the 5-day rolling infection proxy `I_proxy` and the authentic reporting mask `observed_simple`. Ghana's zero-incidence days must not be silently treated as genuine zero transmission; the preprocessing audit identifies plausible reporting gaps.

### Italy

Prepare Italy-primary and Italy-secondary as separate data products. The exact input construction depends on the licensed/downloaded ISS source files and the selected date windows.

```bash
# Primary: dense, single-wave comparator
PYTHONPATH=src python src/prepare_italy.py \
  --input data/raw/italy_primary_raw.csv \
  --outdir data/processed/italy_primary

# Secondary: optional full multi-wave robustness arm
PYTHONPATH=src python src/prepare_italy.py \
  --input data/raw/italy_secondary_raw.csv \
  --outdir data/processed/italy_secondary
```

Place or rename the final master files consistently:

```text
data/processed/italy_primary_national_master.csv
data/processed/italy_secondary_national_master.csv
```

Before fitting, verify that Italy sparse masks have been added to each relevant master file:

```text
observed_sparse20
observed_sparse30
observed_sparse40
```

Synthetic sparsity should be generated separately for the primary and secondary arms. Do not reuse a file or mask generated for one arm in the other.

## Validation gate

Before country-level experiments, validate the shared `ReducedSIRPINN` implementation against Millevoi et al.'s Case 4 benchmark.

Published Table 4 targets are:

| Training method | Relative error for \(I(t)\) | Relative error for \(R_t\) |
|---|---:|---:|
| Split | 0.1331 | 0.4744 |
| Joint | 0.1411 | 0.4961 |

Run the cheaper split gate first:

```bash
PYTHONPATH=src python src/pinn.py \
  --repo reference \
  --out results/tables/validation_case4.json \
  --mode split \
  --epochs-split-data 3000 \
  --epochs-split-ode 1000 \
  --collocation 6000 \
  --seed 34
```

Then validate the joint formulation before beginning a joint country grid:

```bash
PYTHONPATH=src python src/pinn.py \
  --repo reference \
  --out results/tables/validation_case4.json \
  --mode joint \
  --epochs-joint 5000 \
  --collocation 6000 \
  --seed 34
```

If the implementation is materially inconsistent with the published benchmark, stop and diagnose the Case 4 input, scaling, time normalization, seed, batching, collocation count, and epoch settings before investing compute in country experiments.

### Separate hospitalisation-channel ablation

`ReducedSIRPINN` remains the primary infection-only model and its API is
unchanged. For a distinct auxiliary-channel ablation,
`ReducedSIRHospitalisationPINN` adds a non-negative `H_net(t)` head that
shares the infection-time feature trunk. The channel-normalized supervised H
loss can therefore affect the shared representation used by the infection
output. Normalize H by its own positive scale before fitting; `predict()`
returns `(I_pred, R_net, H_pred)` with I and H returned in their original
units.

The H channel is intentionally supervised-only: the reduced-SIR residual
continues to constrain I and `R_net`, and no hospitalisation ODE is added.
For Ghana, the training target is `dH_in_synthetic_<scenario>`, the synthetic
daily admission inflow (`sigma*Q + psi*I`), not occupancy and not observed
hospitalisation data.
Its generating script verifies the Ghana five-day `I_proxy`, derives a
separate approximate I series from a trailing 14-day sum of daily incidence,
and evolves Q using `dQ/dt = delta_q*I - (xi+n+sigma)*Q`. The 14-day duration,
zero-encoded reporting gaps, and Dwomoh parameter scenarios remain explicit
assumptions; the exposed-compartment term is omitted because no E series is
available. Occupancy is exported for diagnostics but is never used as the
auxiliary training target. Therefore this is an exploratory sensitivity
ablation, not replication of Millevoi's two-observed-channel model or a
primary result. Use aligned observations at the same training dates for I and
H, and label Ghana H as synthetic in all outputs and reporting.

The class-level API expects I and H targets normalized by their own scales:

```python
from pinn import Config, ReducedSIRHospitalisationPINN

model = ReducedSIRHospitalisationPINN(
    Config(seed=1), tf=len(I_values), scale_I=I_scale, scale_H=H_scale
)
model.fit_joint(t_scaled, I_values_scaled, H_values_scaled)
I_pred, R_net, H_pred = model.predict(t_scaled)
```

For split training, call `fit_split(t_scaled, I_values_scaled,
H_values_scaled)`. H must be aligned to the same time points passed as
`t_scaled`; do not pass unobserved or synthetic H values as if they were
surveillance observations. The end-to-end synthesis and matched runner are:

```bash
python src/synth_hospitalisation.py \
  --master data/processed/ghana_national_master.csv \
  --out data/processed/ghana_hospitalisation_synth.csv

python src/run_ghana_hospitalisation_ablation.py \
  --master data/processed/ghana_national_master.csv \
  --hospitalisation data/processed/ghana_hospitalisation_synth.csv \
  --out results/tables/ghana_hospitalisation_ablation.csv \
  --origins 45 60 75 --horizon 15 \
  --regimes daily weekly mask --modes split joint --seeds 34 \
  --scenarios dwomoh_s2_s3 dwomoh_s4_s5
```

Both scripts write JSON audits. The runner validates synthesis provenance,
date alignment, the master-file hash, finite nonnegative admissions, and
matched infection-only versus auxiliary-channel output rows. Training
normalization uses each origin/regime's training data only. Results compare
the same forecast protocol and report paired MASE differences; the synthetic
channel remains explicitly marked as model-derived, never as surveillance.

## Experiment protocol

The forecasting protocol is expanding-origin:

1. Fit with observations before origin \(t_k\).
2. Forecast a 15-day horizon after \(t_k\).
3. Repeat the exact origin/regime/seed combination for split and joint modes.
4. Use MASE as the primary accuracy metric; MAE and RMSE are supplementary.

Use the same core model, infection-proxy convention, forecast horizon, metric definitions, random seeds, and training origins across the main country arms. Ghana and Italy-primary default to origins `45 60 75` with a 15-day horizon; these are feasible for Italy-primary's 91-day window. Each runner validates that every requested origin leaves a complete forecast horizon in its processed master.

### Observation regimes

| Arm | Regimes |
|---|---|
| Ghana | `daily`, `weekly`, `mask` |
| Italy-primary | `daily`, `weekly`, `sparse20`, `sparse30`, `sparse40` |
| Italy-secondary | `daily`, `weekly`, `sparse20`, `sparse30`, `sparse40` |

`mask` denotes Ghana's authentic observed-day mask. Italy `sparseNN` regimes refer to controlled random observation masks. They are not interchangeable missingness mechanisms.

## Running the grids

### Split-first policy

Run every `split` grid before its corresponding `joint` grid. Split experiments are substantially less expensive and provide an interpretable result table even if compute access ends before joint training completes.

Recommended order:

1. Ghana split.
2. Italy-primary split.
3. Ghana joint.
4. Italy-primary joint.
5. Italy-secondary split and joint only after the core comparison is complete.

### CPU parallel runner

The original CPU parallel runner is suited to many independent, inexpensive fits. It writes results at the end of its command, so it is best used only where the run can finish safely or where the environment is persistent.

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python src/run_parallel.py \
  --master data/processed/ghana_national_master.csv \
  --out results/tables/grid_ghana_split.csv \
  --workers 4 \
  --origins 45 60 75 \
  --regimes daily weekly mask \
  --modes split \
  --seeds 1 2 3 \
  --horizon 15
```

### Resumable single-GPU/CPU runner

For a free-tier on something like Colab, interruptible runtime, use `checkpoint_grid_runner.py`. It atomically rewrites the output CSV after every completed fit. On restart, rerun the exact same command; already completed `(origin, regime, method, seed)` combinations are skipped.

```bash
PYTHONPATH=src python src/checkpoint_grid_runner.py \
  --country ghana \
  --master data/processed/ghana_national_master.csv \
  --out results/tables/grid_ghana_split.csv \
  --origins 45 60 75 \
  --regimes daily weekly mask \
  --modes split \
  --seeds 1 2 3 \
  --horizon 15 \
  --max-fits 6
```

The same command resumes work in a later session. The required `--country`
selects Ghana's `mask` regime or Italy's `sparseNN` regimes. For Italy-primary,
select the matching arm and processed master:

```bash
PYTHONPATH=src python src/checkpoint_grid_runner.py \
  --country italy_primary \
  --master data/processed/italy_primary_national_master.csv \
  --out results/tables/italy_primary_grid_split.csv \
  --origins 45 60 75 \
  --regimes daily weekly sparse20 sparse30 sparse40 \
  --modes split \
  --seeds 1 2 3 \
  --horizon 15
```

The `.identity.json` sidecar binds the checkpoint to the country, master-file
content, origins, horizon, regimes, modes, seeds, and model settings; do not
change those when resuming. Use a new `--out` path or explicitly pass
`--overwrite` to start a different experiment. Omit `--max-fits` when the
environment is persistent or you have enough time for the remaining tasks.

### Kaggle one- or two-GPU runner

`run_dual_gpu_shards.py` is designed for Kaggle or another environment where one or more CUDA GPUs are exposed. It does not assume that two GPUs exist. At startup, it detects visible devices and uses one process per available GPU. If only one GPU is assigned, it runs correctly with one worker.

Before running it, inspect the allocated hardware:

```bash
nvidia-smi -L
```

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("Visible GPU count:", torch.cuda.device_count())
for gpu_id in range(torch.cuda.device_count()):
    print(gpu_id, torch.cuda.get_device_name(gpu_id))
```

Ghana split example:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python src/run_dual_gpu_shards.py \
  --runner ghana \
  --master data/processed/ghana_national_master.csv \
  --out results/tables/grid_ghana_split.csv \
  --origins 45 60 75 \
  --regimes daily weekly mask \
  --modes split \
  --seeds 1 2 3 \
  --horizon 15
```

Italy-primary split example:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python src/run_dual_gpu_shards.py \
  --runner italy \
  --country italy_primary \
  --master data/processed/italy_primary_national_master.csv \
  --out results/tables/italy_primary_grid_split.csv \
  --origins 45 60 75 \
  --regimes daily weekly sparse20 sparse30 sparse40 \
  --modes split \
  --seeds 1 2 3 \
  --horizon 15
```

When invoking `src/run_italy.py` directly, pass `--country italy_primary` or
`--country italy_secondary`. Origins are validated against the master length
and horizon; every origin must satisfy `1 <= origin <= number_of_days - horizon`.
The default origins (45, 60, 75) fit the 91-day primary window with horizon 15.

The runner sends completed results to a parent process, which alone writes the CSV checkpoint. This prevents concurrent workers from corrupting the output. A task interrupted before completion is not marked complete and is retried on the next invocation.

> **Important:** `run_dual_gpu_shards.py` imports `run_italy.py`. If your file is named `run_italy-2.py`, rename it to `src/run_italy.py` before running. Hyphens are not valid in ordinary Python module imports.

### GPU guidance

Use GPU acceleration only after confirming it improves representative fit time on your allocated hardware. Small fully connected PINNs do not necessarily receive large speed-ups from a free/shared GPU because process launch, Python control flow, and small kernel overhead remain important.

For this project, GPU usage is most attractive for the costly `joint` fits. CPU multi-process execution can remain competitive for many independent `split` fits. Never run multiple CPU-style `run_parallel.py` workers against one GPU; use the GPU-aware sharded runner instead.

## Country labels and result files

Result CSVs must preserve the country/arm identity:

```text
ghana
italy_primary
italy_secondary
```

Do not use the ambiguous label `italy` in final merged grids. The downstream equity and statistical scripts require properly labelled Italy rows to avoid pooling the primary and secondary arms.

Recommended result files:

```text
results/tables/grid_ghana_split.csv
results/tables/grid_ghana_joint.csv
results/tables/italy_primary_grid_split.csv
results/tables/italy_primary_grid_joint.csv
results/tables/italy_secondary_grid_split.csv
results/tables/italy_secondary_grid_joint.csv
```

If split and joint results are written to separate CSVs, concatenate only compatible files before downstream analysis. Retain `country`, `origin`, `regime`, `method`, and `seed` columns intact.

## Data-quality figure

Generate the descriptive data-quality figure after preprocessing. The
Italy-secondary panel is plotted only when its audit records
`ascertainment_mode: none`, its proxy is finite, and its post-cutoff positive
coverage is at least as complete as its raw daily incidence. Stale or
zero-corrupted secondary masters are rejected rather than plotted.

If the secondary master was created with an older preprocessing path, back up
the existing secondary products and regenerate them from the national raw
source before running the figure:

```bash
PYTHONPATH=src python src/prepare_italy.py \
  --input data/raw/italy_case_hosp.csv \
  --outdir data/processed \
  --arm secondary \
  --ascertainment none \
  --overwrite
PYTHONPATH=src python src/sparsify.py \
  --master data/processed/italy_secondary_national_master.csv \
  --country italy_secondary \
  --pct 20 30 40 \
  --inplace
```

The preparation step replaces existing secondary products when `--overwrite`
is supplied; back them up first. Recreate the deterministic sparse masks
afterward because the new master does not carry the previous mask columns.

```bash
PYTHONPATH=src python src/fig_dataquality.py
```

Expected output:

```text
results/figures/fig_data_quality.png
```

The figure should show Ghana's authentic no-report days/gap lengths, Italy-primary as the clean single-wave benchmark, and Italy-secondary explicitly annotated as a regime-shift robustness arm rather than a like-for-like data-richness comparison.

## Equity metrics

The equity analysis quantifies within-country performance loss under lower observation density and seed-to-seed stability.

- **MASE gap:** deterioration relative to the same country's dense (`daily`) baseline.
- **Absolute MASE gap and log ratio:** accompany percentage change; the percentage is left undefined when dense MASE is at or below `1e-8`, avoiding unstable divisions near zero.
- **Ensemble IQR:** instability across random seeds for the same country, method, regime, and origin.

Main Ghana–Italy-primary analysis:

```bash
PYTHONPATH=src python src/equity_metrics.py \
  --grid results/tables/grid_ghana.csv \
  --italy-grid results/tables/grid_italy.csv \
  --out results/tables/equity_summary_primary.csv
```

The Italy grid must include a `country` column distinguishing `italy_primary` and `italy_secondary`. By default, Italy-secondary is excluded from the primary equity result.
The core Ghana–Italy-primary output is restricted to origins present in both
grids. Its audit JSON records the retained origins and percentage-gap
denominator policy. When dense MASE is near zero, use the absolute MASE gap
or epsilon-stabilized log MASE ratio rather than interpreting a percentage.

Optional separate secondary robustness read:

```bash
PYTHONPATH=src python src/equity_metrics.py \
  --grid results/tables/grid_ghana.csv \
  --italy-grid results/tables/grid_italy.csv \
  --out results/tables/equity_summary_primary.csv \
  --include-secondary
```

This writes a separate secondary-only output. Do not merge that output into the main Ghana–Italy-primary equity table.

## Statistical tests

The formal tests answer complementary questions:

| Test | Question addressed |
|---|---|
| Wilcoxon signed-rank | Is split versus joint performance consistently different across forecast origins? |
| Friedman test | Does performance differ across observation regimes within a country/arm and method, using forecast origins as repeated measures? |
| Mann–Whitney U | Do origin-level Ghana clustered-missingness scores differ from Italy controlled-sparsity scores? |

Seeds are repeated fits nested within each forecast origin. The tests first
average seed-level scores within an origin, so seeds do not inflate the
independent sample size. Country/arm comparisons remain separate; `--italy-arm
both` runs Ghana-only tests once and adds one separately labelled
Ghana-versus-Italy comparison per available Italy arm. Results with fewer than
10 independent origins in any tested group are labelled
`exploratory_insufficient_independent_origins`; meeting the 10-origin reporting
threshold does not by itself establish confirmatory evidence, and expanding
forecast windows may overlap across origins. Cross-country Mann–Whitney
comparisons remain exploratory because country and data-source differences
limit causal interpretation.

Run the primary analysis:

```bash
PYTHONPATH=src python src/stat_tests.py \
  --grid results/tables/grid_ghana.csv \
  --italy-grid results/tables/grid_italy.csv \
  --out results/tables/significance_primary.csv \
  --italy-arm primary
```

Run both Italy arms only when you want a separately labelled sensitivity output:

```bash
PYTHONPATH=src python src/stat_tests.py \
  --grid results/tables/grid_ghana.csv \
  --italy-grid results/tables/grid_italy.csv \
  --out results/tables/significance_by_italy_arm.csv \
  --italy-arm both
```

Italy-secondary Mann–Whitney rows must retain their caveat: multi-wave dynamics can confound a missingness-mechanism comparison. Treat those as Objective-3 robustness evidence, not a clean causal comparison of clustered versus random missingness.

## R_t extraction

`src/pinn.py` exposes a learned `R_net(t)` output through
`ReducedSIRPINN.predict()`. The extractor fits that reduced-SIR PINN to the
selected arm's full `I_proxy` series and exports the model's `R_net` output;
it does not estimate R_t with a finite-difference formula. Since `I_proxy` is
itself the preprocessed five-day incidence proxy, interpret the fitted curve
as model-derived for that target series, not as an independent official
epidemiological estimate.

```bash
# Ghana
PYTHONPATH=src python src/extract_rt.py \
  --master data/processed/ghana_national_master.csv \
  --country ghana \
  --outdir results/rt \
  --modes split joint \
  --seeds 1 2 3

# Italy-primary
PYTHONPATH=src python src/extract_rt.py \
  --master data/processed/italy_primary_national_master.csv \
  --country italy_primary \
  --outdir results/rt \
  --modes split joint \
  --seeds 1 2 3

# Italy-secondary: optional robustness presentation
PYTHONPATH=src python src/extract_rt.py \
  --master data/processed/italy_secondary_national_master.csv \
  --country italy_secondary \
  --outdir results/rt \
  --modes split joint \
  --seeds 1 2 3
```

This is a fresh full-series PINN fit, not a lookup of a previously saved
checkpoint. Default training settings match `pinn.py` and can take substantial
time; `--epochs-joint`, `--epochs-split-data`, `--epochs-split-ode`,
`--collocation`, `--device`, and `--delta` expose the model configuration.
Results are written as `rt_trajectories_<country>.csv` (date/mode/seed
predictions), `rt_median_<country>.csv` (daily median and 10th-90th percentile
across seeds), `rt_summary_<country>.csv` (per-fit summaries), and
`fig_rt_<country>.png`. Inputs and fitted R_net outputs are checked for daily
coverage, finiteness, and non-negativity before output is written. Add
external reference points only after verifying them in the cited literature;
do not invent reference R_t values.

## Reproducibility and checkpointing

### Preserve results before reset

Free Colab and Kaggle sessions can end unexpectedly. The working directory may be temporary. Keep the checkpoint CSV on persistent storage, or download it at the end of every session.

In Colab, mount Google Drive before long experiments:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Then write checkpoints directly to Drive, for example:

```text
/content/drive/MyDrive/MPhil_PINN/results/tables/grid_ghana_split.csv
```

### Do not mix experiment designs

Use a new output filename if you change any of the following:

- master dataset or preprocessing version;
- country arm;
- origin set or forecast horizon;
- observation regimes or sparse-mask generation seed;
- model architecture, loss, epoch count, collocation count, or optimizer settings;
- code revision that affects results.

A checkpoint CSV is only valid for the exact experiment specification that produced it. Do not use `--overwrite` unless you intentionally want to discard the existing checkpoint and start afresh.

### Record the environment

For every final grid, save:

```bash
python --version
pip freeze > results/environment_requirements.txt
nvidia-smi > results/gpu_info.txt 2>&1 || true
git rev-parse HEAD > results/git_commit.txt
```

Also record the commands, origin sets, GPU count, random seeds, raw-data provenance, and dates used for preprocessing.

## Suggested .gitignore

Do not commit raw surveillance data, confidential data, large generated outputs, virtual environments, notebook checkpoints, or trained model checkpoints unless you have confirmed that redistribution is allowed.

```gitignore
# Python
__pycache__/
*.py[cod]
.venv/
venv/
.env

# Jupyter
.ipynb_checkpoints/

# Data and generated outputs
data/raw/
data/processed/
results/
*.pt
*.pth
*.ckpt

# Operating system/editor files
.DS_Store
.vscode/
.idea/

# Temporary checkpoint writes
*.tmp
```

If you need small synthetic examples for reproducibility, place them in `data/example/` and explicitly allow that directory in Git.

## Responsible interpretation

- Ghana reporting gaps are treated as a surveillance-quality feature, not evidence of zero disease transmission.
- Italy synthetic dropout is a controlled stress test, not authentic missingness.
- Italy-primary is the main comparator; Italy-secondary is a separate multi-wave robustness arm.
- Forecast performance does not establish causal effects of surveillance quality, intervention, variant dynamics, or health-system policy.
- The fitted R_t series is the reduced-SIR PINN's learned `R_net(t)` for the processed `I_proxy` target. Present its model/data assumptions and do not treat it as a replacement for official epidemiological estimation.

## Citation

If you use or adapt this pipeline, cite the methodological reference:

```bibtex
@article{millevoi2024pinn,
  title={A Physics-Informed Neural Network approach for compartmental epidemiological models},
  author={Millevoi, Chiara and Pasetto, Davide and Ferronato, Marco},
  journal={PLOS Computational Biology},
  volume={20},
  number={9},
  year={2024}
}
```

Also cite the precise Ghana and Italy data sources, preprocessing dates, and any external R_t references used in final figures.

## License

Choose a license before making the repository public. For an open research-code repository, MIT or BSD-3-Clause is often suitable. Data licensing is separate: do not assume raw surveillance data can be redistributed under the software license.
