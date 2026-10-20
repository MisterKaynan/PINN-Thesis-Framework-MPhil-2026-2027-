# Resumable PINN Experiments for Sparse COVID-19 Surveillance Data

**Thesis:** *Physics-Informed Neural Networks for Infectious-Disease Modelling under Ghanaian Surveillance Constraints: An Adaptation of Millevoi et al.'s Compartmental Model* — Atua-Ntow, 2026/2027.

This repository evaluates a reduced SIR Physics-Informed Neural Network (PINN) under incomplete surveillance. It compares Ghana's authentic reporting gaps with deterministic synthetic observation masks applied to two distinct Italy data arms. The primary scientific comparison is Ghana versus Italy-primary. Italy-secondary is an optional multi-wave robustness arm and must remain separately labelled.

> **Research use.** This code is for research and education. It does not provide clinical forecasts, public-health guidance, or operational estimates of transmission. The learned `R_net(t)` is model-derived and must be interpreted with its data and model assumptions.

## Study Design

| Arm | Role | Observation design | Main label |
|---|---|---|---|
| Ghana | Core low-fidelity surveillance arm | HERA-derived reporting mask | `ghana` |
| Italy-primary | Core dense, single-wave comparator | Deterministic nested 20/30/40% dropout masks | `italy_primary` |
| Italy-secondary | Optional multi-wave robustness arm | Same controlled masks over a longer series; regime shifts remain a confounder | `italy_secondary` |

Do not pool Italy-primary and Italy-secondary or call both simply `italy`. Cross-country differences are exploratory: country, data source, and preprocessing differ, so they do not isolate a causal effect of missingness. Italy-secondary is not a clean data-richness comparator.

## Repository Layout

```text
.
├── README.md
├── docs/PIPELINE.md
├── italy_ghana_pinn_colab.ipynb
├── italy_ghana_pinn_local_refactor_2.ipynb
├── src/
│   ├── pinn.py
│   ├── prepare_ghana.py
│   ├── prepare_italy.py
│   ├── sparsify.py
│   ├── run_ghana.py
│   ├── run_italy.py
│   ├── run_stages.py
│   ├── run_stages_italy_arms.py
│   ├── checkpoint_grid_runner.py
│   ├── run_parallel.py
│   ├── synth_hospitalisation.py
│   ├── run_ghana_hospitalisation_ablation.py
│   ├── extract_rt.py
│   ├── fig_dataquality.py
│   ├── equity_metrics.py
│   └── stat_tests.py
├── data/
│   ├── raw/                 # Source data; check redistribution rights
│   └── processed/           # Prepared masters, weekly series, audits
├── reference/               # Case 4 validation fixture, if available
└── results/
    ├── figures/
    ├── logs/
    ├── rt/
    └── tables/
```

The `_v1.py` and `_v2.py` scripts under `src/` are retained historical implementations, not the canonical entry points in the commands below. `docs/PIPELINE.md` contains additional pipeline notes; some of its older stage descriptions may not match the current runner defaults.

## Setup

Use Python 3.10 or newer and install the runtime dependencies in a virtual environment. There is currently no root `requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy matplotlib torch
```

On macOS/Linux, activate with `source .venv/bin/activate`. For notebook work, also install `jupyter ipykernel`.

Check the selected PyTorch device:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPUs:', torch.cuda.device_count())"
```

Run commands from the repository root. The scripts create their output directories where applicable. Keep raw inputs and large generated results out of version control unless data and output redistribution are permitted.

### Local and Cloud Runs

The same `src/` commands and experiment arguments work locally and in hosted notebooks such as Colab or Kaggle. Locally, keep the project, processed masters, and results on disk; on a cloud runtime, first make the project and required data available, install the same Python dependencies, and check whether the assigned runtime exposes CUDA. A GPU is optional and does not guarantee a faster fit.

Cloud working disks may be temporary. Save the checkpoint CSV and its matching `.identity.json` and `.audit.json` sidecars to a mounted drive or other persistent storage, and keep the exact master CSV available at the same experiment revision. Resume with the same command, output path, and settings; changing data or settings requires a new output path. Record Python/package versions, code revision, and device details for either environment.

## Inputs and Preprocessing

### Ghana

The current HERA source file is `data/raw/gha_subnational_covid19_hera.csv`:

```bash
python src/prepare_ghana.py --input data/raw/gha_subnational_covid19_hera.csv --outdir data/processed
```

Key outputs are `ghana_national_master.csv`, `ghana_weekly.csv`, `ghana_millevoi_format.csv`, `ghana_national_excl_unspecified.csv`, `revision_log.csv`, and `audit_summary.json`. The master includes the five-day rolling incidence proxy `I_proxy` and `observed_simple`; the latter represents reporting availability, not whether incidence is truly zero. Review the audit before fitting.

### Italy

The current source is `data/raw/italy_case_hosp.csv`, with daily `date`, `new_infections`, and `new_hospitalisations` fields. The two arms have different windows and ascertainment settings. Use the wrapper to prepare both arms consistently, generate their sparse masks, and audit the saved masks:

```bash
python src/run_stages_italy_arms.py --stage prepare
```

The wrapper configures Italy-primary with the fixed `r=6` correction over the primary window (2020-02-20 through 2020-05-20). Italy-secondary uses the restricted correction: `r=6` through the configured validity cutoff and reported incidence afterward. These dates and correction choices are study assumptions, not independently validated epidemiological claims. Preprocessing rejects missing daily rows unless `--missing-days zero` is explicitly selected; any imputed dates are recorded in the audit.

Each arm produces a national master, weekly data, Millevoi-format export, and JSON audit. The wrapper then adds `observed_sparse20`, `observed_sparse30`, and `observed_sparse40` to each master. Masks are deterministic per arm, nested, and have exact full-series dropout counts; missingness within an individual training window can differ from the nominal percentage. An in-place sparsification keeps a `.pre_sparsify.csv` backup. Do not reuse masks across arms.

The prepare stage refuses to replace existing arm outputs by default. If you intentionally want to regenerate the current processed files, back them up first, then run `python src/run_stages_italy_arms.py --stage prepare --overwrite`; the wrapper will regenerate the masters and masks. For direct preprocessing, `prepare_italy.py` also exposes `--arm`, `--start`, `--end`, `--ascertainment`, `--valid-until`, `--missing-days`, and `--overwrite`.

## Validation Gate

Validate the shared `ReducedSIRPINN` against Millevoi et al.'s Case 4 benchmark before interpreting country-level fits. Published Table 4 relative-error targets are:

| Method | Relative error for $I(t)$ | Relative error for $R_t$ |
|---|---:|---:|
| Split | 0.1331 | 0.4744 |
| Joint | 0.1411 | 0.4961 |

Run both modes with the default configuration:

```bash
python src/pinn.py --repo reference --out results/tables/validation_case4.json --mode both
```

The defaults are 3,000 split data epochs, 1,000 split ODE epochs, 5,000 joint epochs, 6,000 collocation points, and seed 34. If validation is materially inconsistent, investigate the fixture, scaling, time normalization, seed, collocation, and training settings before launching the experiment grids.

## Main Experiment Grids

The expanding-origin protocol trains only on observations before each origin and forecasts the following 15 days. The shared primary comparison uses origins `45 60 75`, regimes `daily weekly mask` for Ghana and `daily weekly sparse20 sparse30 sparse40` for Italy-primary, and matching model modes and seeds.

### Ghana

Run split and joint fits with the same three seeds used by the Italy wrapper:

```bash
python src/run_ghana.py \
  --master data/processed/ghana_national_master.csv \
  --out results/tables/ghana_grid.csv \
  --origins 45 60 75 --horizon 15 \
  --regimes daily weekly mask \
  --modes split joint --seeds 1 2 3
```

The Ghana runner's command-line defaults use seed 34; specify seeds explicitly when making matched comparisons. Results include PINN and baseline rows, and an audit sidecar is written for the run.

### Italy-primary and Italy-secondary

After the Italy prepare stage, run each grid stage:

```bash
python src/run_stages_italy_arms.py --stage split
python src/run_stages_italy_arms.py --stage joint
```

This writes `italy_primary_grid_split.csv`, `italy_primary_grid_joint.csv`, `italy_secondary_grid_split.csv`, and `italy_secondary_grid_joint.csv` under `results/tables/`. Italy-primary uses origins `45 60 75`; Italy-secondary uses `100 300 500 700 900`. Both use a 15-day horizon and seeds `1 2 3`. The wrapper checks that masks are present, binary, nested, and have the expected full-arm dropout counts before fitting.

The explicit `run_italy.py` entry point accepts one arm at a time via `--country italy_primary` or `--country italy_secondary`. Check that requested origins fit the selected master and leave a complete forecast horizon.

### Resumable Checkpoint Runner

For interruptible local or hosted sessions, `checkpoint_grid_runner.py` atomically checkpoints after each fit and writes `.identity.json` and `.audit.json` sidecars. On restart, rerun the same command and output path to skip completed fit keys:

```bash
python src/checkpoint_grid_runner.py \
  --country ghana \
  --master data/processed/ghana_national_master.csv \
  --out results/tables/ghana_checkpoint.csv \
  --origins 45 60 75 --horizon 15 \
  --regimes daily weekly mask \
  --modes split joint --seeds 1 2 3 \
  --max-fits 6
```

The country must be `ghana`, `italy_primary`, or `italy_secondary`; choose only regimes valid for that arm. The checkpoint identity includes the master content and model/protocol settings. Do not change the experiment while resuming. Use a new output path for a different design; `--overwrite` intentionally removes that output's checkpoint state.

`run_parallel.py` is the non-checkpointing process-pool runner. It writes the CSV after its fits finish and supports `--workers`, `--quick` (smoke testing only), and machine sharding with `--shard`/`--of`. Do not use `--quick` for research results. `run_stages.py` is the Ghana legacy staged workflow; it uses origins `100 140 180 220 260 300` and should not be substituted for the matched `45 60 75` Ghana grid used in the primary Ghana–Italy comparison.

## Combine and Analyse Results

Keep the country and arm labels, and retain `origin`, `regime`, `method`, and `seed`. For the equity and statistical scripts, provide a Ghana grid and an Italy CSV that contains both Italy arms with their explicit `country` values. For example, after both modes finish:

```python
from pathlib import Path
import pandas as pd

tables = Path("results/tables")
italy = pd.concat(
    [pd.read_csv(tables / name) for name in (
        "italy_primary_grid_split.csv",
        "italy_primary_grid_joint.csv",
        "italy_secondary_grid_split.csv",
        "italy_secondary_grid_joint.csv",
    )],
    ignore_index=True,
)
italy.to_csv(tables / "italy_grid.csv", index=False)
```

The Ghana runner writes both modes into the requested `ghana_grid.csv` when invoked as above.

### Equity Metrics

```bash
python src/equity_metrics.py \
  --grid results/tables/ghana_grid.csv \
  --italy-grid results/tables/italy_grid.csv \
  --out results/tables/equity_summary_primary.csv
```

The primary output compares Ghana with Italy-primary at common origins. It reports within-arm MASE gaps relative to the dense baseline, absolute and log gaps, across-seed IQR, and audit details. Percentage gaps are undefined when dense MASE is at or below the configured epsilon. Add `--include-secondary` only for a separate Italy-secondary robustness output; it is not merged into the primary comparison.

### Statistical Tests

```bash
python src/stat_tests.py \
  --grid results/tables/ghana_grid.csv \
  --italy-grid results/tables/italy_grid.csv \
  --out results/tables/significance_primary.csv \
  --italy-arm primary
```

Tests include split-versus-joint Wilcoxon, within-country Friedman comparisons across regimes, and origin-level Ghana-versus-Italy Mann–Whitney comparisons. Repeated seeds are averaged within origins rather than counted as independent observations. `--italy-arm both` emits separately labelled comparisons; it does not pool Italy arms. Results with fewer than 10 independent origins are marked exploratory, and overlapping forecast windows and cross-country confounding remain limitations.

### Data-Quality Figure

```bash
python src/fig_dataquality.py
```

The figure uses the processed masters and audits in `data/processed/`, saves to `results/figures/fig_data_quality.png`, and includes Italy-secondary only when its coverage and audit checks pass. Its secondary panel is labelled as a multi-wave robustness arm, not a like-for-like data-richness comparator.

## Optional Ghana Hospitalisation Ablation

The main model remains infection-only `ReducedSIRPINN`. The separate `ReducedSIRHospitalisationPINN` adds a non-negative `H_net(t)` head sharing the infection-time feature trunk. H is supervised only; the reduced-SIR residual still constrains I and `R_net`, and no hospitalisation ODE is added.

For Ghana, the H target is synthetic daily admission inflow (`dH_in_synthetic_*`), not observed admissions or occupancy. The generator verifies the five-day Ghana `I_proxy`, derives a separate approximate I series from a trailing 14-day incidence sum, and evolves Q with an assumed reduced equation that omits E because no exposed series is available. These assumptions and Dwomoh scenario groupings are documented in the generated metadata. Treat the result as an exploratory sensitivity ablation, not a primary result or reproduction of Millevoi's two-observed-channel model.

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

Both scripts write audit metadata. The runner verifies synthesis provenance and date alignment, uses training-only normalization, and reports paired infection-only versus auxiliary-channel results. Keep the synthetic origin of H explicit in all tables and reporting.

## PINN-Derived $R_t$

`extract_rt.py` fits the selected arm's full `I_proxy` series and exports the learned `R_net(t)`; it does not estimate $R_t$ by finite differences. Example for Ghana:

```bash
python src/extract_rt.py \
  --master data/processed/ghana_national_master.csv \
  --country ghana --outdir results/rt \
  --modes split joint --seeds 1 2 3
```

Use `italy_primary` or `italy_secondary` and the corresponding master for Italy. This is a fresh full-series fit, not a checkpoint lookup, and may take substantial time. The output includes per-fit trajectories, seed summaries, and a figure. The fitted series is conditional on the preprocessed five-day `I_proxy`; it is not an official independent epidemiological estimate.

## Reproducibility

- Preserve the exact raw-data provenance, preprocessing command, date windows, audit JSON, code revision, and environment for each final grid.
- Use a new output path when changing the master, arm, origins, horizon, masks, seed set, model settings, or code affecting results.
- Keep checkpoint CSVs and sidecars on persistent storage during hosted runs; copy them off temporary runtimes before reset.
- Record `python --version`, `pip freeze`, `git rev-parse HEAD`, and GPU information where applicable.
- Do not overwrite processed Italy products without backing them up. Rebuild sparse masks after regenerating a master.

## Responsible Interpretation

- Ghana's reporting gaps are surveillance limitations, not evidence of zero transmission.
- Italy's sparse masks are controlled synthetic dropout, not authentic missingness.
- Italy-primary is the primary comparator; Italy-secondary is a separately reported multi-wave robustness arm.
- Forecast metrics do not establish causal effects of surveillance quality, interventions, variants, or health-system policy.
- Report `R_net(t)` as a model output conditional on the target series and assumptions.
- Cite the precise national data sources, preprocessing choices and dates, and any external epidemiological references used.

## Citation

If you use or adapt the PINN method, cite Millevoi, Pasetto, and Ferronato (2024):

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

Choose a software license before public release. Data licensing is separate; do not assume source surveillance data can be redistributed under the software license.