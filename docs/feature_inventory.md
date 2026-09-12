# Feature inventory — what the two original generic repos could do, and where it lives now

Purpose: nothing from `4dvarnet-fm-opencode` or `4dvarnet-ocean-reanalyses` may disappear without
being either **ported**, **planned** (with the enabling design already in place) or **dropped on
purpose with a written reason**. This file is the audit; `PLAN.md` only tracks the open items.

Procedure for closing a *planned* line: `docs/porting_guide.md`. Equivalence harness: `tests/legacy/`.

Legend: **done** = usable today with a test · **planned** = not written yet, but the abstraction it
needs exists (link to the PLAN item) · **dropped** = deliberately not carried over.

Sources audited: `4dvarnet-fm-opencode` @ master (≈39 k lines of Python, 60 test files),
`4dvarnet-ocean-reanalyses` @ master, `NOSC` @ master and @ first_implementation.

## 1. `4dvarnet-fm-opencode`

### 1.1 Dynamical systems and data

| Capability | Original | Status | Where / why |
|---|---|---|---|
| Lorenz-63 dynamics | `models/lorenz63_dynamics.py`, `data/lorenz63.py` | **done** | `oceanml3d/dynamics/lorenz.py` (RK4, 4th-order convergence tested) |
| Lorenz-96, two-scale (the one the fm case studies use: NO slow + NO x J fast, external forcing, clamp) | `models/lorenz96_dynamics.py` | **done** | `lorenz96_2scale` in `oceanml3d/dynamics/lorenz.py`; equivalence case written, golden file pending (PLAN 3.0) |
| Lorenz-96, single-scale (textbook) | `data/lorenz96.py` | **done** | `lorenz96` + `config/data/lorenz96.yaml`, `scripts/make_toy_data.py` |
| Toy state as a task of the ocean framework | *(separate pipeline)* | **done** | a K-dim state is stored as a `(time, 1, K)` grid: same datamodule/patches/export |
| Quasi-geostrophic dynamics (1-layer, ψ, neural, interp) | `models/qg*.py`, `data/qg*.py` | **planned** | PLAN 3.1 — needs `Dynamics` with a 2D state; the base class already allows it |
| Random-parameter / random-bias datasets (joint estimation) | `data/random_param_dataset.py`, `random_bias_dataset.py` | **planned** | PLAN 3.3 — needs a per-sample parameter channel (`role: aux` already exists) |
| Normalisation, dataloader | `data/normalization.py`, `dataloader.py` | **done** | `OceanDataModule` (train-split stats, cached in the run dir) |
| Pre-computed norm-stat scripts | `precompute_*_norm_stats.py` | **dropped** | computed once at `setup()` and saved to `norm_stats.json`; no separate step |

### 1.2 Models

| Capability | Original | Status | Where / why |
|---|---|---|---|
| Direct U-Net | `models/direct_unet.py`, `unet.py` | **done** | `nosc_unet` (`models/nn/unet2d.py`), with heads/attention options |
| 4DVarNet solver (grad solver + learned prior) | `models/fourdvarnet.py`, `solver.py`, `residual.py` | **done** | `models/fourdvarnet/model.py` (`fourdvarnet`, `ablation=gradsolver`) |
| Conditional flow matching (vanilla, joint, coupled, Tweedie, predict-state) | `models/vanilla_cfm.py`, `interpolant.py` | **planned** | PLAN 3.2 — the two-stage hook (`stages` / `set_stage`) and the ensemble product format (v2) are in place |
| Score-based data assimilation (SDA) | `models/sda.py`, `evaluation/sda_sampler.py` | **planned** | PLAN 3.2, same enablers |
| Joint state+parameter head | `models/param_head.py` | **planned** | PLAN 3.3 |
| MONAI U-Net adapter | `models/monai_unet_adapter.py` | **dropped** | third-party backbone swap; the model registry makes it a 30-line plugin if needed |
| `DynamicsBase` + `get_dynamics()` factory | `models/dynamics.py` | **done (redesigned)** | `oceanml3d/dynamics/base.py` with a registry instead of an `if/elif` factory |

### 1.3 Training

| Capability | Original | Status | Where / why |
|---|---|---|---|
| Lightning module, losses | `training/lightning_module.py`, `losses.py` | **done** | `models/base.py`, `training/losses.py` (+ grouping, Sobel, uncertainty) |
| Two-stage pipeline (stage-1 prior, stage-2 conditional) | `training/pipeline.py`, `stage1.py`, `stage2.py` | **done (mechanism)** | `BaseOceanModel.stages` / `set_stage`, `training.stages`, `training.stage_trainer`; the CFM/SDA models that use it are PLAN 3.2 |
| Checkpoint compatibility, config persistence | `tests/test_checkpoint_compat.py`, `test_config_persistence.py` | **done** | `VersioningCallback` writes config + git hash + norm stats next to the checkpoints |

### 1.4 Evaluation

| Capability | Original | Status | Where / why |
|---|---|---|---|
| EnKF (inflation, Gaspari-Cohn localisation) | `evaluation/baselines.py::EnKF` | **done** | `models/assimilation/model.py::EnsembleKalmanFilter` (a registered model, so it exports products) |
| ETKF | `::ETKF` | **planned** | PLAN 3.4 — same class structure, deterministic square-root update |
| Strong / weak 4D-Var | `::Strong4DVar`, `::Weak4DVar` | **planned** | PLAN 3.4 |
| Joint (state+parameter) variants of the four filters | `::Joint*`, `::Joint*L96` | **planned** | PLAN 3.3/3.4 |
| Optimal interpolation | *(reanalyses repo)* | **done** | `models/assimilation/model.py::OptimalInterpolation` |
| RMSE, parameter RMSE, spread | `evaluation/metrics.py` | **done** | `oceanml3d-eval`: `gridded_rmse`, `ensemble_scores` |
| CRPS, energy score | `evaluation/metrics.py` | **done** | `oceanml3d-eval/metrics/ensemble.py` (fair/unbiased forms, tested) |
| Spread-skill ratio, rank histogram | *(absent)* | **done (added)** | idem — the calibration diagnostics an ensemble reanalysis needs |
| Explained variance, SW component metrics | `evaluation/metrics.py` | **planned** | PLAN 3.5 |
| Experiment runner, sweeps, tuning | `evaluation/run*.py`, `batch/*sweep*.py`, `tune_*.py` | **done (redesigned)** | Hydra multirun (`-m`) + `config/ablation/` + the benchmark runner's cache |
| Per-case reports and figures | `reports/**` (≈30 scripts), `demos/` | **dropped** | one-off paper figures; the reusable part is `oceanml3d_eval/plots.py` + `report --depth-profile` |
| Case studies CS1…CS7 | `config/case_study/`, `config/experiment/*` (≈60 files) | **planned** | PLAN 3.6 — one `config/data/*.yaml` per case study; the naming convention is already there |

## 2. `4dvarnet-ocean-reanalyses`

| Capability | Original | Status | Where / why |
|---|---|---|---|
| Typed config schema | `conf/schema.py` | **done (redesigned)** | `oceanml3d/config_schema.py`: dataclasses for the stable groups + `validate_config` (a dict of variables cannot be a frozen dataclass) |
| GLORYS / ERA5 readers, ocean grid | `data/glorys.py`, `era5.py`, `grid.py` | **done** | `oceanml3d/data/open.py` (+ `catalog`), `scripts/prepare/regrid.py` recipes |
| Dataset / dataloader | `data/dataset.py`, `dataloader.py` | **done** | `oceanml3d/data/{patches,datamodule}.py` |
| Synthetic observation operators (nadir SSH, SST, Argo) | `data/observations.py` | **done (superseded)** | `oceanml3d/obs/`: real repeat orbits, per-mission masks, cloud masks, virtual Argo — strictly richer |
| 2D U-Net backbone | `models/backbone_unet2d.py` | **done** | `models/nn/unet2d.py` |
| Surface model (per-cell MLP) | `models/surface_model.py` | **done** | `linear` baseline (per-pixel map) and `nosc_unet`; a per-cell MLP is a 20-line plugin |
| Interior model + sinusoidal depth embedding | `models/interior_model.py` | **planned** | PLAN 3.7 — `depth_indices` gives the levels; the depth *embedding* variant is not ported |
| Weighted-depth MSE, SSH spectral loss | `training/losses.py` | **planned** | PLAN 3.5 (`loss_group_weights` already covers per-depth weighting) |
| Profile RMSE, spatial RMSE, anomaly correlation | `training/metrics.py` | **partly done** | `gridded_rmse` + `report --depth-profile` give the profile view; anomaly correlation is PLAN 3.5 |
| Optimal interpolation baseline | `evaluation/baselines.py` | **done** | `optimal_interpolation` model (a real OI, not the 8-line placeholder) |
| Case studies CS1–CS4 (sparse obs, wind bias, stress test) | `config/experiment/CS*.yaml` | **planned** | PLAN 3.6 |
| Training pipeline, report generation | `training/pipeline.py`, `reports/generate_report.py` | **done** | `oceanml3d` CLI, `oceanml3d-eval report` |
| GLORYS preparation script | `scripts/prepare_glorys.py` | **done** | `scripts/prepare/recipes/glorys_gs_multidepth.yaml` |
| Project conventions (AGENTS/PLAN/CHANGELOG, pytest gate) | root files | **done** | same conventions in both new repos |

## 3. `NOSC` (both branches)

Covered by `docs/migration_nosc.md`, which maps every module of `code/multivar_drifter_`,
`process_data/` and `env/velocity_metrics` to its destination. Summary: model, data pipeline,
multivariate mechanism, weights, OSSE chain, ablations and metrics are ported; the ~190 notebooks,
the `config/xp/old/` tree and the one-off plotting scripts are deliberately not.

## 4. Deliberate drops — the full list

| Dropped | Reason |
|---|---|
| `reports/**`, `demos/`, `notebooks/` of both repos (≈50 scripts) | one-off figures tied to a paper draft; regenerable from the cached scores. The reusable 5% is `oceanml3d_eval/plots.py`. |
| `eval_*.py` / `evaluate_all*.py` at the repo root (≈15 scripts) | one script per (model × system) pair; replaced by `oceanml3d-eval run -b <benchmark> -p <product>`. |
| `batch/*sweep*.py`, `rerun_*.py` | hand-rolled sweeps; Hydra `-m` + the ablation group cover them. |
| `precompute_*_norm_stats.py` | folded into `OceanDataModule.setup`. |
| MONAI adapter | optional third-party backbone; trivially re-addable as a plugin. |
| `archive/`, `logs/`, `L96_*_PROGRESS.md` | historical artefacts. |

Anything in this table that turns out to be needed can be re-added: the originals stay in their
repos, and the mapping above says exactly where each piece would land.
