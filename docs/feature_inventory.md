# Feature inventory — what the origin repos could do, and where it lives now

Purpose: nothing from `4dvarnet-fm-opencode` or `NOSC` may disappear without being either
**ported**, **planned** (with the enabling design already in place) or **dropped on purpose with a
written reason**. This file is the audit; `PLAN.md` only tracks the open items. Provenance and the
independence decision: `PROVENANCE.md`.

Legend: **done** = present in this tree and covered by a test · **planned** = not written yet, but
the abstraction it needs exists (link to the PLAN item) · **dropped** = deliberately not carried
over.

> **Rewritten 2026-09-14, and the previous version was wrong in one systematic way.** It was
> inherited byte-for-byte from the NOSC porting prototype, where the `oceanml3d/` package was a
> greenfield rewrite that genuinely did not yet contain the `4dvarnet-fm-opencode` science. It
> marked QG dynamics, conditional flow matching, SDA, the parameter head, ETKF, Strong/Weak 4D-Var
> and the joint filters as **planned**. In *this* repository all of them are **done** — this tree
> *is* the upstream code, imported wholesale, not a reimplementation of it. Sixteen lines flipped
> from planned to done. The lesson is worth keeping: a document copied between two repositories
> describes whichever one it was written for, and here it had been describing the other one since
> the initial commit.

Verified against the working tree on 2026-09-14, after the `oceanml3d/` namespace move and the NOSC
transplant. Gate at that point: `pytest -q -m "not slow"`, `ruff check .` clean.

## How to read the paths

Two model families live side by side, and the distinction is load-bearing:

* `oceanml3d/models/*.py` — the toy / Lorenz / QG family from `4dvarnet-fm-opencode`, built through
  `oceanml3d/models/factory.py` (`model_type` dispatch) and trained by `train.py`.
* `oceanml3d/models/ocean/` — the gridded-ocean family from the NOSC port, built through the
  `@register_model` registry (`oceanml3d/registry.py`) and the `oceanml3d.models` entry-point group,
  trained by the `oceanml3d` CLI.

They are not interchangeable, which is why the transplant put the second one a level down:
`models/fourdvarnet.py` (the 620-line L96 `FourDVarNetSolver`) and
`models/ocean/fourdvarnet/model.py` (the 149-line gridded `FourDVarNet`) are different classes with
the same name, and a module cannot coexist with a package of the same name.

## 1. `4dvarnet-fm-opencode`

### 1.1 Dynamical systems and data

| Capability | Original | Status | Where / why |
|---|---|---|---|
| Lorenz-63 dynamics | `models/lorenz63_dynamics.py`, `data/lorenz63.py` | **done** | both carried over as `oceanml3d/models/lorenz63_dynamics.py` and `oceanml3d/data/lorenz63.py`; the port's RK4 registry copy is `oceanml3d/dynamics/lorenz.py` (4th-order convergence tested) |
| Lorenz-96, two-scale (the one the fm case studies use) | `models/lorenz96_dynamics.py` | **done** | `oceanml3d/models/lorenz96_dynamics.py`, plus `lorenz96_2scale` in `oceanml3d/dynamics/lorenz.py`; golden file pending (PLAN 3.0) |
| Lorenz-96, single-scale (textbook) | `data/lorenz96.py` | **done** | `oceanml3d/data/lorenz96.py`; `lorenz96` + `config/data/lorenz96.yaml` + `scripts/make_toy_data.py` on the registry side |
| Toy state as a task of the ocean framework | *(separate pipeline)* | **done** | a K-dim state is stored as a `(time, 1, K)` grid and goes through the same datamodule / patches / export — `tests/test_toy_and_assimilation.py` |
| Quasi-geostrophic dynamics (1-layer, ψ, neural, interp) | `models/qg*.py`, `data/qg*.py` | **done** | `oceanml3d/models/{qg_base,qg_dynamics,qg1l_dynamics,qg_psi_dynamics,qg_interp}.py`, `oceanml3d/data/{qg,qg_neural}.py`; 9 test files (`test_qg_*.py`) |
| Random-parameter / random-bias datasets (joint estimation) | `data/random_param_dataset.py`, `random_bias_dataset.py` | **done** | both present under `oceanml3d/data/`; `tests/test_random_param_dataset.py` |
| Normalisation, dataloader | `data/normalization.py`, `dataloader.py` | **done** | carried over as-is; the registry side adds `OceanDataModule` (train-split stats cached in the run dir) |
| Lazy xarray data layer, patch extraction | *(absent — added by the port)* | **done (added)** | `oceanml3d/data/{open,patches,datamodule,transforms,augment}.py`, `oceanml3d/catalog.py` |
| Pre-computed norm-stat scripts | `precompute_*_norm_stats.py` | **dropped** | on the registry path these are computed once at `setup()` and saved to `norm_stats.json`; the toy path still normalises in `data/normalization.py` |

### 1.2 Models

| Capability | Original | Status | Where / why |
|---|---|---|---|
| Direct U-Net | `models/direct_unet.py`, `unet.py` | **done** | both kept (`oceanml3d/models/direct_unet.py`, `unet.py`); the registry equivalent is `nosc_unet` (`models/ocean/nn/unet2d.py`) with heads/attention options |
| 4DVarNet solver (grad solver + learned prior) | `models/fourdvarnet.py`, `solver.py`, `residual.py` | **done** | all three kept; gridded counterpart `models/ocean/fourdvarnet/model.py` (`ablation=gradsolver`) |
| Conditional flow matching (vanilla, joint, coupled, Tweedie, predict-state) | `models/vanilla_cfm.py`, `interpolant.py` | **done** | `oceanml3d/models/vanilla_cfm.py` carries `VanillaCFM`, `JointCFM`, `JointCFMCoupled`, `PredictStateCFM`, `TweedieCFM`; `interpolant.py` alongside. `tests/test_vanilla_cfm.py`, `test_joint_estimation*.py`, `test_interpolant.py` |
| Score-based data assimilation (SDA) | `models/sda.py`, `evaluation/sda_sampler.py` | **done** | both kept; `UnconditionalPriorCFM` / `ConditionalPriorCFM`; `tests/test_sda.py`, `test_sda_sampler.py`, `test_eval_sda_l96.py` |
| Joint state+parameter head | `models/param_head.py` | **done** | `StateParamHead`, `StateParamUNet`, `StateParamModel`; `tests/test_param_head.py` |
| MONAI U-Net adapter | `models/monai_unet_adapter.py` | **done (optional)** | kept as `oceanml3d/models/monai_unet_adapter.py` behind the `monai` extra — MONAI pulls torch forward, so it installs in its own environment and its test skips when absent |
| Model construction | *(in `train.py`)* | **done (redesigned)** | extracted to `oceanml3d/models/factory.py` during the namespace move: it was imported *from the root script* `train.py`, which only resolves with the repo root on `sys.path`. `train.py` re-exports it |
| `DynamicsBase` + `get_dynamics()` factory | `models/dynamics.py` | **done (both forms)** | the original `if/elif` factory is `oceanml3d/models/dynamics.py`; the port's registry version is `oceanml3d/dynamics/base.py`. PLAN 3.0 converges them |

### 1.3 Training

| Capability | Original | Status | Where / why |
|---|---|---|---|
| Lightning module, losses | `training/lightning_module.py`, `losses.py` | **done** | kept; `training/losses.py` was **merged** at the transplant — `StateMSELoss` (toy path) and the port's `weighted_mae` / `weighted_mse` / `gradient_loss` / `sobel` (gridded path, NaN-aware) now live in one module with the distinction documented at the top |
| Two-stage pipeline (stage-1 prior, stage-2 conditional) | `training/pipeline.py`, `stage1.py`, `stage2.py` | **done** | all three kept, *and* the registry mechanism (`BaseOceanModel.stages` / `set_stage`, `training.stages`) came over with the port |
| Loss grouping, per-depth weights, vertical modes | *(absent — from NOSC)* | **done (added)** | `oceanml3d/training/{loss_grouping,weights,vertical_modes,optim}.py` |
| Checkpoint compatibility, config persistence | `tests/test_checkpoint_compat.py`, `test_config_persistence.py` | **done** | both suites kept; `VersioningCallback` (`training/callbacks.py`) writes config + git hash + norm stats next to the checkpoints |

### 1.4 Evaluation

| Capability | Original | Status | Where / why |
|---|---|---|---|
| EnKF (inflation, Gaspari-Cohn localisation) | `evaluation/baselines.py::EnKF` | **done** | `oceanml3d/evaluation/baselines.py::EnKF`; registry counterpart `models/ocean/assimilation/model.py::EnsembleKalmanFilter` (a registered model, so it exports products) |
| ETKF | `::ETKF` | **done** | `oceanml3d/evaluation/baselines.py::ETKF`; `tests/test_baselines_enkf.py` |
| Strong / weak 4D-Var | `::Strong4DVar`, `::Weak4DVar` | **done** | same module; `tests/test_baselines_strong4dvar.py`, `test_baselines_weak4dvar.py`, `test_qg_baselines_4dvar.py`. **Open bug**: `Strong4DVar` diverges on a minority of windows — `tests/KNOWN_FAILURES.md` |
| Joint (state+parameter) variants of the four filters | `::Joint*`, `::Joint*L96` | **done** | all seven present: `JointWeak4DVar`, `JointStrong4DVar`, `JointEnKF`, `JointETKF`, `JointEnKFL96`, `JointETKFL96`, `JointStrong4DVarL96`; `tests/test_joint_estimation_l96*.py` |
| Optimal interpolation | *(absent)* | **done (added)** | `models/ocean/assimilation/model.py::OptimalInterpolation`, from the port |
| RMSE, parameter RMSE, spread | `evaluation/metrics.py` | **done** | kept here; `oceanml3d-eval` adds `gridded_rmse`, `ensemble_scores` for scoring exported products |
| CRPS, energy score | `evaluation/metrics.py` | **done** | `crps` was returning **minus** the CRPS — fixed, see `tests/KNOWN_FAILURES.md`; fair/unbiased forms in `oceanml3d-eval/metrics/ensemble.py` |
| Spread-skill ratio, rank histogram | *(absent)* | **done (added)** | `oceanml3d-eval` — the calibration diagnostics an ensemble reanalysis needs |
| Explained variance, SW component metrics | `evaluation/metrics.py` | **planned** | PLAN 3.5 |
| Experiment runner, sweeps, tuning | `evaluation/run*.py`, `batch/*sweep*.py`, `tune_*.py` | **done** | `oceanml3d/evaluation/{run,run_l96,run_l96_sweep,run_l96_sweep2,run_qg_baselines,sweep_qg_baselines,tune_l96_weak4dvar,experiment,estimate_metrics}.py` all kept; Hydra multirun + `config/ablation/` cover new sweeps |
| Neural inference from a checkpoint | `evaluation/neural_inference.py` | **done** | kept; now builds through `oceanml3d/models/factory.py` like `train.py` does, so a model registered in one is usable from the other |
| Per-case reports and figures | `reports/**` (≈30 scripts), `demos/` | **kept as-is** | `reports/` (131 files), `demos/` (8), `batch/` (178) and `notebooks/` (7) are all still tracked and unchanged since the import. They are not in the wheel and nothing in `oceanml3d/` imports them — they import *it*, so the namespace move touched them. Triage is PLAN 3.x, not done |
| Case studies CS1…CS7 | `config/case_study/`, `config/experiment/*` | **done** | 78 presets under `config/experiment/` (67 from the toy/L96/QG side, 11 from the port), plus `config/{case_study,baselines,data,model,training,paths,ablation}/` groups |

## 2. `4dvarnet-ocean-reanalyses` — audited, reimplemented, no code carried

The porting prototype audited a third repository (1.3 k lines of Python, 5 test files, 6 YAML) and
its capabilities are present here, but — on the evidence of `docs/MIGRATION_PLAN.md` §0 and
`docs/porting_guide.md` — they arrived as a **reimplementation in the port**, not as copied code. It
is not one of the two origins named in `PROVENANCE.md` and the repository is not available locally,
so the right-hand column below is what this tree contains, not a diff against that source.

| Capability | Status | Where |
|---|---|---|
| Typed config schema (`conf/schema.py`) | **done (two of them)** | `oceanml3d/conf/schema.py` from the fm side and `oceanml3d/config_schema.py` from the port — see §4 |
| GLORYS / ERA5 readers, ocean grid | **done** | `oceanml3d/data/open.py` + `oceanml3d/catalog.py`; `scripts/prepare/` recipes |
| Dataset / dataloader | **done** | `oceanml3d/data/{patches,datamodule}.py` |
| Synthetic observation operators (nadir SSH, SST, Argo) | **done (superseded)** | `oceanml3d/obs/`: real repeat orbits, per-mission masks, cloud masks, virtual Argo — strictly richer than the 1-file original |
| 2D U-Net backbone | **done** | `oceanml3d/models/ocean/nn/unet2d.py` |
| Surface model (per-cell MLP) | **done (equivalent)** | the `linear` baseline (per-pixel map) and `nosc_unet`; a per-cell MLP is a ~20-line registry plugin |
| Interior model + sinusoidal depth embedding | **planned** | PLAN 3.7 — `depth_indices` gives the levels; the depth *embedding* variant is not ported |
| Weighted-depth MSE, SSH spectral loss | **partly done** | `oceanml3d/training/{loss_grouping,weights}.py` cover per-depth weighting; the spectral loss is PLAN 3.5 |
| Profile RMSE, spatial RMSE, anomaly correlation | **partly done** | `gridded_rmse` + `report --depth-profile` in `oceanml3d-eval`; anomaly correlation is PLAN 3.5 |
| Optimal interpolation baseline | **done (superseded)** | `optimal_interpolation` — a real OI, where the original was an 8-line placeholder |
| GLORYS preparation script | **done** | `scripts/prepare/recipes/glorys_gs_multidepth.yaml` |

## 3. `NOSC`

Covered in full by `docs/migration_nosc.md`, which maps every module of `code/multivar_drifter_`,
`process_data/` and `env/velocity_metrics` to its destination.

| Capability | Status | Where |
|---|---|---|
| Multivariate surface-current model (NOSC U-Net, multi-head) | **done** | `oceanml3d/models/ocean/nosc/`, registered as `nosc_unet` |
| 2D U-Net backbone, attention, heads | **done** | `oceanml3d/models/ocean/nn/` |
| Gridded baselines (passthrough, linear, climatology) | **done** | `oceanml3d/models/ocean/baselines/model.py` |
| OSSE chain: repeat orbits, per-mission masks, cloud masks | **done** | `oceanml3d/obs/{orbits,missions,sampling,pseudo_obs}.py`; `tests/test_obs_simulation.py`, `test_osse_smoke.py` |
| Virtual Argo profiles | **done** | `oceanml3d/obs/{argo,argo_virtual}.py`; `tests/test_argo.py` |
| Catalog, variable specs (`role: input/target/static/aux`) | **done** | `oceanml3d/{catalog,variables}.py`; `tests/test_catalog.py`, `test_variables.py`, `test_depth_variables.py` |
| Model registry + entry points | **done** | `oceanml3d/registry.py`, group `oceanml3d.models` in `pyproject.toml`; `tests/test_registry.py` |
| Product export in the v2 format | **done** | `oceanml3d/inference/{export,predict,product_contract}.py`; `tests/test_export.py`, `test_product_contract.py` |
| CLI | **done** | `oceanml3d/cli.py`, console script `oceanml3d` |
| Data preparation (GLORYS/ERA5 download, regrid recipes) | **done** | `scripts/prepare/`, behind the `prepare` extra |
| NOSC config importer | **done** | `scripts/import_nosc_config.py`; `tests/test_import_nosc_config.py` |
| ~190 notebooks, `config/xp/old/`, one-off plotting scripts | **dropped** | see `docs/migration_nosc.md` |

## 4. Known duplication, carried on purpose for now

The transplant put two working systems in one tree. These are the places where that shows, and each
is a PLAN item rather than an accident:

| Duplication | Why it exists | Resolution |
|---|---|---|
| `oceanml3d/conf/schema.py` (412 lines, ~25 dataclasses) **and** `oceanml3d/config_schema.py` (146 lines, `validate_config`) | two Hydra entry points: `config/config.yaml` for `train.py`, `config/main.yaml` for the `oceanml3d` CLI. Disjoint importers — the first is used by 5 test files, the second only by `cli.py` and `test_config_validation.py` | PLAN 3.x: one schema, or an explicit statement that the two CLIs are permanent |
| `oceanml3d/models/dynamics.py` (`if/elif` factory) **and** `oceanml3d/dynamics/base.py` (registry) | the fm code and the port each brought a dynamics abstraction | PLAN 3.0 — converge on the registry |
| `models/fourdvarnet.py` **and** `models/ocean/fourdvarnet/` | genuinely different models (L96 solver vs gridded ocean), same name | **not** a duplication to remove; the `models/ocean/` split is the fix |
| `config/experiment/` mixes 67 toy/L96/QG presets with 11 gridded ones | one flat directory, two pipelines | naming convention or a subdirectory split |

## 5. Deliberate drops — the full list

| Dropped | Reason |
|---|---|
| `notebooks/` of both origins (≈190 from NOSC) | exploratory; nothing imports them |
| `eval_*.py` / `evaluate_all*.py` beyond those kept at the root | one script per (model × system) pair; `oceanml3d-eval run -b <benchmark> -p <product>` replaces the reporting half |
| `precompute_*_norm_stats.py` | folded into `OceanDataModule.setup` on the registry path |
| `config/xp/old/` (NOSC) | superseded config generations |
| `archive/`, `logs/`, `L96_*_PROGRESS.md` | historical artefacts; `.gitignore`d or removed |
| **the 24 commits published on `4dvarnet-fm-opencode` after the import** | deliberate: this repository is independent of its origin, with no upstream remote and no rebase. `PROVENANCE.md` |

Anything in this table that turns out to be needed can be re-added: both origins are public, and the
mapping above says exactly where each piece would land.
