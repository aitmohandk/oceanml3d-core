# Gridded models on real data — configuration and use

How to configure, validate, train, and export a model on **real gridded ocean data** through the
`oceanml3d` CLI. The two worked examples at the end are the two settings that exist today:

* **`nosc_15m_duacs`** — an OSE: real observations in, real (sparse) drifter velocities as targets.
* **`osse3d_gs21_multivar_unet`** — an OSSE: GLORYS12 as truth, a simulated observing system in,
  64 gridded targets out.

Companion documents: `adding_a_model.md` (writing a new model), `data_preparation.md` (getting the
files), `product_format.md` (what comes out), `migration_nosc.md` (if you are coming from the NOSC
repository), `feature_inventory.md` (where every capability lives).

---

## 1. Which path are you on?

This repository contains **two model families that do not mix**. Nothing below applies to the other
one, which since 2026-09-14 sits in reserve under `oceanml3d/legacy/` — set aside, not removed
(`oceanml3d/legacy/README.md`).

| | toy / Lorenz / QG (reserve) | **gridded ocean (this document)** |
|---|---|---|
| Models | `oceanml3d/legacy/models/*.py` | `oceanml3d/models/ocean/` |
| Built by | `legacy/models/factory.py`, `model_type` dispatch | `@register_model` registry + entry points |
| Driver | `python legacy/train.py experiment=<n>` | `oceanml3d <args>` |
| Hydra root | `config/legacy/config.yaml` | `config/main.yaml` |
| Config schema | `oceanml3d/legacy/conf/schema.py` | `oceanml3d/config_schema.py` |

`legacy/models/fourdvarnet.py` (the 620-line L96 `FourDVarNetSolver`) and
`models/ocean/fourdvarnet/model.py` (the 149-line gridded `FourDVarNet`) are **different classes
with the same name**. If you are reading `legacy/train.py`, you are on the wrong path for this
document.

---

## 2. Install and smoke-test

```bash
pip install -e '.[dev,plots]'        # + '.[zarr]' if your catalog points at .zarr stores
                                     # + '.[prepare]' only to download/regrid raw data
oceanml3d command=list-models
# climatology, enkf, fourdvarnet, linear, nosc_unet, optimal_interpolation, passthrough
```

`config/` and the driver scripts stay at the repository root: `@hydra.main(config_path=…)` resolves
relative to the file carrying the decorator, so an installed wheel gives you the *library*, and the
CLI must be run **from a clone**.

Two configs exist for running without any real data:

```bash
oceanml3d experiment=smoke training=debug        # surface-currents shape, synthetic
oceanml3d experiment=osse3d_smoke training=debug # OSSE-3D shape, synthetic
```

They rely on `scripts/make_synthetic_data.py`. Use them to check that an edit did not break the
plumbing before spending a GPU-hour. To exercise the whole chain — preparation tools included — on the
real task configuration, run [`tutorial.md` Part 1](tutorial.md#part-1--a-rehearsal-on-your-machine)
(`experiment=tutorial`).

---

## 3. Anatomy of an experiment

`config/main.yaml` composes five groups. **The order in `defaults:` is load-bearing:**

```yaml
defaults:
  - paths: local             # where the files are, per machine
  - data: surface_currents_15m   # the task: variables, domain, splits, patching
  - model: nosc_unet         # architecture + hyper-parameters
  - training: default        # loss, optimiser, Trainer
  - _self_                   # main.yaml's own body (export, seed) BEFORE the experiment...
  - experiment: null         # ...so the experiment can override it...
  - ablation: null           # ...and an ablation can override the experiment.
```

So the precedence is: `ablation` > `experiment` > `_self_` > the four groups. An experiment is a
**complete description** (it picks a `data` and a `model` via `override`); an ablation is a **delta**
of one or two keys.

```bash
oceanml3d experiment=osse3d_gs21_multivar_unet                    # reference run
oceanml3d experiment=osse3d_gs21_multivar_unet ablation=heads     # one delta
oceanml3d experiment=osse3d_gs21_multivar_unet paths=odyssey \
          training=ddp training.trainer.max_epochs=200            # ad-hoc overrides
oceanml3d -m experiment=osse3d_gs21_multivar_unet \
          ablation=baseline,heads,attention,vertical_modes        # Hydra multirun
```

Outputs land in `outputs/<experiment_name>/<YYYY-MM-DD_HH-MM-SS>/`, where `experiment_name` is
resolved from `${hydra:runtime.choices.experiment}` — i.e. **an experiment invoked purely with
command-line overrides and no `experiment=` gets a null run directory name.** Always go through an
experiment file.

---

## 4. The `data` group — defining a task

A task file lives in `config/data/<task>.yaml` and starts with `# @package data`.

### 4.1 `variables:` — the single source of truth

Every variable is `name: {source, var_name, role, …}`. This one block drives the data module, the
model's channel layout, the loss grouping and the export metadata. There is **no channel arithmetic
anywhere else** — that was the point of replacing NOSC's `multivar` dict.

```yaml
variables:
  ssh:       {source: ssh_l4_duacs_4th, var_name: zos,  role: input}
  lat:       {source: ssh_l4_duacs_4th, var_name: lat,  role: static}
  u_drifter: {source: drifters_aoml_15m_4th, var_name: u_drifter, role: target,
              standard_name: eastward_sea_water_velocity, units: m s-1}
```

**Roles** (`oceanml3d/variables.py::Role`):

| `role` | Fed to the net | Supervised | Notes |
|---|---|---|---|
| `input` | yes | no | NaN → 0 before the forward pass (`BaseOceanModel.inputs`) |
| `target` | no | yes | NaN **kept**; the loss ignores non-finite cells |
| `both` | yes (masked) | yes | the masked field is the input, the full field the target — the OSSE idiom |
| `static` | yes | no | time-invariant, broadcast over the window (lat, bathymetry) |
| `aux` | no | no | loaded for augmentations/diagnostics only (per-mission masks) |

A `VariableSet` **requires at least one target** or it raises at construction.

**Per-variable keys** (`VariableSpec`):

| Key | Effect |
|---|---|
| `source` | catalog key **or** a path; resolved by `oceanml3d/catalog.py` |
| `var_name` | name inside the file; defaults to the variable's own name |
| `mask` | catalog key of an observation mask; cells where `mask <= 0` become NaN |
| `transform` | registered transform applied at load time: `log_grad`, `log1p`, `identity` |
| `fill_nan` | constant fill, applied **before** the mask |
| `standard_name`, `units` | CF attributes written at export; **required for anything you want scored** |
| `depth_index` | exact `isel` position on the file's depth axis; `None` = surface/2D |
| `depth_m` | informative only |
| `group` | loss / head group; defaults to the variable's own name |

**`depth_indices:` expands into one variable per level.** This replaces NOSC's generated
`config/vars/*_gs21.yaml` fragments:

```yaml
thetao: {source: glorys_gs_multidepth, var_name: thetao, role: target,
         depth_indices: [0, 2, 4, 10], group: temperature, units: degC}
```

becomes `thetao_d00`, `thetao_d02`, `thetao_d04`, `thetao_d10`, all with `group: temperature`.
Duplicate indices raise. The naming is `<name>_d<ii>` with `ii` zero-padded to two digits.

**Removing a variable in an override:** set it to `null`. `VariableSet.from_config` skips it. This
is how `fourdvarnet_ssh_osse.yaml` reduces the 64-target OSSE-3D task to SSH only:

```yaml
data:
  variables: {argo_thetao: null, argo_mask: null, thetao: null, uo: null, vo: null}
```

### 4.2 `domain` and `eval_domain`

```yaml
domain:      {lat: [32, 44], lon: [-66, -54]}   # what is loaded and trained on
eval_domain: {lat: [33, 43], lon: [-65, -55]}   # 1-degree rim excluded from metrics
```

`domain` is applied at load time (`data/open.py`). `eval_domain` restricts every **validation and
test metric** computed during training (below), and is consumed again by `oceanml3d-eval` — it
exists so that the boundary artefacts of a patched reconstruction never enter a score.
`validate_config` refuses an `eval_domain` that extends outside `domain`.

### 4.3 `splits`

```yaml
splits:
  train: {time: ["2010-01-01", "2017-12-15"]}
  val:   {time: ["2018-01-01", "2018-12-10"]}
  test:  {time: ["2018-12-22", "2020-01-10"]}
```

All three are mandatory, and **disjoint with gaps longer than the window** (`osse3d_gs21`). The ocean
is correlated over weeks: validation days adjacent to the training period make the validation
optimistic, and a test window that overlaps the validation scores days the checkpoint was selected
on. The test still starts **10 days before** 2019-01-01, so the first day of the exported year is
covered by complete `time: 11` windows; `export.time` clips the product back to the calendar year.

`validate_config` refuses a `val` window that starts inside `train`, and a `test` window that starts
inside `val`. The NOSC reproduction (`surface_currents_15m`) keeps NOSC's published splits, which
overlap by 12 days, and says so with `splits_overlap_ok: true`.

### 4.3a Following training: what is logged, and what selects the checkpoint

Every epoch, on the validation split (and once on the test split at the end), in CSV and
TensorBoard under the run directory:

| Metric | What it is |
|---|---|
| `val/nrmse` | mean over targets of the RMSE in units of each target's train std. **Selects the checkpoint.** |
| `val/rmse_<group>` | RMSE in physical units, pooled over the group's targets — `temperature` in °C, `currents_u`/`currents_v` in m/s, `ssh` in m |
| `val/rmse_<target>` | the same per target, e.g. `val/rmse_thetao_d10`: the profile of the error with depth |
| `val/loss`, `val/<target>_loss`, `val/group_<g>_loss` | the training objective recomputed on val |

The RMSEs pool the squared errors over the whole epoch (not a mean of batch RMSEs), weight them with
the reconstruction weight like the exported product, and count only `eval_domain` and finite targets.
They do not depend on the loss, which `val/loss` does: `flat_sum` and `group_mean` aggregate
differently, `no_grad_loss` drops a term, `uncertainty` adds learned log-variances that can lower it
without lowering any error. So ablations are compared — and their checkpoints selected — on
`val/nrmse`, never on `val/loss`.

`training.monitor` changes the selection metric (any `val/` metric); `training.early_stopping`,
off by default, takes Lightning's arguments, e.g. `{patience: 15}`. The last epoch is always kept as
`last.ckpt` besides the `save_top_k` best.

### 4.4 `patch`, `stride`, `chunks`

```yaml
patch:  {time: 11, lat: 144, lon: 144}
stride: {time: 1,  lat: 136, lon: 136}
chunks: {time: 32}
```

* `patch.time` **is** the model's `window`. Everything downstream reads it from here.
* `stride < patch` gives overlap, which is what the reconstruction weights blend. `validate_config`
  rejects `stride > patch` (coverage gaps).
* The overlap must be at least `2 × rec_weight.crop`, or the cropped borders are never covered by
  any patch. With `crop: {lat: 4, lon: 4}`, an overlap of 8 cells is the minimum — the examples use
  exactly that (`144 − 136 = 8`).
* **`auto` (lat, lon)** fits both to the grid at run time — `osse3d_gs21` uses it, so the same task
  runs at any `OCEANML3D_TARGET_RES`. `patch` = the domain's cell count trimmed to a multiple of
  `data.patch_multiple` (16, what the U-Net's down-sampling divides); `stride` = `patch − 2 × crop`,
  the largest stride that still leaves no seam. 145 cells (1/12°) → 144 / 136, the former hand-set
  values; 49 cells (0.25°) → 48 / 40. The resolved numbers are printed, written back into the run's
  saved config, and validated like hand-set ones. `time` cannot be `auto`: it is the model's window.
* `chunks` is passed to dask at open time. It controls memory, not science.
* `PatchSpec.pad_to_cover` (default `True`) adds a final window flush to the end of each dimension so
  the domain is fully covered.

### 4.5 `norm_stats`

```yaml
norm_stats: null    # computed on the TRAIN split at setup(), cached in the run directory
```

Leave it `null`. This is the fix for NOSC's hard-coded statistics lists: the numbers are derived
from the train split only, saved next to the checkpoints by `VersioningCallback`, and reloaded for
inference. Set it explicitly only to reproduce an older run bit-for-bit; if you do,
`validate_config` checks that it has exactly one entry per variable.

### 4.6 `augment` — train split only

```yaml
augment:
  mission_dropout:
    obs: [ssh_obs]
    mask: obs_mask
    missions: [mask_jason3, mask_sentinel6, mask_sentinel3a, mask_sentinel3b, mask_saral, mask_hy2b]
    n_drop: [1, 2]
    p: 0.5
  obs_noise: {channels: {ssh_obs: 0.02}}   # re-draw instrument noise each time a sample is drawn
```

`mission_dropout` removes 1–k altimetry missions from the observed field of a sample, using the
per-mission masks written by `build_mask_dataset(per_mission=True)` and loaded as `role: aux`
variables. It answers "what does losing Jason-3 cost?" and doubles as an operational-robustness
regulariser. It **requires** `per_mission: true` in the `prepare` block and the `mask_<mission>`
variables declared as `aux`.

Augmentations run on numpy patches **before** normalisation, and only on the train split —
validation and test keep the deterministic baked noise.

### 4.7 `prepare` — OSSE only

This block declares the observing-system simulation. It is idempotent: existing outputs are skipped.

```yaml
prepare:
  pseudo_obs:
    ssh:
      truth: glorys_gs_surface          # catalog key of the truth
      truth_var: zos
      output: pseudo_obs_ssh_gs         # catalog key of the file to write
      missions: [jason3, sentinel6, sentinel3a, sentinel3b, saral, hy2b]
      historical: false                 # true -> activity windows, non-stationary constellation
      per_mission: true                 # also store mask_<mission> (needed by mission_dropout)
      real_mask: null                   # catalog key of a dated L3 mask to use REAL tracks instead
      noise_std: 0.02
      seed: 1234
    sst:
      truth: glorys_gs_multidepth
      truth_var: thetao
      depth_index: 0
      output: pseudo_obs_sst_gs
      clouds: {coverage: 0.6, scale_cells: 8, seed: 7}   # or real_mask: <L3 cloud mask>
      noise_std: 0.2
      seed: 4321
  virtual_argo:
    truth: glorys_gs_multidepth
    truth_var: thetao
    profiles: argo_profiles_gs          # coverage table from scripts/prepare/argo_profiles.py
    output: argo_virtual_thetao_gs21
    depth_indices: [0, 2, 4, ...]
    noise_std: 0.05
```

What the simulator actually does (`oceanml3d/obs/`):

* `orbits.py` — closed-form repeat ground-track model for inclined circular orbits. No SGP4: the
  ascending-node longitude advances by the mission's exact repeat-geometry drift, which encodes J2
  nodal precession implicitly and closes the repeat cycle exactly.
* `missions.py` — real mission parameters (inclination, cycle length, orbits per cycle).
  **`orbits_per_cycle` is full revolutions, not passes.** AVISO/CNES handbooks quote passes
  (half-revolutions); mixing the conventions roughly doubles the simulated sampling density and
  silently makes every downstream OSSE optimistic. `validate_missions()` runs at import and enforces
  a plausible LEO period so this cannot recur.
* `sampling.py` — rasterises tracks/swaths onto the truth grid, one boolean `obs_mask` per day.
* `pseudo_obs.py` — truth sampled where `obs_mask == 1`, plus Gaussian instrument noise. Written
  once to NetCDF, so train/val/test see the identical realisation.
* `argo_virtual.py` — keeps the real floats' *geometry* (position, date, vertical coverage) and
  replaces their values with the sampled truth, binned onto the grid, one variable per depth index.

Run it explicitly, or let `train` run it automatically when `data.prepare` is present:

```bash
oceanml3d command=prepare-obs experiment=osse3d_gs21_multivar_unet paths=local
```

---

## 5. The `paths` group — the catalog

An experiment never contains a filesystem path. It contains **logical keys**, and the mapping lives
in `config/paths/<site>.yaml` (or in `$OCEANML3D_CATALOG`):

```yaml
# @package paths
root: ${oc.env:OCEANML3D_DATA,./data}
datasets:
  ssh_l4_duacs_4th: {path: ssh_l4/SSH_L4_CMEMS_2010-01-01-2024-01-01_4th.nc,
                     source: CMEMS SEALEVEL_GLO_PHY_L4}
```

Relative paths are resolved against `root`; `${VAR}` is expanded; a real path passed where a key is
expected is passed through unchanged. Adding a machine means copying `local.yaml` to
`<site>.yaml` and running with `paths=<site>` — nothing else changes.

### 5.1 The OSSE-3D keys, and the two rules a site file follows

`local.yaml`, `datarmor.yaml` and `jeanzay.yaml` all carry the sixteen keys of the gridded path,
including the seven the OSSE-3D task reads (`glorys_gs_surface`, `glorys_gs_multidepth`,
`bathy_gs`, `argo_profiles_gs`, and the three `prepare-obs` outputs). `odyssey.yaml` is deliberately
partial: it carries the surface datasets only, and `command=validate` reports the rest as missing
if you try the OSSE task there.

```yaml
  # --- OSSE-3D, Gulf Stream (config/data/osse3d_gs21.yaml) ---
  # truth, produced by scripts/prepare/recipes/glorys_gs_multidepth.yaml then _concat.yaml
  glorys_gs_multidepth:     {path: glorys/glorys_gs_multidepth_2010-2020.zarr, source: CMEMS GLORYS12 reanalysis}
  glorys_gs_surface:        {path: glorys/glorys_gs_multidepth_2010-2020.zarr}   # same store: zos, lat
  # declared, not produced, and not read: the task ships with `bathy: null` (see `ablation=bathy`)
  bathy_gs:                 {path: bathy/bathymetry_gs.nc, source: GEBCO on the GLORYS12 grid}
  # ARGO coverage table, produced by scripts/prepare/argo_profiles.py
  argo_profiles_gs:         {path: argo/argo_profiles_gs.csv}
  # written by `oceanml3d command=prepare-obs` — declare them, do not create them by hand
  pseudo_obs_ssh_gs:        {path: osse/pseudo_obs_ssh_gs.nc}
  pseudo_obs_sst_gs:        {path: osse/pseudo_obs_sst_gs.nc}
  argo_virtual_thetao_gs21: {path: osse/argo_virtual_thetao_gs21.nc}
```

The last three are *outputs* of `prepare-obs`. They must be in the catalog **before** you run it —
that is how the simulator knows where to write. This is why the CLI validates the config twice:
once without the catalog (`check_config(cfg, variables=variables)`), then runs `prepare-obs`, then
once with it.

Two rules, both enforced by `tests/test_config_validation.py`:

* **A site file changes paths, never keys.** A key only one site knows is a typo, or a key the
  others silently lack (`test_a_site_file_does_not_invent_keys_of_its_own`), and a site that claims
  the OSSE task must define every key it resolves.
* **Every key a task reads has a producer** — a recipe in `scripts/prepare/recipes/` whose `output`
  is that path, or `prepare-obs`. A key with nothing filling it validates cleanly and then stops the
  training run on a missing file, which is exactly what `bathy_gs` did
  (`test_every_source_of_the_osse_task_is_produced_by_something`).

---

## 6. The `model` group

| Registered name | Class | What it is |
|---|---|---|
| `nosc_unet` | `models/ocean/nosc/model.py::NOSCUNet` | 2D U-Net trunk (MONAI `DiffusionModelUNet` by default, the historical `UNetNosc` via `ablation=trunk_nosc`), time folded into channels or a 3D stem; the workhorse |
| `fourdvarnet` | `models/ocean/fourdvarnet/model.py::FourDVarNet` | iterative variational solver, learned prior (bilinear AE) + learned gradient step (ConvLSTM) |
| `enkf` | `models/ocean/assimilation/model.py` | EnKF with inflation and Gaspari–Cohn localisation, as a registered model so it exports products |
| `optimal_interpolation` | idem | real OI (length/time scales, obs and background variance) |
| `passthrough` | `models/ocean/baselines/model.py` | copy an input channel onto each target. **The control every learned model must beat** |
| `linear` | idem | one learned affine combination of all inputs per target |
| `climatology` | idem | learned constant per target and per position in the window; ignores inputs |

### 6.1 `nosc_unet` options

```yaml
# @package model
name: nosc_unet
widths: [64, 128, 256, 512, 1024]   # U-Net level widths; 3 levels is the OSSE-3D setting
dropout: 0.1
bilinear: true                      # `trunk: nosc` only
trunk: monai                        # monai | nosc
num_res_blocks: 2                   # `trunk: monai` only
norm_num_groups: null               # `trunk: monai` only; null -> min(32, gcd(widths))
head: single                        # single | grouped | vertical_modes
neck_channels: 64                   # `grouped`/`vertical_modes`: trunk output width
head_hidden: 32
head_layers: 2
mode_specs: null                    # `vertical_modes`: {group: {npz: ..., n_modes: 8}}
attention_levels: []                # e.g. [2] -> self-attention at the coarse level
attention_heads: 4
time_mode: channels                 # channels (NOSC default) | conv3d
temporal_channels: 16               # `conv3d` only
```

#### The two trunks

`trunk: monai` builds `monai.networks.nets.DiffusionModelUNet` (`spatial_dims=2`), wrapped by
`models/ocean/nn/unet_monai.py::MonaiUNet2d`. `trunk: nosc` builds the repository's own residual
U-Net, `models/ocean/nn/unet_nosc.py::UNetNosc`, which was the default until MONAI replaced it —
`ablation=trunk_nosc` is the reproduction path for every run published before the switch, and the
only way to reload their checkpoints.

They are not interchangeable at equal cost. **At identical `widths`, the MONAI trunk has ~12–13×
more parameters** (measured, 50 in / 10 out channels):

| `widths` | `trunk: monai` (`num_res_blocks: 2`) | `num_res_blocks: 1` | `trunk: nosc` |
|---|---|---|---|
| `[64, 128, 256]` (OSSE-3D) | 14.4 M | 10.2 M | 1.10 M |
| `[64, 128, 256, 512, 1024]` (the default) | **223.7 M** | 157.0 M | 17.8 M |
| `[8, 16, 32]` (smoke) | 0.23 M | 0.16 M | 0.02 M |

MONAI stacks `num_res_blocks` blocks per level (one more per level on the way up), adds a mid block
with attention, and never narrows the bottleneck — `UNetNosc` halves it when `bilinear: true`. The
defaults are left at the historical `widths` rather than silently rescaled, so **a nosc-vs-monai
comparison must equalise the budget first** (drop a level, or `num_res_blocks: 1`), and the
full-width default is a 224 M-parameter model: size it against your GPU before launching.

Three behavioural differences, all deliberate:

* `bilinear` has **no MONAI equivalent** (it upsamples by nearest interpolation + convolution). It is
  accepted and ignored, with a one-per-process notice.
* MONAI needs every spatial dimension to halve exactly `len(widths) - 1` times. `MonaiUNet2d` pads
  up to the next multiple and crops back, so odd patch sizes work as they did before.
* MONAI zero-initialises the output convolution and every residual block's second convolution, so a
  freshly built MONAI trunk **outputs exactly zero**. That is the diffusion-model convention, not a
  bug; training proceeds normally from the first step.

The heads matter for 3D targets. With 21 depth levels × 3 variables + SSH, `head: single` asks one
convolution to produce 64 × 11 channels at once; `head: grouped` gives each physical quantity its own
small conv stack on a shared feature map; `head: vertical_modes` predicts EOF coefficients instead of
levels, which is the right inductive bias if the vertical structure is low-rank.

### 6.2 How model keys reach the constructor

`cli.py::build_model` inspects the signature of the model class **and of its base**, then drops any
config key that neither accepts, printing:

```
[oceanml3d] model 'passthrough' ignores config key 'widths' (inherited from the experiment body)
```

This is a normal message when an experiment body carries keys meant for a different model (e.g.
switching to `ablation=gradsolver`). It is also the message you will see if you **mistype a
hyper-parameter** — a typo is silently dropped, not rejected. Read it.

Always injected from the `training` group, never from the `model` group: `loss`, `optimizer`,
`optimizer_kw`, `norm_stats`, `loss_combine`, `loss_group_weights`, `grad_loss_weight`, plus
`variables`, `window` (= `data.patch.time`) and `rec_weight`.

---

## 7. The `training` group

```yaml
# @package training
batch_size: 1
num_workers: 8
loss: mae                   # mae | mse   -- weighted, NaN-aware
loss_combine: flat_sum      # flat_sum | group_mean | uncertainty
loss_group_weights: {}
grad_loss_weight: 0.0       # Sobel gradient MSE per target, for fine scales
save_top_k: 3
moving_patches: false       # random window offsets on the train split
cache: false                # write the stacked array to zarr once under <paths.root>/cache/
stages: null                # null = the model's own `stages`
stage_trainer: {}           # per-stage Trainer overrides, e.g. {prior: {max_epochs: 50}}
optimizer: {name: cosine_adam, lr: 1.0e-4, t_max: ${training.trainer.max_epochs}}
rec_weight:
  kind: triangular          # constant | triangular
  crop: {time: 0, lat: 4, lon: 4}
  kw: {offset: 1}
trainer:
  accelerator: auto
  devices: 1
  max_epochs: 150
  gradient_clip_val: 0.5
  limit_train_batches: 102
  precision: 32
  inference_mode: false
  log_every_n_steps: 10
```

**`loss_combine`** — this is the decision that matters most for a multivariate 3D task:

* `flat_sum` — sum over targets. With 21 temperature levels and one SSH, temperature gets 21× the
  weight. Correct only if that is what you want.
* `group_mean` — mean inside each `group`, then a weighted sum across groups, so each physical
  quantity contributes equally regardless of its number of levels. This is the OSSE-3D reference.
  `validate_config` warns if you ask for it with a single group (no effect).
* `uncertainty` — learned per-target log-variance (Kendall et al. 2018). Supersedes grouping.

**`rec_weight`** is used **twice**: in the loss (border cells weigh 0) and when stitching patches
back into a field at export. `crop` must satisfy `2 × crop[d] < patch[d]` — `validate_config`
enforces it — and the patch overlap must be at least `2 × crop`, which it does **not** check.

**`limit_train_batches: 102`** is inherited from the NOSC reference run (150 epochs × 102 batches).
Set it to `null` to use the full train split.

**`cache: true`** writes the stacked `(channel, time, lat, lon)` array to a zarr store once, then
reopens it lazily: fast random access, bounded RAM. Worth it as soon as you do more than one run on
the same task. Needs the `zarr` extra.

Preset variants: `training=debug` (CPU, 1 epoch, 2 batches) and `training=ddp` (4 GPUs, `ddp`,
`16-mixed`, `sync_batchnorm`).

---

## 8. Commands, in order

```bash
oceanml3d command=list-models
oceanml3d command=show-config  experiment=<xp> paths=<site>   # resolved config + validation report
oceanml3d command=validate     experiment=<xp> paths=<site>   # exit non-zero on any problem
oceanml3d command=prepare-obs  experiment=<xp> paths=<site>   # OSSE only, idempotent
oceanml3d command=eofs         experiment=<xp> paths=<site>   # only for ablation=vertical_modes
oceanml3d command=train        experiment=<xp> paths=<site>   # default command
oceanml3d command=predict      experiment=<xp> ckpt=<path.ckpt>
```

`validate` reports **every** problem at once rather than failing on the first: stride vs patch, crop
vs patch, invalid `loss`/`loss_combine`/`rec_weight.kind`, missing or inverted splits, val/train
leakage, `export.time` outside its split, `norm_stats` length mismatch, negative `depth_index`,
unknown catalog keys, and missing files. Run it before every job; it costs a second and it is the
difference between failing now and failing three hours into a queue.

`train` runs one `trainer.fit` per stage (`training.stages` or the model's own `stages`), calling
`set_stage` in between, then `trainer.test`, then the export if `export.enabled`. Resume with
`ckpt=<path>` — it is passed to the **first** stage only.

---

## 9. Export and the product

```yaml
export:
  enabled: true
  split: test
  time: ["2019-01-01", "2019-12-31"]   # sub-range of the split; null = all
  depth_m: 15                          # informative, written into the manifest
```

`predict_field` runs the model over every patch of the split, then `PatchArray.reconstruct` blends
the overlapping windows with `rec_weight`. `predict_step` returns **denormalised** values, so the
inference layer never needs to know anything about the model.

Output:

```
outputs/<xp>/<run>/product/
├── product.yaml
└── daily/<xp>_YYYY-MM-DD.nc
```

Each target variable is written under its `standard_name`/`units`. **A target without those keys
cannot be scored** — set them in the `variables:` block. Details and the v2 ensemble fields
(`ensemble_size`, `coords.member`, for `enkf` with `return_ensemble: true` or any sampler) are in
`product_format.md`; the format itself is `oceanml3d/inference/product_contract.py`, shared byte for
byte with `oceanml3d-eval` and pinned by SHA-256 on both sides.

Scoring happens in the other repository:

```bash
oceanml3d-eval run -b surface_currents_15m -p outputs/<xp>/<run>/product/product.yaml
```

---

## 10. Worked example A — real data (OSE): 15 m currents

**Task.** Reconstruct AOML drifter velocities at 15 m from L4 altimetry and ERA5 winds. The targets
are *sparse and real*: a daily 1/4° drifter map is mostly NaN, which is exactly why the losses are
NaN-aware and defined on the observed subset.

**Files needed** (`data_preparation.md`, `scripts/prepare/`):

```bash
python scripts/prepare/download_copernicus.py --config scripts/prepare/recipes/duacs_4th.yaml
python scripts/prepare/regrid.py              --config scripts/prepare/recipes/duacs_4th.yaml
python scripts/prepare/download_era5.py       --config scripts/prepare/recipes/era5_daily_4th.yaml
python scripts/prepare/regrid.py              --config scripts/prepare/recipes/era5_daily_4th.yaml
python scripts/prepare/drifters_daily_maps.py --config scripts/prepare/recipes/drifters_aoml_15m_4th.yaml
```

Then point `config/paths/<site>.yaml` at the results.

**Run the control first.** It is one epoch and it sets the bar:

```bash
oceanml3d experiment=baseline_15m_duacs paths=<site>
```

`passthrough` with `source: {u_drifter: ugos, v_drifter: vgos}` predicts the geostrophic input
directly. A learned model that does not beat this has learned nothing.

**Then the reference run:**

```bash
oceanml3d command=validate experiment=nosc_15m_duacs paths=<site>
oceanml3d experiment=nosc_15m_duacs paths=<site> training=ddp
```

11-day window, 560×1440 patches, weighted MAE, triangular time weight with a 4-cell crop,
cosine-annealed Adam at 1e-4, 150 epochs × 102 batches. This reproduces NOSC's
`unet_uv_aoml_15m_10y_11d_bathy_no_sst_mae_duacs_RonanUnet.yaml`. To compare bit-for-bit with the
original, set `data.norm_stats` to the values in that YAML.

**Variants already wired:** `nosc_15m_neurost_sst` swaps DUACS for NeurOST and adds SST through the
`log_grad` transform (the front proxy — `log(|∂T/∂x| + |∂T/∂y|)`, NOSC's `sst_transfo`);
`nosc_00m_duacs` targets undrogued 0 m drifters.

---

## 11. Worked example B — OSSE: 3D multivariate, Gulf Stream

**Task.** GLORYS12 at native 1/12° is the truth over 32–44 °N, 66–54 °W. Inputs: SSH pseudo-obs along
six simulated nadir tracks + the mask channel, cloud-masked SST pseudo-obs + its mask, virtual ARGO
temperature on 21 levels + its mask, and one static (latitude; the bathymetry is set aside —
`ablation=bathy` adds it back once `bathy_gs` exists). Targets: `zos` plus
`thetao`, `uo`, `vo` on 21 GLORYS levels — **64 output variables**.

```bash
# 1. truth + ARGO coverage table
python scripts/prepare/download_copernicus.py --config scripts/prepare/recipes/glorys_gs_multidepth.yaml
python scripts/prepare/regrid.py              --config scripts/prepare/recipes/glorys_gs_multidepth.yaml
python scripts/prepare/argo_profiles.py       --config scripts/prepare/recipes/argo_profiles_gs.yaml

# 2. check the OSSE-3D keys of §5.1 are in config/paths/<site>.yaml (local, datarmor, jeanzay carry them)

# 3. simulate the observing system (writes three NetCDF files)
oceanml3d command=prepare-obs experiment=osse3d_gs21_multivar_unet paths=<site>

# 4. check everything resolves
oceanml3d command=validate experiment=osse3d_gs21_multivar_unet paths=<site>

# 5. reference run
oceanml3d experiment=osse3d_gs21_multivar_unet paths=<site> training=ddp
```

The reference is `loss_combine: group_mean` + `grad_loss_weight: 20.0` with `widths: [64, 128, 256]`
and `dropout: 0.1`. Every ablation is a delta from it:

| `ablation=` | Question it answers |
|---|---|
| `baseline` | the reference itself, stated explicitly |
| `flat_sum_loss` | does equalising physical quantities across depth counts matter? |
| `no_grad_loss` | what does the Sobel fine-scale term buy? |
| `uncertainty` | does learned per-target weighting beat hand-set grouping? |
| `heads` | one conv head per target group vs a single head |
| `attention` | self-attention at the coarse U-Net level |
| `temporal_conv3d` | explicit 3D temporal mixing vs time-as-channels |
| `vertical_modes` | predict 8 EOF coefficients per group instead of 21 levels |
| `gradsolver` | **architecture swap**: iterative 4DVarNet instead of the direct U-Net |
| `bathy` | **data, not architecture**: adds the bathymetry back as a static input (needs `bathy_gs` prepared) |

`vertical_modes` needs its bases first:

```bash
oceanml3d command=eofs experiment=osse3d_gs21_multivar_unet paths=<site>
# writes <paths.root>/eofs/{thetao,uo,vo}_gs21.npz
oceanml3d experiment=osse3d_gs21_multivar_unet ablation=vertical_modes paths=<site>
```

`gradsolver` overrides `/model: fourdvarnet` and sets `obs_map: {zos: ssh_obs}` — the solver needs to
know which input is the observed counterpart of which target. Targets declared `role: both` are
mapped automatically; everything else must be listed.

Two reduced variants exist: `osse3d_gs21_surface_only` (surface targets only) and
`fourdvarnet_ssh_osse` (classic SSH-mapping OSSE, all 3D targets nulled out).

---

## 12. Adding a task or a model

**A new task** = a new `config/data/<task>.yaml` (§4) + the catalog keys (§5) + a
`config/experiment/<xp>.yaml` that overrides `/data` and `/model`. No Python.

**A new model** — see `adding_a_model.md`. Two corrections to that file, which predates the
namespace move:

* built-in models go in **`oceanml3d/models/ocean/<name>/model.py`**, not `oceanml3d/models/<name>/`;
* they are registered in **`oceanml3d/models/ocean/__init__.py`**; `oceanml3d/models/__init__.py` is
  empty.

Third-party models need neither: publish into the `oceanml3d.models` entry-point group and the
registry picks them up at import.

```toml
[project.entry-points."oceanml3d.models"]
my_model = "mypkg.model:MyModel"
```

The group name is `oceanml3d.models` even though the built-ins now live one level down. That string
is a **published contract**; do not change it.

---

## 13. Troubleshooting

| Symptom | Cause |
|---|---|
| `unknown model 'x'. Available: [...]` | typo, or a plugin that failed to import — the registry prints the plugin error separately |
| `[oceanml3d] model 'x' ignores config key 'y'` | normal when an experiment body carries keys for another model; **also what a hyper-parameter typo looks like** |
| `source 'k' is not in the catalog (paths=<site>)` | missing key — §5.1 for the OSSE-3D set |
| `<path> does not exist (run the prepare recipes, or command=prepare-obs)` | key declared, file not produced yet |
| `data.stride.lat > data.patch.lat: gaps in coverage` | stride must be ≤ patch |
| `training.rec_weight.crop.lat removes the whole patch` | `2 × crop ≥ patch` |
| `data.splits.val starts inside the train window: leakage` | fix the dates; do not silence it |
| `export.time [...] is outside the 'test' split` | the split must cover the export range plus the window history |
| `a VariableSet needs at least one target variable` | an override nulled every target |
| `training.loss_combine='group_mean' with a single target group has no effect` | set `group:` on your targets, or use `flat_sum` |
| Blank border stripes in the exported field | patch overlap smaller than `2 × rec_weight.crop` — **not** caught by `validate` |
| Run directory named `null` | no `experiment=` was given; `experiment_name` resolves from the Hydra choice |
| `isinstance` failures around Lightning | you installed `lightning` instead of `pytorch_lightning`; this tree uses `pytorch_lightning` throughout, including the `importorskip` guards |
| `this checkpoint was trained with the NOSC trunk ...` | a pre-MONAI checkpoint: it has no `trunk` hyper-parameter, so it is rebuilt as MONAI. Add `ablation=trunk_nosc` |
| `[oceanml3d] trunk 'monai' ignores 'bilinear'` | informational; `bilinear` has no MONAI equivalent (§6.1) |
| Out of memory right after the switch to MONAI | at equal `widths` the MONAI trunk is ~12–13× larger (§6.1); drop a level or set `num_res_blocks: 1` |

---

## 14. What is not wired yet

Honest limits, so you do not look for them:

* **Depth is not a patch dimension.** 3D targets are flattened into channels via `depth_indices`.
  `PatchArray` is generic over its `DIMS` tuple and adding `"depth"` is a small change, but nobody
  has done it.
* **Two config schemas coexist** (`legacy/conf/schema.py` for `legacy/train.py`, `config_schema.py`
  for the CLI) with disjoint importers. PLAN 3.x.
* **The interior model with sinusoidal depth embedding** is not ported (PLAN 3.7); the SSH spectral
  loss and anomaly correlation are PLAN 3.5.
* **Scoring lives elsewhere.** Metrics beyond the training loss come from `oceanml3d-eval`.
