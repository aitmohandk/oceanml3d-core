# Migrating NOSC into oceanml3d-core

NOSC (`code/multivar_drifter_`, ~4.6 k lines of Python + ~190 notebooks/scripts) maps to
the framework as follows.

| NOSC | oceanml3d-core | Notes |
|---|---|---|
| `config/xp/*.yaml` → `multivar:` dict (`var_path`, `var_name`, `input_arch`, `output_arch`, `broadcast_time`) | `config/data/surface_currents_15m.yaml` → `variables:` (`source`, `var_name`, `role`) | `prior_input`→`input`, `full_output`→`target`, `broadcast_time: True`→`static`. Paths → catalog keys. |
| `config/xp/*.yaml` → `domain`, `datamodule.domains`, `xrds_kw.patch_dims/strides` | `data.domain`, `data.splits`, `data.patch`, `data.stride` | identical semantics |
| `datamodule.norm_stats` (hard-coded list) | `data.norm_stats: null` → computed on train, saved to `norm_stats.json` | reproducible |
| `contrib/data_loading/data.py::open_multivar_datasets` | `oceanml3d/data/open.py::open_variable_set` | lazy, no prints, transforms registry |
| `contrib/moving_patches/*`, `src/data.py::XrDataset` | `oceanml3d/data/patches.py` | numpy, tested; moving-patch offset = Phase 2 |
| `contrib/multivar/multivar_data.py::MultivarDataModule` | `oceanml3d/data/datamodule.py::OceanDataModule` | |
| `contrib/multivar/multivar_utils.py::MultivarBatchSelector`, `get_multivar_*_dims_*` | `VariableSet.inputs/targets`, `BaseOceanModel.inputs()/targets()` | no more channel arithmetic in configs |
| `get_multivar_mapping_wei`, `src.utils.get_constant_crop` | `oceanml3d/training/weights.py` | |
| `contrib/multivar/multivar_models_unet_mae.py::MultivarUNet_mae` | `oceanml3d/models/ocean/nosc/model.py::NOSCUNet` | `weighted_mae` → `training/losses.py` |
| `contrib/multivar/parts_drop.py`, `contrib/4dvarnet_latent/unet.py::UNetModel` | `oceanml3d/models/ocean/nn/unet_nosc.py` | one residual U-Net, configurable widths; no longer the default trunk (MONAI's `DiffusionModelUNet` is), kept as `ablation=trunk_nosc` |
| `cosanneal_lr_adam_unet` | `training/optim.py::cosine_adam` | |
| `src/versioning_cb.py` | `training/callbacks.py::VersioningCallback` | also dumps config + norm stats |
| `src/train.py::base_training(only_rec=…)`, `concat_rec*.py`, `launch_rec*.sh` | `oceanml3d command=predict` → `inference/predict.py` + `export.py` | one command, one output format |
| `metric/dictionary/<xp>.json` (product descriptors) | `product/product.yaml` written automatically | see `product_format.md` |
| `metric/dictionary/region_*.json` | `oceanml3d-eval/regions/*.json` | |
| `run_rmse_GL.py`, `run_lagrangian_GL.py`, `run_spectrum_GL.py`, `launch_*.sh` | `oceanml3d-eval run --benchmark surface_currents_15m --product product.yaml` | |
| `env/velocity_metrics/` (OceanDataLab, LGPL) | `oceanml3d-eval/third_party/velocity_metrics` (vendored, optional backend) | |
| `code/download_data_/*` | `scripts/prepare/download_{copernicus,era5}.py` + recipes | |
| `code/process_data/interpolation/*`, `glorys/*`, `era5/*`, `sst/*` | `scripts/prepare/regrid.py` + `recipes/*.yaml` | |
| `code/process_data/make_train_dataset_drifters/*`, `drifters/*` | `scripts/prepare/drifters_daily_maps.py` | |
| `code/plot_article/*`, `metric/*.ipynb`, `finescale/*.ipynb` | `oceanml3d-eval` report + notebooks kept in a `notebooks/` archive of NOSC | not migrated |
| `config/xp/old/*`, `old_launch/*`, `*.log`, `.nc`, `.png` | dropped | history stays in the NOSC repo |
| `contrib/{lorenz63, ose2osse, ose_pipeline, three_d, forecast_plus, …}` | not migrated | unrelated 4dvarnet-starter contribs; port on demand as new models |

## Reproducing the reference run

```bash
oceanml3d experiment=nosc_15m_duacs paths=odyssey
```
corresponds to `config/xp/unet_uv_aoml_15m_10y_11d_bathy_no_sst_mae_duacs_RonanUnet.yaml`
(11-day window, 560×1440 patches, MAE, cosine-annealed Adam 1e-4, 150 epochs × 102 batches,
triangular time weight with 4-cell crop). To compare bit-for-bit, set `data.norm_stats`
to the values of the original YAML.


## Branch `first_implementation` (OSSE-3D) additions

| NOSC (first_implementation) | oceanml3d-core | Notes |
|---|---|---|
| `contrib/synthetic_obs/{missions,orbits,sampling}.py` | `oceanml3d/obs/{missions,orbits,sampling}.py` | ported verbatim (numpy), tests ported |
| `contrib/synthetic_obs/build_masks.py` (pickle list) + `make_pseudo_obs.py` | `oceanml3d/obs/pseudo_obs.py` → NetCDF `obs_mask` + `<var>_obs` | dated NetCDF only (no index-aligned pickle); real L3 masks via `real_mask` |
| `contrib/synthetic_obs/build_masks_swot_official.py` | not ported | wire the CNES/JPL simulator as `real_mask` producer if needed |
| `contrib/argo/{download,qc,vertical_interp,run_pipeline}.py` | `oceanml3d/obs/argo.py` + `scripts/prepare/argo_profiles.py` | argopy chunked or local GDAC → QC → coverage table (tests ported) |
| `contrib/argo/virtual.py`, `build_argo_dataset.py` | `oceanml3d/obs/argo_virtual.py` | one NetCDF, variables `<var>_d<ii>` |
| `config/depths/gs21_indices.yaml` + `contrib/data_loading/depth_fragments.py` + `config/vars/*_gs21.yaml` | `depth_indices:` on a `VariableSpec` (`VariableSet.from_config` expansion) | no code generation |
| `depth_index` key of multivar entries | `VariableSpec.depth_index` (exact `isel`) | |
| `head_group` tags | `VariableSpec.group` (auto = variable name for `depth_indices`) | drives loss grouping and heads |
| `contrib/multivar/loss_grouping.py` | `oceanml3d/training/loss_grouping.py` | `training.loss_combine: flat_sum|group_mean`, `loss_group_weights` |
| `grad_loss_weight` (kornia sobel) | `training.grad_loss_weight`, `losses.sobel` (pure torch) | |
| `multivar_models_unet_uncertainty.py` | `training.loss_combine: uncertainty` (`BaseOceanModel.log_vars`) | optimiser covers it automatically (`self.parameters()`) |
| `multivar_models_unet_heads.py` | `model.head: grouped` (`models/ocean/nn/heads.GroupedHeads`) | |
| `contrib/multivar/vertical_modes.py` | `training/vertical_modes.py` (EOF) + `models/ocean/nn/heads.VerticalModesHead`, `oceanml3d command=eofs` | |
| `attention_resolutions` of `UNetModel` | `model.attention_levels` (MONAI `SpatialAttentionBlock`, or `nn/unet_nosc.SelfAttention2d`) | |
| `config/ablation/*.yaml` | `config/ablation/*.yaml` (same names) | deltas from the experiment body |
| `config/xp/osse3d_gs21_{multivar_unet,surface_only,cpu_smoke}.yaml` | `config/experiment/osse3d_gs21_multivar_unet`, `osse3d_gs21_surface_only`, `osse3d_smoke` + `config/data/osse3d_gs21*.yaml` | |
| `paths:` block with `NOSC_DATA_ROOT` | `config/paths/<site>.yaml` + `OCEANML3D_DATA` | |
| `device_utils.py`, CPU fixes | not needed: no hard-coded `.cuda()` in oceanml3d (`accelerator: auto`) | |
| `metric/eulerian/depth_profile_metrics.py` | `oceanml3d-eval`: `gridded_rmse` + `spectral_score(isotropic=true)` per level, `oceanml3d-eval report --depth-profile nrmse` | |
| `tests/test_synthetic_obs.py`, `test_loss_and_modes.py` | `tests/test_obs_simulation.py`, `test_loss_grouping_and_modes.py` | ported |
| `TODO.md` §1 (real masks, mission dropout, mission windows, obs-noise resampling) | `real_mask`, `historical: true`, `per_mission` + `augment.mission_dropout`, `augment.obs_noise` | |
| `TODO.md` §2 (cloud-masked SST, ARGO noise, ARGO presence channels) | `prepare.pseudo_obs.sst.clouds`, `virtual_argo.noise_std`, `<var>_mask` channels | |
| `TODO.md` §3 (GradSolver, temporal variant) | `models/ocean/fourdvarnet` (`ablation=gradsolver`), `time_mode: conv3d` | |
| `TODO.md` §0 (RAM of eager load) | `training.cache: true` (zarr) | |
| `CHANGES_OSSE.md`, `TODO.md` | `PLAN.md` | |
