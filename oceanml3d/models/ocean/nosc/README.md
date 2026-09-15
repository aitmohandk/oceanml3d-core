# nosc_unet — Neural Ocean Surface Currents

Direct U-Net from a temporal window of gridded surface inputs to drifter-derived currents at
0 m / 15 m. Original code: https://github.com/aitmohandk/NOSC.

The trunk is MONAI's `DiffusionModelUNet` by default (`models/ocean/nn/unet_monai.py`); NOSC's own
residual U-Net (`models/ocean/nn/unet_nosc.py::UNetNosc`) is kept as `ablation=trunk_nosc` and is
what every run published before the switch used. They differ in size by ~12x at equal `widths` —
see `docs/gridded_models.md` §6.1.

| | |
|---|---|
| Inputs | `ssh`, `ugos`, `vgos` (DUACS or NeurOST L4), `u10`, `v10` (ERA5), `lat`, `bathy`, optional `sst` (`log_grad`) |
| Targets | `u_drifter`, `v_drifter` (daily binned drifter velocities, sparse) |
| Window | 11 days (channels = vars x days) |
| Loss | weighted MAE on finite target cells, triangular time weight |
| Config | `config/model/nosc_unet.yaml`, experiments `config/experiment/nosc_*.yaml` |
| Export | daily NetCDF `u`,`v` + `product.yaml` for `oceanml3d-eval` benchmark `surface_currents_15m` |

Migration notes from the NOSC repo: `docs/migration_nosc.md`.
