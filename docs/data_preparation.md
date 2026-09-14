# Data preparation

All recipes are YAML files under `scripts/prepare/recipes/` and run with
`python scripts/prepare/<tool>.py --config <recipe>` (or `sbatch batch/prepare_data.sbatch ...`).
Paths use `${OCEANML3D_RAW}` (downloads) and `${OCEANML3D_DATA}` (catalog root).

| Step | Tool | Output (catalog key) |
|---|---|---|
| CMEMS L4 SSH / geostrophy | `download_copernicus.py` → `regrid.py recipes/duacs_4th.yaml` | `ssh_l4_duacs_4th` |
| ERA5 10 m winds | `download_era5.py` → `regrid.py recipes/era5_daily_4th.yaml` | `era5_daily_4th` |
| Bathymetry | `regrid.py` (GEBCO → 1/4°) | `bathy_4th` |
| AOML / CMEMS drifters | `drifters_daily_maps.py recipes/drifters_aoml_15m_4th.yaml` | `drifters_aoml_15m_4th` |
| SST L4 | `download_copernicus.py` + `regrid.py` | `sst_l4_odyssea_4th` |

Add the resulting paths to `config/paths/<site>.yaml`.
