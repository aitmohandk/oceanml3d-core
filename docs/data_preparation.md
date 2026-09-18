# Data preparation

All recipes are YAML files under `scripts/prepare/recipes/` and run with
`python scripts/prepare/<tool>.py --config <recipe>` (or through `jobs/`, see `jobs/README.md`).
Paths use `${OCEANML3D_RAW}` (downloads) and `${OCEANML3D_DATA}` (catalog root).

| Step | Tool | Output (catalog key) |
|---|---|---|
| CMEMS L4 SSH / geostrophy | `download_copernicus.py` → `regrid.py recipes/duacs_4th.yaml` | `ssh_l4_duacs_4th` |
| ERA5 10 m winds | `download_era5.py` → `regrid.py recipes/era5_daily_4th.yaml` | `era5_daily_4th` |
| Bathymetry | `regrid.py` (GEBCO → 1/4°) | `bathy_4th` |
| AOML / CMEMS drifters | `drifters_daily_maps.py recipes/drifters_aoml_15m_4th.yaml` | `drifters_aoml_15m_4th` |
| SST L4 | `download_copernicus.py` + `regrid.py` | `sst_l4_odyssea_4th` |

## OSSE-3D (Gulf Stream)

| Step | Tool | Output (catalog key) |
|---|---|---|
| GLORYS12 truth, 21 levels | `download_copernicus.py` → `regrid.py recipes/glorys_gs_multidepth.yaml` | `glorys_gs_multidepth` |
| GLORYS12 truth, surface | same recipe, surface selection | `glorys_gs_surface` |
| Bathymetry on the GLORYS grid | `regrid.py` (GEBCO → 1/12°) | `bathy_gs` |
| ARGO coverage table | `argo_profiles.py recipes/argo_profiles_gs.yaml` | `argo_profiles_gs` |

Then the observing system itself, which is **simulated, not downloaded**:

```bash
oceanml3d command=prepare-obs experiment=osse3d_gs21_multivar_unet paths=<site>
```

It writes `pseudo_obs_ssh_gs`, `pseudo_obs_sst_gs` and `argo_virtual_thetao_gs21` — simulated nadir
altimeter tracks, cloud-masked SST, and virtual ARGO floats keeping the real floats' geometry. Those
three keys must already be **declared** in `config/paths/<site>.yaml` before you run it: that is how
the simulator knows where to write, and it is why the CLI validates the configuration twice.

The step is idempotent and rank-safe: under DDP one process simulates and the others wait, and the
writes are atomic, so a waiting rank never reads a half-written file.

## Adding a site

Copy `config/paths/local.yaml` and change the **paths**, not the keys — a site file that invents a
key is either a typo or a key the other sites silently lack, and
`test_a_site_file_does_not_invent_keys_of_its_own` says so. `config/paths/local.yaml` is the
reference list of all sixteen keys the gridded configs use.
