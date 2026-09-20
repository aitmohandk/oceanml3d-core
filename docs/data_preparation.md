# Data preparation

Which recipe produces which catalog key. The steps themselves are run through `jobs/prepare.sh`
(`glorys <year>`, `concat`, `argo`, `obs`), which is what the PBS and Slurm wrappers call; the
runbooks [`platforms/datarmor.md`](platforms/datarmor.md) and
[`platforms/jeanzay.md`](platforms/jeanzay.md) walk through it step by step, and
[`pipeline_3d.md`](pipeline_3d.md) explains the task end to end.

All recipes are YAML files under `scripts/prepare/recipes/` and can also be run directly with
`python scripts/prepare/<tool>.py --config <recipe>`. Paths use `${OCEANML3D_RAW}` (downloads) and
`${OCEANML3D_DATA}` (catalog root). A recipe named `<name>.<site>.yaml` is preferred over
`<name>.yaml` when that site is selected — that is how Datarmor reads its local mirrors instead of
downloading.

Three things decide where the files land and what they are:

* **`OCEANML3D_TARGET_RES`** — the grid step in degrees. Set, the data root becomes
  `$OCEANML3D_DATA/res<step>` for every step, training included, so two resolutions never share a
  directory ([`pipeline_3d.md` §3.1a](pipeline_3d.md)). Unset: the native grid.
* **Zarr for everything intermediate** — the per-year GLORYS stores and the merged one. NetCDF is
  only what is read from outside (the GLORYS files, the ARGO GDAC) and what is exported (the
  product, the contract with `oceanml3d-eval`).
* **The catalog** — `config/paths/<site>.yaml` maps each key below to a path under its root.

| Step | Tool | Output (catalog key) |
|---|---|---|
| CMEMS L4 SSH / geostrophy | `download_copernicus.py` → `regrid.py recipes/duacs_4th.yaml` | `ssh_l4_duacs_4th` |
| ERA5 10 m winds | `download_era5.py` → `regrid.py recipes/era5_daily_4th.yaml` | `era5_daily_4th` |
| Bathymetry | `regrid.py` (GEBCO → 1/4°) | `bathy_4th` |
| AOML / CMEMS drifters | `drifters_daily_maps.py recipes/drifters_aoml_15m_4th.yaml` | `drifters_aoml_15m_4th` |
| SST L4 | `download_copernicus.py` + `regrid.py` | `sst_l4_odyssea_4th` |

## OSSE-3D (Gulf Stream)

| Step | How | Output (catalog key) |
|---|---|---|
| GLORYS12 truth, one year | `jobs/prepare.sh glorys <year>` → `by_year/glorys_gs_multidepth_<year>.zarr` | — |
| GLORYS12 truth, merged | `jobs/prepare.sh concat` | `glorys_gs_multidepth` |
| GLORYS12 truth, surface | the same Zarr store (`zos` has no depth axis, `lat` is a coordinate) | `glorys_gs_surface` |
| Bathymetry on the GLORYS grid | **not prepared, and not needed**: the task ships with `bathy: null`. When you want it, GEBCO with `regrid.py` and `reference:` pointing at the prepared truth, then `ablation=bathy` | `bathy_gs` |
| ARGO coverage table | `jobs/prepare.sh argo` — from the site's GDAC mirror when it has one (`source: gdac`, Datarmor's `/home/ref-argo/gdac`, read through its global index), else downloaded with `argopy` | `argo_profiles_gs` |

On Datarmor neither GLORYS nor ARGO is downloaded: both are mirrored under `/home/ref-*`. The
per-year step reads the mirror directly, cuts the Gulf Stream box in each file before any read, and
refuses an output above `max_gb` — the guard that came out of a 1949 GB run.

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
