# The 3D multivariate model, end to end

How to prepare the data and run a training or an inference for the **3D multivariate gridded model**
— latitude, longitude *and* depth — on Datarmor and on Jean Zay.

This is the descendant of NOSC's `osse3d_gs_multivar_unet`. It reads a simulated observing system
(altimeter tracks, cloud-masked SST, virtual ARGO floats) and reconstructs temperature and the two
horizontal current components on 21 depth levels, plus sea surface height. Sixty-four output
variables.

**This page is platform-independent: what the model is, what the data means, which command does
what.** The step-by-step runbooks for a given centre — reserving a node, the exact paths, the local
traps — are separate, because the two kinds of knowledge go stale on different schedules:

* **`docs/platforms/datarmor.md`** — Ifremer, PBS Pro, V100 32 GB
* **`docs/platforms/jeanzay.md`** — IDRIS, Slurm, V100 / A100 / H100

Companions: [`tutorial.md`](tutorial.md) (the same pipeline as a step-by-step manual, with the check
after each step), `docs/gridded_models.md` (configuration reference — the `variables:` block, the
catalog, patching, losses), `jobs/README.md` (how the scheduler layer works), `docs/data_preparation.md`
(every recipe, including the surface-current tasks this page does not cover).

---

## 1. What the task actually is

`config/data/osse3d_gs21.yaml`, Gulf Stream box, 32–44 °N × 66–54 °W, GLORYS12 at 1/12°.

| | Variables | Source |
|---|---|---|
| **Inputs** | `ssh_obs` + `obs_mask` | simulated nadir tracks over GLORYS SSH |
| | `sst_in` + `sst_mask` | cloud-masked SST |
| | `argo_thetao` × 21 + `argo_mask` × 21 | virtual ARGO on the real floats' geometry |
| | `lat` | static (`bathy` is set aside for now — `ablation=bathy` adds it back) |
| **Targets** | `zos` | GLORYS SSH |
| | `thetao`, `uo`, `vo` × 21 levels | GLORYS truth |

The 21 levels are `depth_indices: [0, 2, 4, 6, 8, 10, 11, …, 25]` — **positions on the stored file's
depth axis**, not depths in metres. That distinction is the single most consequential thing on this
page: change what the preparation stores and every index in the data config silently addresses a
different depth. See §3.3.

An OSSE, so the truth is known everywhere and the observing system is simulated. That is what makes
it possible to score a reconstruction at 150 m, where no real observation exists.

---

## 2. Prerequisites

* A clone of this repository (the CLI runs from a clone — Hydra resolves `config_path` relative to
  the file carrying `@hydra.main`, so an installed wheel alone is not enough).
* An environment: the container is the recommended route on both centres
  (`apptainer build oceanml3d.sif container/oceanml3d.def`), `environment.yml` otherwise.
* CMEMS credentials for the downloads: `copernicusmarine login`, once, on a node **with network**.
* `config/paths/<site>.yaml` and `jobs/env/<site>.sh` for your centre. `local`, `datarmor` and
  `jeanzay` ship complete; check `OCEANML3D_DATA` in the site script points where you want the
  data before the first preparation job.

### The one thing to check before anything else

```bash
jobs/run.sh --site <site> command=validate experiment=osse3d_gs21_multivar_unet
```

It reports every problem at once: missing catalog keys, missing files, splits that leak, a crop
larger than the patch overlap. It costs a second. A queued job that fails on a typo costs a day.

---

## 3. Preparing the data

Three stages, in order. The first two download and reduce; the third simulates.

### 3.1 GLORYS truth

```bash
# 1. download — needs network, so a pre/post node on Jean Zay, a login node on Datarmor
python scripts/prepare/download_copernicus.py --config scripts/prepare/recipes/glorys_gs_multidepth.yaml

# 2. subset and store — compute node, this is the expensive one
python scripts/prepare/regrid.py --config scripts/prepare/recipes/glorys_gs_multidepth.yaml
```

**Do it one year at a time and merge.** Eleven years in one job is what kills walltime. Each year
goes to `by_year/glorys_gs_multidepth_<year>.zarr`, then the merge writes the store the catalog
reads. `regrid.py` is idempotent — a completed output is skipped — so a job killed on walltime is
resumed by resubmitting, not restarted. Outputs are written under `.tmp.<name>` and renamed at the
end, so a killed job never leaves a half-written store that the next run would skip.

**Every intermediate is Zarr**; NetCDF is only what is read from outside (the GLORYS files) and what
is exported (the product). The merge is the same tool with a glob, and since both ends are Zarr it
reads with threads:

```yaml
input: ${OCEANML3D_DATA}/by_year/glorys_gs_multidepth_*.zarr
output: ${OCEANML3D_DATA}/glorys/glorys_gs_multidepth_${YEAR}-${NEXT}.zarr
variables: {thetao: thetao, uo: uo, vo: vo, zos: zos}
keep_depth: true
method: none
zarr_chunks: {time: 32}
parallel: true
dask_scheduler: threads
```

### 3.1a Choosing the resolution

GLORYS is 1/12°, and that is what you get by default. To work coarser — NOSC's `--target-res` /
`NOSC_TARGET_RES` — set one variable for the whole preparation:

```bash
jobs/prepare.sh --site datarmor --res 0.25 glorys 2010        # or OCEANML3D_TARGET_RES=0.25
qsub -v OCEANML3D_SITE=datarmor,OCEANML3D_TARGET_RES=0.25 jobs/pbs/prepare_glorys.pbs

# or once for every job, since a batch job does not inherit your shell's environment
# (the file is yours to create; nothing creates it -- see jobs/README.md):
mkdir -p ~/.config/oceanml3d
echo 'export OCEANML3D_TARGET_RES="${OCEANML3D_TARGET_RES:-0.25}"' > ~/.config/oceanml3d/env.sh
```

**The data root follows the resolution.** With `OCEANML3D_TARGET_RES` set, every step — per-year
GLORYS, merge, ARGO, observation simulation, training (`jobs/run.sh --res` too) — uses
`$OCEANML3D_DATA/res<step>` (`jobs/env/_lib.sh`, `resolve_data_dir`). Native and coarsened products
never share a directory, and no step can look in the wrong one because a path was typed differently.

The GLORYS recipes carry `resolution: ${OCEANML3D_TARGET_RES}`; unset, it means native. When set,
each file is cut to the domain (plus one cell of margin) and interpolated bilinearly onto a regular
grid **anchored on the domain bounds** — `arange(lo, hi + res/2, res)`, 49 × 49 for the Gulf Stream
box at 0.25°. The grid depends only on the domain and the step, never on the source, so every product
prepared with the same pair lands on exactly the same grid; `data/open.py` refuses one that does not.

What to keep in mind:

- **One resolution per data directory**, derived as above. File names do not carry it; the value is
  also written in each output's attributes, and `regrid.py` refuses to skip — or to merge — files at
  another resolution instead of silently mixing them. A merge that finds nothing but sees the files
  under a sibling `res*/` says so.
- **The patch follows by itself.** `osse3d_gs21` declares `patch`/`stride` `auto` in lat/lon: they
  are fitted to the grid on disk at run time — 144 / 136 at 1/12° (145 cells), 48 / 40 at 0.25°
  (49 cells). Nothing to pass on the command line ([gridded_models.md §4.4](gridded_models.md)).
- **Everything downstream follows the truth grid**: the pseudo-obs and virtual ARGO are simulated on
  it. A dataset that does not come from the truth — the bathymetry, if you add it back — is the
  exception: regrid it onto the prepared truth with `reference:`.
- Bilinear from 1/12° to 1/4° is what NOSC did; it subsamples rather than averages. For
  an area mean, `method: conservative` needs `xesmf` in the image.

### 3.2 ARGO coverage table

```bash
jobs/prepare.sh --site <site> argo
```

Reads real ARGO profiles — from a local GDAC mirror when the site has one
(`argo_profiles_gs.gdac.yaml`, picked whenever `ARGO_GDAC` is set, `source: gdac`: Datarmor's `/home/ref-argo/gdac`, read through its
global index so only the floats in the box are opened), otherwise downloaded with `argopy`, which
needs outbound network. Then it applies QC on the standard flags, interpolates vertically, and writes
the **coverage table**: where and when a float was, and how deep it reached. The values are thrown
away — §3.3 replaces them with the truth. That is what makes the ARGO input *virtual*: real
geometry, simulated values.

**This is the step that gets killed on walltime**, because it opens a few hundred float files off a
shared, read-only mirror and that is slow in a way no amount of code can fix. Three things make it
survivable, all on by default in `argo_profiles_gs.gdac.yaml`:

| | What it does |
|---|---|
| `cache_dir:` | one small file per float. A re-run skips what is done, so nothing is ever read twice. Delete the directory to force a full rebuild. |
| `time_budget_s:` | stop cleanly before the scheduler kills the job — cache written, exit code **75**. `jobs/pbs/prepare_argo.pbs` then resubmits itself, up to `OCEANML3D_MAX_ATTEMPTS` (4), each attempt continuing where the last stopped. |
| the log | every stage is announced before it runs and timed after: Python started, imports done, the mirror and its top level, the index and its size, one line per chunk of index read, the first three files by name, then a rate and an estimate, plus any single file that took more than 20 s. |

Read the log top-down when something goes wrong: the last line it reached *is* the diagnosis.
Nothing after `python started` means the container or the imports; nothing after the index lines
means the mirror is not delivering; a rate of seconds per file means the filesystem, not the code.

**A count is not a check.** The run ends with what the table actually contains, because 8 640
profiles spread over a decade and 8 640 concentrated in two years are the same number and only one
of them is usable:

```
[argo] table: 8640 profiles, 217 floats, 2010-01-03 .. 2020-01-10
[argo] per year: 2010 812  2011 905  ...  2019 1034  2020 26
[argo] reaching each level: d00 99%  d02 99%  ...  d25 96%
```

A year inside the window with almost nothing in it, or a level most profiles never reach, is called
out as a warning — the second matters as much as the first, since a level no float reaches means the
model reconstructs that depth from no observation at all. The same summary runs on a table that
already exists, without re-reading the GDAC:

```bash
python scripts/prepare/argo_profiles.py --summary $OCEANML3D_DATA/argo/argo_profiles_gs.csv \
    --time 2010-01-01 2020-01-11
```

Bathymetry is **not** part of this step. `config/data/osse3d_gs21.yaml` ships with `bathy: null`:
the first model to validate is the simple one, and nothing here produces `bathy_gs`. When it is
wanted, it is GEBCO regridded onto the prepared truth, once, with `regrid.py` and `reference:`
pointing at the truth; `oceanml3d … ablation=bathy` then declares it again as a static input, and
nothing else changes.

### 3.3 The observing system, simulated

```bash
jobs/run.sh --site <site> command=prepare-obs experiment=osse3d_gs21_multivar_unet
```

Writes three files: `pseudo_obs_ssh_gs`, `pseudo_obs_sst_gs`, `argo_virtual_thetao_gs21`.

**Those three keys must already be in `config/paths/<site>.yaml` before you run this.** They are
outputs, but the simulator learns where to write from the catalog. This is why the CLI validates
twice — once without the catalog, then `prepare-obs`, then once with it — and why
`validate_config` checks declaration but not existence for them.

Idempotent and rank-safe: under DDP one process simulates while the others wait, and the writes go
through a temporary name plus `os.replace`, so a waiting rank never reads a half-written file.

**Idempotent does not mean "reuse whatever is there".** An existing output is checked before it is
skipped: it must be on the truth's time axis and grid, and the virtual ARGO file must be newer than
the coverage table it was simulated from. Otherwise it is rebuilt, and the log says why
(`rebuilding …: its time axis is …, the truth's is …`). The log always says which of the two
happened — `virtual ARGO: reusing …` or `virtual ARGO: <path>` — because the failure this guards
against was silent: a virtual ARGO file with no date in common with the truth was reused, and a
training run went ahead on an ARGO input that was zero everywhere. Should such a file still reach
training, `open_variable_set` refuses it by name instead of filling 42 channels with zeros.

What it produces, and why each part is not a toy:

* **Altimeter tracks** — a closed-form repeat ground-track model, six real missions (Jason-3,
  Sentinel-6, Sentinel-3A/B, SARAL, HY-2B) parameterised by inclination, cycle length and orbits per
  cycle. `orbits_per_cycle` is full revolutions, **not passes**: handbooks quote passes, which are
  half-revolutions, and mixing the conventions doubles the simulated sampling density and makes
  every downstream result quietly optimistic. `validate_missions()` runs at import to forbid it.
* **Cloud-masked SST** — synthetic cloud fields, or a real dated L3 mask through `real_mask`.
* **Virtual ARGO** — the coverage table from §3.2 with GLORYS values sampled at those positions and
  depths, binned onto the grid, one variable per depth index.

#### The depth-index coupling, stated plainly

`depth_index: 7` means *the eighth level of the stored file*. The chain is:

```
download depth: [0, 200]  →  regrid keeps N levels  →  config's depth_index addresses 0…N-1
```

Change the download range, or add `depth_indices` to the recipe, and every `depth_index` in
`config/data/osse3d_gs21.yaml` points somewhere else — with no error, because the indices are still
valid. The safe habit: **store every level the download produced and let the task select.** If you
must reduce at preparation time, change the config's indices in the same commit.

Two guards now exist: a recipe with `depth_indices` and `keep_depth: false` raises rather than
silently producing a surface file, and a variable with `depth_index` pointed at a file with no depth
axis raises (it used to pass for index 0, which is falsy).

---

## 4. Training

```bash
# Datarmor (PBS Pro)
qsub -v OCEANML3D_SITE=datarmor,OCEANML3D_ARGS="experiment=osse3d_gs21_multivar_unet" jobs/pbs/train.pbs

# Jean Zay (Slurm)
sbatch --export=ALL,OCEANML3D_SITE=jeanzay jobs/slurm/train.sbatch experiment=osse3d_gs21_multivar_unet
```

Outputs land in `outputs/<experiment>/<timestamp>/`: checkpoints, the resolved config, the git hash,
and `norm_stats.json`.

**Following it.** Validation runs on 2018 every epoch. Watch `val/nrmse` (it selects the
checkpoint) and the physical RMSEs `val/rmse_temperature` (°C), `val/rmse_currents_u|v` (m/s),
`val/rmse_ssh` (m), all on `eval_domain`; `val/rmse_thetao_dNN` gives the error per level. They are in
`metrics.csv` and in TensorBoard (`tensorboard --logdir outputs/<experiment>`). At the end the best
checkpoint is scored on the test split with the same metrics (`test/…`). Details and why not
`val/loss`: [gridded_models.md §4.3a](gridded_models.md).

### Precision, which the hardware decides

| Site | GPU | `training.trainer.precision` |
|---|---|---|
| Datarmor | Tesla V100 32 GB (sm_70) | `16-mixed` — **fp16 only, bf16 does not exist on Volta** |
| Jean Zay V100 | V100 (sm_70) | `16-mixed` |
| Jean Zay A100/H100 | sm_80 / sm_90 | `bf16-mixed` |
| Odyssey | RTX 8000 (sm_75) | `16-mixed` |

`jobs/env/<site>.sh` sets the right default; override for a different partition.

### Memory

`osse3d_gs21` patches are 144 × 144, which is comfortable on any of these cards even with the
default MONAI trunk at ~224 M parameters. (The surface-current task at 560 × 1440 is the tight one —
see `jobs/README.md`.) Never put level 0 in `model.attention_levels`: attention at full resolution is
not a memory problem you tune around.

### The ablations

Each is a delta on the reference run, so they are directly comparable:

```bash
OCEANML3D_ARGS="experiment=osse3d_gs21_multivar_unet ablation=vertical_modes"
```

| `ablation=` | Question |
|---|---|
| `flat_sum_loss` | does equalising quantities across depth counts matter? (21 temperature levels versus one SSH) |
| `no_grad_loss` | what does the Sobel fine-scale term buy? |
| `uncertainty` | does learned per-target weighting beat hand-set grouping? |
| `heads` | one conv head per target group versus a single head |
| `vertical_modes` | predict 8 EOF coefficients per group instead of 21 levels |
| `attention` | self-attention at the coarse level |
| `temporal_conv3d` | explicit 3D temporal mixing versus time-as-channels |
| `gradsolver` | iterative 4DVarNet instead of the direct U-Net |
| `trunk_nosc` | NOSC's own residual U-Net instead of the MONAI trunk |
| `bathy` | adds the bathymetry back as a static input (needs `bathy_gs` prepared: §3.2) |

`vertical_modes` needs its bases first:

```bash
jobs/run.sh --site <site> command=eofs experiment=osse3d_gs21_multivar_unet
```

### Multi-GPU

Add `training=ddp` and raise the resource request. Lightning reads Slurm's variables itself — do not
also wrap the command in `srun`, or every rank relaunches the script. Two things are already handled:
`prepare-obs` runs on one rank while the others wait, and the export runs on rank 0 with a
single-device Trainer, because `trainer.predict` shards the dataloader and a field stitched from one
rank's share is wrong without looking wrong.

---

## 5. Inference

```bash
jobs/run.sh --site <site> command=predict \
    experiment=osse3d_gs21_multivar_unet \
    ckpt=outputs/osse3d_gs21_multivar_unet/<run>/checkpoints/<best>.ckpt
```

Produces `product/product.yaml` and one NetCDF per day under `product/daily/`, CF variables on the
lat/lon grid.

**The normalisation statistics travel inside the checkpoint**, along with the channel layout. This
matters: `predict_step` denormalises, those statistics come from the *train* split, and a prediction
run that recomputed them from the current config would denormalise with numbers the model never saw
— producing a finite, plausible, wrong product. A checkpoint from a different channel layout is
refused; one written before this was added warns that it is falling back.

Scoring happens in the companion repository:

```bash
oceanml3d-eval run -b osse3d_gs21 -p outputs/<xp>/<run>/product/product.yaml
```

---

## 6. Site specifics

Not repeated here. Each centre has its own runbook, and they differ in more than syntax — storage
purges, which nodes have network, how a container is allowed to run:

| | `docs/platforms/datarmor.md` | `docs/platforms/jeanzay.md` |
|---|---|---|
| Scheduler | PBS Pro (`qsub`) | Slurm (`sbatch`, `srun`) |
| GPU | Tesla V100 32 GB | V100 / A100 / H100 |
| Precision | `16-mixed` | `bf16-mixed` on A100/H100, `16-mixed` on V100 |
| Downloads run on | queue `ftp` | `--partition=prepost` |
| Container | runs from `$DATAWORK` directly | registered with `idrcontmgr`, runs from `$SINGULARITY_ALLOWED_DIR` |
| Data live on | `$SCRATCH`, master copy on `$DATAWORK` | `$SCRATCH`, archive on `$STORE` |
| Purge | `$SCRATCH` after 10 days | `$SCRATCH` after 30 days without access |
| The quota that bites | volume | **inodes** (500 000 on `$WORK`) |

Adding a third centre means one file in `docs/platforms/`, one in `jobs/env/`, and one in
`config/paths/` — nothing in the code.

## 7. Known pitfalls

Inherited from NOSC's `GUIDE_UTILISATION.md` §8, re-verified against this code.

* **`depth_index` is a position, not a depth.** §3.3. The one that bites hardest, because it fails
  silently.
* **`argo_thetao` and `thetao` are different targets**, not one variable duplicated: sparse in-situ
  input versus dense GLORYS truth. They carry different `group`s for that reason.
* **`sst_in` is not raw temperature.** With `transform: log_grad` it is `log(|∂T/∂x| + |∂T/∂y|)`, a
  front proxy (NOSC's `sst_transfo`). For an actual temperature target, use a variable without the
  transform.
* **Group order.** Targets sharing a `group` should be contiguous in the `variables:` block; grouped
  and vertical-mode heads slice by group.
* **A GPU is required**, even for short tests — except the synthetic smoke runs
  (`experiment=osse3d_smoke training=debug`), which are CPU and exist for exactly that.
* **Paths in any inherited config are someone else's.** Nothing in `config/` should contain an
  absolute path; if you find one, it belongs in `config/paths/<site>.yaml`.

---

## 8. Checking that it works, without any data

The fullest check is [`tutorial.md` Part 1](tutorial.md#part-1--a-rehearsal-on-your-machine): every
stage above on the real tools and the real configuration, on miniature sources fabricated in the
GLORYS and GDAC layouts, in about ten minutes — and `pytest tests/test_tutorial.py` runs it. For a
quicker look at the model path alone:

```bash
oceanml3d command=list-models                                   # the registry loads
jobs/run.sh --site local experiment=osse3d_smoke training=debug   # synthetic, CPU, minutes
pytest -m "not slow" -q -k "osse or open_and_patches or model_smoke"
pytest -q tests/test_osse_smoke.py                              # includes the end-to-end, marked slow
```

The last one runs the whole chain — simulate, train one epoch, predict, stitch, export, validate the
manifest — on synthetic data, and asserts the product is not merely well-formed: every target must
vary in space. A trunk stuck at MONAI's zero initialisation exports its per-channel mean, which is
finite, correctly shaped and spatially constant, and no "does the file exist" check can see it.
