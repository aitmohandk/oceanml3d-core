# Build a pipeline end to end, step by step

From the raw sources to a scored reconstruction of the 3D ocean: prepare the truth, build the Argo
coverage table, simulate the observing system, check the configuration, train, run inference, score.

The tutorial has three parts:

* **[Part 1 — A rehearsal on your machine](#part-1--a-rehearsal-on-your-machine)**, about ten
  minutes. The **real tools** and the **real task configuration**, run the way a cluster job runs them,
  on miniature sources fabricated in the exact layouts of the GLORYS mirror and the Argo GDAC. Nothing
  is mocked, so every command, file name and check below is the one you will meet on the cluster.
  `tests/test_tutorial.py` extracts the commands of this part from this very page and runs them,
  so the page cannot drift from the code unnoticed.
* **[Part 2 — The real run on a cluster](#part-2--the-real-run-on-a-cluster)**, the same steps on
  Datarmor or Jean Zay, submitted as jobs, with what to expect from each.
* **[Part 3 — Make it yours](#part-3--make-it-yours)**: another resolution, domain, model or site.

What each stage *means* is in [`pipeline_3d.md`](pipeline_3d.md); every configuration key is in
[`gridded_models.md`](gridded_models.md). This page is about doing it.

---

## Part 1 — A rehearsal on your machine

You need a clone, Python ≥ 3.10 and `bash` (on Datarmor, whose login shell is `csh`, type `bash`
first — `export VAR=value` is bash syntax).

### Step 0 — Install, choose a data root, fabricate the sources

<!-- tutorial-test: skip (installation) -->
```bash
git clone https://github.com/aitmohandk/oceanml3d-core && cd oceanml3d-core
pip install -e '.[dev,zarr]'
```

Then, from the clone:

```bash
export OCEANML3D_DATA=$PWD/tutorial/data     # the data root: every step reads and writes under it
export OCEANML3D_TARGET_RES=0.5              # the grid step; everything below goes to .../res0.5/
python scripts/make_tutorial_sources.py $OCEANML3D_DATA
export ARGO_GDAC=$OCEANML3D_DATA/gdac        # where the Argo GDAC is; Datarmor's site file sets it
```

**What you have now**, under `tutorial/data/` (ignored by git):

* `raw/glorys/2018/` and `raw/glorys/2019/` — 45 daily files per year named like the CMEMS originals
  (`mercatorglorys12v1_gl12_mean_20190101_R20190101.nc`), with `thetao, uo, vo, zos` on the 26 upper
  GLORYS levels over a box a little larger than the task domain. `raw/glorys` is where the job layer
  looks for GLORYS by default on a site without a mirror (`GLORYS_SRC`), so nothing needs setting.
* `gdac/` — `ar_index_global_prof.txt` and `dac/aoml/<wmo>/<wmo>_prof.nc` for 12 floats: the native
  GDAC tree, with QC flags stored as the GDAC stores them.

**Why 0.5°.** It is what makes the rehearsal fast, and it is also a resolution people run: at 0.5° the
Gulf Stream box is 25 × 25 cells, the patches are fitted to it by themselves, and the one setting that
does *not* follow — the border crop — is dealt with in step 4.

### Step 1 — The GLORYS truth: one year at a time, then the merge

```bash
jobs/prepare.sh --site local glorys 2018
jobs/prepare.sh --site local glorys 2019
jobs/prepare.sh --site local concat
```

**What happens.** `jobs/prepare.sh` loads the site (`jobs/env/local.sh`, then your
`~/.config/oceanml3d/env.sh` if you have one), derives the data directory from the resolution, and runs
`scripts/prepare/regrid.py` with the recipe `scripts/prepare/recipes/glorys_gs_multidepth.yaml`: every
daily file is cut to the domain before being read, interpolated onto a regular 0.5° grid anchored on
the domain bounds, and written as one Zarr store per year. `concat` merges the years into the store the
catalog reads. A site without a mirror would download the year first; here the files are already in
`raw/glorys/<year>`, so it does not.

**What it writes**, under `tutorial/data/res0.5/`:

```
by_year/glorys_gs_multidepth_2018.zarr
by_year/glorys_gs_multidepth_2019.zarr
glorys/glorys_gs_multidepth_2010-2020.zarr      ← the truth: catalog keys glorys_gs_multidepth and glorys_gs_surface
```

The `2010-2020` in the name is a label (`GLORYS_YEARS`, default `2010:2020`), not a promise about the
content: the catalog refers to the store by that name, so leave it.

**Check.** The last lines of each step are `[prepare] done: glorys` and `[prepare] done: concat`. Then:

```bash
python -c "import xarray as xr; print(xr.open_zarr('$OCEANML3D_DATA/res0.5/glorys/glorys_gs_multidepth_2010-2020.zarr'))"
```

Expect dimensions `time: 90, depth: 26, lat: 25, lon: 25` and the four variables `thetao, uo, vo, zos`.

### Step 2 — The Argo coverage table

```bash
jobs/prepare.sh --site local argo
```

**What happens.** Because `ARGO_GDAC` is set, the step uses `argo_profiles_gs.gdac.yaml` (without it,
it would download with argopy). It reads the GDAC index first, so only the floats that crossed the box
in the period are opened; applies the standard QC; interpolates each profile onto the truth's depth
levels; and writes one row per profile: where, when, and which of the 21 levels it reached. The values
themselves are not used later — step 3 replaces them with the truth. That is what makes the Argo input
*virtual*: real geometry, simulated values.

**What it writes**: `res0.5/argo/argo_profiles_gs.csv`, and a per-float cache in `res0.5/argo/.cache_gs/`
that lets an interrupted run resume.

**Check.** The run ends with a summary of what the table holds:

```
96 profiles -> .../res0.5/argo/argo_profiles_gs.csv
[argo] table: 96 profiles, 12 floats, 2018-01-03 .. 2019-02-14
[argo] per year: 2018 49  2019 47
[argo] reaching each level: d00 0%  d02 0%  d04 100%  d06 100%  ...  d25 100%
[argo] WARNING fewer than half the profiles reach d00, d02: the virtual ARGO input will be nearly empty at those depths
[argo] WARNING fewer than 100 profiles in 2010, 2011, ... -- the requested window is not evenly covered
```

Both warnings are expected here. The second because the fabricated data only covers two years. The
first is **real physics, and you will see it on the real data too**: an Argo float stops its ascent a
few decibars below the surface, so the 0.5 m and 2.6 m levels are rarely or never sampled. Those
channels will be empty, which is correct — the surface is observed by the SST input instead.

### Step 3 — Simulate the observing system

```bash
jobs/run.sh --site local command=prepare-obs experiment=tutorial
```

**What happens.** `jobs/run.sh` runs one `oceanml3d` command with the site's catalog appended
(`paths=local`). `experiment=tutorial` is the real task, `osse3d_gs21_multivar_unet`, shrunk to run on a
laptop (`config/experiment/tutorial.yaml` says exactly what is shrunk). `prepare-obs` reads the truth and
the coverage table and simulates what an observing system would have seen of it.

**What it writes**, under `res0.5/osse/`:

| File | Content |
|---|---|
| `pseudo_obs_ssh_gs.nc` | SSH along the ground tracks of six altimeters, with noise; the track mask; one mask per mission |
| `pseudo_obs_sst_gs.nc` | the surface temperature under synthetic clouds (60 % coverage), with noise; its mask |
| `argo_virtual_thetao_gs21.nc` | the truth sampled where and when the real floats were, one channel per level plus a presence mask |

**Check.** Three lines, one per file: `pseudo-obs: …ssh…`, `pseudo-obs: …sst…`, `virtual ARGO: …`.
Run the command again and the last one says `virtual ARGO: reusing …` — the files are kept when they are
still valid (same time axis and grid as the truth, newer than the table) and rebuilt, with the reason,
when they are not.

### Step 4 — Validate

```bash
jobs/run.sh --site local command=validate experiment=tutorial
```

**What happens.** Everything that can be checked without training is checked, and every problem is
reported at once: the splits (disjoint, with gaps longer than the window), the patches and the overlap,
the catalog keys, the files, and whether the border crop leaves an empty rim inside the region the
metrics score.

**Check.** Three lines:

```
[oceanml3d] patch/stride from the grid (lat: 25 cells -> patch 16, stride 12, lon: 25 cells -> patch 16, stride 12)
[oceanml3d] exported product empty over the outer rim: lat 1 deg (2 cells x 0.5), lon 1 deg (2 cells x 0.5)
configuration 'tutorial' is valid
```

**About the rim.** Each patch loses a border of `training.rec_weight.crop` cells, and nothing covers
the outer edge of the domain, so the exported field is empty over a rim of `crop × grid step`. The
reference task crops 4 cells: 0.33° at 1/12°, well inside the 1° margin that `eval_domain` leaves
out of the scores. At 0.5° it would be 2°, and validation refuses it, naming the crop that fits —
which is why `experiment=tutorial` sets 2. Try it:

<!-- tutorial-test: skip (meant to fail; tests/test_eval_metrics.py checks the refusal) -->
```bash
jobs/run.sh --site local command=validate experiment=tutorial training.rec_weight.crop.lat=4
```

`validate` exits with `the crop leaves a 2 deg rim of the product empty ... Either
training.rec_weight.crop.lat=2 ...`. Every refusal reads like that one: what is wrong, why it matters,
what to change.

### Step 5 — Train

```bash
jobs/run.sh --site local experiment=tutorial
```

**What happens.** `train` is the default command. It re-runs `prepare-obs` (which reuses the files),
validates again, opens the data lazily, computes the normalisation statistics on the train split, then
trains; validates each epoch on `val/nrmse`, keeps the best checkpoints, **tests** the best one on the
test split, and **exports** the test split as a product. Two epochs of four batches: about half a
minute on a laptop CPU.

**What it writes** — a run directory, `outputs/tutorial/<date>_<time>/`:

```
config.yaml  .hydra/  git_hash.txt  hparams.yaml    what ran, exactly
norm_stats.json                                   the statistics (also inside every checkpoint)
metrics.csv                                       every logged value, one row per step or epoch
checkpoints/000-<val_nrmse>.ckpt  001-….ckpt  last.ckpt
product/product.yaml  product/daily/tutorial_2019-02-02.nc … _2019-02-14.nc
```

**Check.**

* The log shows `val/nrmse` in the progress bar, then a table of `test/…` metrics, then
  `product manifest written to …/product/product.yaml`.
* Two lines you will also see: `WARNING 2 channel(s) have no finite value in the train window:
  argo_thetao_d00, argo_thetao_d02` — the unsampled levels of step 2, handled — and, if TensorBoard is not
  installed, `TensorBoard logging off …; metrics go to …/metrics.csv only`.
* Lines you must **not** see: `… time steps (…%) absent …` or `these files share no date with the rest
  of the task`. Either means an input is not aligned with the truth; the message names the file.

```bash
RUN=$(dirname $(ls -t outputs/tutorial/*/metrics.csv | head -1))      # the training run: it has metrics.csv
python -c "import pandas as pd; m = pd.read_csv('$RUN/metrics.csv'); print(m.filter(like='nrmse').dropna(how='all').tail())"
```

Every command creates a run directory under `outputs/tutorial/`, `validate` and `prepare-obs` included;
the training one is the one with a `metrics.csv`. Expect `val/nrmse` near 1 after two epochs — the
spread of the truth, i.e. nothing learnt yet, which is all two epochs can do.

### Step 6 — Inference from a checkpoint

```bash
CKPT=$(ls -t outputs/tutorial/*/checkpoints/last.ckpt | head -1)
jobs/run.sh --site local command=predict experiment=tutorial ckpt=$CKPT
```

**What happens.** The model is rebuilt from the configuration, its weights and normalisation
statistics are read from the checkpoint, and the test split is predicted patch by patch, stitched with
the reconstruction weights, denormalised and exported. No training. This is the step you run to apply
a trained model to another period (`data.splits.test.time=…`) or to regenerate a product.

**What it writes**: a new run directory with a `product/` like step 5's.

**Check** that the product satisfies the contract `oceanml3d-eval` relies on:

```bash
MANIFEST=$(ls -t outputs/tutorial/*/product/product.yaml | head -1)
python -c "
import yaml; from oceanml3d.inference.product_contract import validate_manifest
m = yaml.safe_load(open('$MANIFEST'))
print('contract:', validate_manifest(m, strict=True) or 'OK', '|', len(m['variables']), 'variables')"
```

Expect `contract: OK | 64 variables` — `ssh`, `thetao_d00 … thetao_d25`, `u_…`, `v_…`, one file per day.

### Step 7 — Score it

The scores live in the companion repository, which reads the product through its manifest and the
truth through its own.

<!-- tutorial-test: skip (needs the oceanml3d-eval clone) -->
```bash
pip install -e ../oceanml3d-eval            # a clone next to this one
export OCEANML3D_DATA=$OCEANML3D_DATA/res0.5       # the evaluation reads the prepared directory itself

oceanml3d-eval split --input $OCEANML3D_DATA/glorys/glorys_gs_multidepth_2010-2020.zarr \
    --out $OCEANML3D_DATA/glorys_gs21_truth --name glorys_gs21_truth \
    --var ssh=zos --var thetao=thetao --var u=uo --var v=vo --depth-indices 0,2,4,6,8,10-25
oceanml3d-eval run -b osse3d_gs21 -p $MANIFEST --out $OCEANML3D_DATA/eval
```

`split` turns the truth into daily files in the same layout as a product, once. `run` prints every
score per variable, then a leaderboard row. **The numbers mean nothing here** — thirteen days of a
synthetic front and eight gradient steps — and the rehearsal is not about them: it is about every
stage having run, on the real code, with the real configuration.

### Clean up

<!-- tutorial-test: skip -->
```bash
rm -rf tutorial/ outputs/tutorial/
```

---

## Part 2 — The real run on a cluster

The same steps, as jobs. The differences are what the cluster adds: a container, a site file,
schedulers, walltimes — and data a thousand times larger.

### Before the first job

1. **The image.** Build it on your own machine and copy it over:
   `apptainer build oceanml3d.sif container/oceanml3d.def` — runbooks
   [Datarmor A–B](platforms/datarmor.md#a-build-the-image-on-your-own-machine),
   [Jean Zay A–C](platforms/jeanzay.md#a-build-the-image-on-your-own-machine). The image carries the
   dependencies; the code that runs is always your clone's.
2. **The site.** `jobs/env/datarmor.sh` / `jobs/env/jeanzay.sh` and `config/paths/<site>.yaml` ship
   filled in. Check `OCEANML3D_DATA` (the data root) and `OCEANML3D_SIF` (the image) in the site file.
   On Jean Zay, put your account in `jobs/slurm/*.sbatch`.
3. **The resolution**, once for every job — a batch job does not inherit your shell:
   ```bash
   mkdir -p ~/.config/oceanml3d
   echo 'export OCEANML3D_TARGET_RES="${OCEANML3D_TARGET_RES:-0.25}"' > ~/.config/oceanml3d/env.sh
   ```
   Leave the file out for the native 1/12°. Coarser than 0.25°, lower the crop (step 4 of Part 1):
   at 0.25° the default 4 cells make a rim of exactly the 1° `eval_domain` leaves out, and pass; at 0.5°
   add `training.rec_weight.crop.lat=2 training.rec_weight.crop.lon=2`. `validate` gives the number.

Every job writes its log in `logs/` in the clone — `logs/<jobid>.OU` on Datarmor,
`logs/<job-name>-<jobid>.out` on Jean Zay. Its first lines say which site, which data directory and
which resolution it used; read them first.

### The steps

| # | Datarmor (PBS Pro) | Jean Zay (Slurm) |
|---|---|---|
| 1 | `qsub -v OCEANML3D_SITE=datarmor jobs/pbs/prepare_glorys.pbs` | `sbatch --export=ALL,OCEANML3D_SITE=jeanzay jobs/slurm/prepare_glorys.sbatch` |
| 2 | `qsub -v OCEANML3D_SITE=datarmor jobs/pbs/concat_glorys.pbs` | `sbatch --export=ALL,OCEANML3D_SITE=jeanzay jobs/slurm/concat_glorys.sbatch` |
| 3 | `qsub -v OCEANML3D_SITE=datarmor jobs/pbs/prepare_argo.pbs` | `sbatch --export=ALL,OCEANML3D_SITE=jeanzay jobs/slurm/prepare_argo.sbatch` |
| 4 | `qsub -v OCEANML3D_SITE=datarmor jobs/pbs/prepare_obs.pbs` | `sbatch --export=ALL,OCEANML3D_SITE=jeanzay jobs/slurm/prepare_obs.sbatch` |
| 5 | in an interactive session (below): `jobs/run.sh --site datarmor command=validate experiment=osse3d_gs21_multivar_unet` | the same with `--site jeanzay` |
| 6 | `qsub -v OCEANML3D_SITE=datarmor,OCEANML3D_ARGS="experiment=osse3d_gs21_multivar_unet" jobs/pbs/train.pbs` | `sbatch --export=ALL,OCEANML3D_SITE=jeanzay jobs/slurm/train.sbatch experiment=osse3d_gs21_multivar_unet` |
| 7 | `qsub -v OCEANML3D_SITE=datarmor,OCEANML3D_ARGS="command=predict experiment=osse3d_gs21_multivar_unet ckpt=<path>" jobs/pbs/train.pbs` | `sbatch --export=ALL,OCEANML3D_SITE=jeanzay jobs/slurm/train.sbatch command=predict experiment=osse3d_gs21_multivar_unet ckpt=<path>` |

Submit a step once the previous one has finished (`qstat -u $USER` / `squeue -u $USER` empty). Steps
1–4 write under `$OCEANML3D_DATA/res<step>/` exactly the files of Part 1; step 6 writes the run
directory under `outputs/` in the clone unless you move it (below).

**1 — GLORYS, eleven years.** An array: one job per year, 2010–2020, so a walltime overrun costs one
year, and resubmitting skips the finished ones. On Datarmor the files are read in place from the
centre's read-only mirror (`/home/ref-ocean-reanalysis/...`, never modified); on Jean Zay each task
downloads its year into `$GLORYS_SRC/<year>` first, which is why it runs on a pre/post node. At native
resolution a year of the box is about 4.6 GB before compression. Check: `by_year/` holds eleven stores;
each log ends with `[prepare] done: glorys`. A single year: `qsub -J 2015-2015 …` / `sbatch --array=2015 …`.

**2 — Merge.** One job, threads, a few minutes. Check: `glorys/glorys_gs_multidepth_2010-2020.zarr`,
log ending in `[prepare] done: concat`. If it says it finds nothing but sees stores under a sibling
`res*/`, the two jobs did not see the same resolution: that is what `~/.config/oceanml3d/env.sh` prevents.

**3 — Argo.** On Datarmor, from the GDAC mirror `/home/ref-argo/gdac`: for the Gulf Stream box over
2010–2020 the index read takes about 15 s and selects about 9 000 profiles from about 220 floats,
opened at about 0.3 s each — a minute or two in all, about 8 600 profiles after QC. On Jean Zay,
downloaded with argopy, much longer. Check the summary at the end: the years 2010–2019 each with
several hundred profiles, and the level coverage (the top one or two levels low, as in Part 1). If the
job reaches its time budget it stops cleanly with exit code 75 and resubmits itself; nothing is read
twice.

**4 — Observing system.** Check: the three `osse/*.nc` files, and three lines in the log.

**5 — Validate**, interactively, before spending GPU hours:

```bash
qsub -I -q omp -l select=1:ncpus=4:mem=16g -l walltime=00:30:00      # Datarmor
srun -A <project>@cpu --time=00:30:00 --pty bash                       # Jean Zay
cd <your clone>
jobs/run.sh --site <site> command=validate experiment=osse3d_gs21_multivar_unet
```

**6 — Train.** A V100 is enough for the reference configuration (about 15 M parameters). The training
job re-runs step 4 at start, which only reuses the files. Put the run directory on the scratch space by
adding `hydra.run.dir=$SCRATCH/oceanml3d/runs/<name>` to the overrides, and copy it to permanent storage
before the purge ([Datarmor H](platforms/datarmor.md#h-inference-and-getting-results-off-scratch)).
Check, in the log: the container line with `--nv`, `GPU available: True`, `val/nrmse` decreasing from
epoch to epoch, then `product manifest written to …`. Duration and memory for the full run have not
been recorded yet — that is the acceptance run of `PLAN.md`; please add them to `CHANGELOG.md` when you
have them.

**7 — Inference** from the best checkpoint (the one with the lowest `val_nrmse` in its name), on the
test split or any other period: `data.splits.test.time=[2019-01-01,2019-12-31]`.

**8 — Score** with `oceanml3d-eval`, as in Part 1 step 7, with `OCEANML3D_DATA` pointing at the prepared
directory. Add `--baselines` once the climatology and persistence products are built
(`oceanml3d-eval baseline-product`, see `benchmarks/osse3d_gs21.yaml` in that repository): a model that
does not beat the climatology has learnt nothing beyond the season.

### When something goes wrong

Read the log from the top; the last line it reached locates the problem. The runbooks list every
failure met so far with its cause and fix:
[Datarmor](platforms/datarmor.md#datarmor-specific-pitfalls), [Jean Zay](platforms/jeanzay.md#jean-zay-specific-pitfalls),
and [`gridded_models.md` §13](gridded_models.md#13-troubleshooting) for configuration messages.

---

## Part 3 — Make it yours

| To change | Do | Read |
|---|---|---|
| the resolution | `OCEANML3D_TARGET_RES` (and the crop, as `validate` says); everything else follows | [`pipeline_3d.md` §3.1a](pipeline_3d.md#31a-choosing-the-resolution) |
| the period or the splits | `data.splits.*` on the command line or in a new experiment file | [`gridded_models.md` §4.3](gridded_models.md#43-splits) |
| an ablation | `ablation=<name>`: loss, heads, attention, time mixing, EOF modes, 4DVarNet solver, trunk, bathymetry | [`README.md` §3.4](../README.md#34-ablations--configablation) |
| the model | `model=<name>` or a new one registered with `@register_model` | [`adding_a_model.md`](adding_a_model.md) |
| the domain or the variables | a new file in `config/data/`, a new experiment pointing at it, and the recipes that produce its catalog keys | [`gridded_models.md` §4](gridded_models.md#4-the-data-group--defining-a-task), [`data_preparation.md`](data_preparation.md) |
| the machine | `jobs/env/<site>.sh` + `config/paths/<site>.yaml` | [`jobs/README.md`](../jobs/README.md#adding-a-site) |

A new experiment is a small file. The one this tutorial ran, `config/experiment/tutorial.yaml`, is a
complete example: it inherits the reference and changes only what it names.
