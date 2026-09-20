# oceanml3d on Datarmor — step by step

From building the image to a first training run, in order. Validate each step before moving on.

Datarmor is Ifremer's cluster (Pôle de Calcul et de Données Marines, Brest). This runbook covers the
3D multivariate gridded model; `docs/pipeline_3d.md` describes what the model *is* and what the data
means, and is worth reading first.

**Three properties of Datarmor govern everything below:**

* **The scheduler is PBS Pro.** `qsub`, `qstat`, `qdel`; job directives start with `#PBS`. The GPU
  queue is **`gpuq`**.
* **Containers run directly.** Drop the `.sif` on `$DATAWORK` and run it with `singularity exec`. No
  import step, no imposed directory.
* **Compute nodes have no Internet.** One queue does: **`ftp`**. You should rarely need it: GLORYS
  and the Argo GDAC are both mirrored under `/home/ref-*` (F.1, F.3). Only a real download goes there.

> **Your login shell is probably csh.** That is Datarmor's default, and it changes how you set a
> variable for one command. `VAR=value command` is bash syntax; csh reads the whole first word as a
> command name and answers `OCEANML3D_SITE=datarmor: Command not found.` Every example here uses
> `jobs/run.sh --site <name>` instead, which works in any shell. If you need a variable that has no
> flag, either `setenv NAME value` first, or prefix with `env`:
>
> ```csh
> setenv OCEANML3D_DATA $SCRATCH/oceanml3d/data      # csh
> env OCEANML3D_DATA=$SCRATCH/oceanml3d/data some-command   # any shell
> ```
>
> Batch scripts are their own case: a `.pbs` file declares its interpreter on the first line, so
> `#!/bin/csh` and `#!/usr/bin/env bash` jobs coexist happily. The ones in `jobs/pbs/` say which.

> **Sources.** Adapted from NOSC's `env/README_conteneur_datarmor.md`, which draws on Datarmor's
> public community documentation. Points marked **[confirm]** are those the public documentation
> does not settle — mostly GPU details; each says which command answers it in two minutes once you
> are connected. Otherwise: `assistance@ifremer.fr`, "Calcul et données scientifiques".

---

## A. Build the image, on your own machine

Building from a definition file needs root, which you do not have on a shared cluster. Build
locally, transfer the `.sif`.

```bash
# Ubuntu 22.04 or earlier
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt update && sudo apt install -y apptainer

# Ubuntu 24.04+: take the .deb from https://github.com/apptainer/apptainer/releases
# (it installs the AppArmor profile that permits unprivileged namespaces)
```

Then, from a clone of this repository:

```bash
apptainer build oceanml3d-$(date +%Y-%m).sif container/oceanml3d.def
apptainer exec oceanml3d-*.sif python -c "import torch, monai, xarray; print(torch.__version__)"
```

Expect `2.6.0+cu124`. That build runs on Datarmor's driver (530.30.02, CUDA 12.1) through CUDA
minor-version compatibility — a CUDA 12.x build runs on any driver supporting 12.0 or later.

---

## B. Get onto Datarmor, and get the image there

Access host: **`datarmor-access.ifremer.fr`**, intranet credentials. From outside, connect the
**Pulse Secure** VPN first (extranet credentials for the VPN itself, intranet for SSH).

```
Host datarmor
    HostName datarmor-access.ifremer.fr
    User <intranet-login>
    ForwardX11 yes
    ServerAliveInterval 60
```

Transferring the image (several GB) depends on where you are, because the dedicated hosts are not
routed from the same networks. **A `connect to host … port 22: Connection timed out` is not a bad
password — it means that host is not reachable from where you are**, typically `datacopy` from
outside, even with the VPN. Three routes, in order of preference.

**1. From an Ifremer building (or a VPN that routes the whole internal network).** The dedicated
host is `datacopy.ifremer.fr`: scp / sftp / rsync over SSH, intranet credentials. Check it answers
before starting a multi-gigabyte transfer:

```bash
ssh <login>@datacopy.ifremer.fr        # must ask for a password, not time out
rsync -avP oceanml3d-2026-09.sif <login>@datacopy.ifremer.fr:      # lands in your Datarmor $HOME
```

FileZilla over SFTP does the same. If `datacopy` times out while `datarmor-access` answers, your VPN
does not route that host — use route 3.

**2. From outside.** `eftp.ifremer.fr` speaks **FTP** with **extranet** credentials and exposes
`$SCRATCH/eftp` of your Datarmor account.

```bash
# a. create the landing directory, from an SSH session on Datarmor
ssh datarmor
mkdir -p $SCRATCH/eftp
exit

# b. from your machine, FTP with EXTRANET credentials (lftp, or FileZilla in FTP mode)
lftp -u <extranet-login> eftp.ifremer.fr
#   cd eftp
#   put oceanml3d-2026-09.sif
#   bye
```

> **The extranet account is not your Datarmor account.** Different password — Ifremer forbids making
> them identical — and possibly not active yet. A `530 Login incorrect` on `eftp` is nearly always
> this. Created from `teletravail.ifremer.fr` (credentials sent to your Ifremer mail), password reset
> at `https://www.ifremer.fr/chpass/`. Route 3 avoids the account entirely.

**3. Fallback, from anywhere: the access node itself.** You already reach
`datarmor-access.ifremer.fr` over SSH — it is how you log in — so you can drop the `.sif` there
directly. It needs no extranet account and no particular routing.

```bash
ssh datarmor 'mkdir -p ~/containers'
rsync -avP oceanml3d-2026-09.sif <login>@datarmor-access.ifremer.fr:containers/
```

Keep this for occasional transfers: the access node is shared, and is not meant to carry large
repeated transfers. But when `datacopy` is unroutable and the extranet account is not working, it is
the route that always exists.

`rsync -avP` shows progress and resumes an interrupted transfer, which matters over several
gigabytes; `scp` stays silent until it finishes.

**Whichever route you took, park the image on `$DATAWORK`.** Neither `$HOME` (50 GB, backed up) nor
`$SCRATCH` (purged after 10 days) is the right home for it:

```bash
ssh datarmor
mkdir -p $DATAWORK/containers
mv $SCRATCH/eftp/oceanml3d-2026-09.sif $DATAWORK/containers/    # route 2
# or
mv ~/containers/oceanml3d-2026-09.sif $DATAWORK/containers/     # routes 1 and 3
```

---

## C. Storage, and the purge that eats results

Four spaces with very different properties. Confusing them is the first cause of lost data.

| Space | Size | Backed up | Purge | Use |
|---|---|---|---|---|
| `$HOME` | 50 GB | **yes** | no | the clone, scripts, configuration |
| `$DATAWORK` | 1 TB | no | no | the `.sif`, the **master copy** of prepared data |
| `$SCRATCH` | 10 TB | no | **anything older than 10 days is deleted** | working data, run outputs |
| `/home/ref-<theme>/…` | — | — | no | Ifremer reference data, read-only (`ocean-reanalysis`, `argo`, `ecmwf`…) |

```bash
ssh datarmor
git clone <this-repo> $DATAWORK/oceanml3d-core
mkdir -p $DATAWORK/oceanml3d/data $SCRATCH/oceanml3d/runs

# bash
echo 'export OCEANML3D_DATA=$DATAWORK/oceanml3d/data' >> ~/.bashrc
# csh
echo 'setenv OCEANML3D_DATA $DATAWORK/oceanml3d/data' >> ~/.cshrc
```

**Two rules follow from the 10-day purge.** Keep the master copy of prepared data on `$DATAWORK`;
bring checkpoints and products back from `$SCRATCH` after every campaign. Never leave the only copy
of a result on `$SCRATCH` — ten days untouched and it is gone.

**`/home/ref-…` is not a footnote.** Datarmor mirrors the central CMEMS products read-only, GLORYS
included, so the download step most guides start with is usually unnecessary here. See F.1 before
fetching anything.

---

## D. Configure the site

Two files. **Paths only — never keys.**

```bash
cd $DATAWORK/oceanml3d-core
cp config/paths/local.yaml config/paths/datarmor.yaml     # edit the paths, keep all 16 keys
$EDITOR jobs/env/datarmor.sh                              # OCEANML3D_DATA, OCEANML3D_SIF
```

`test_a_site_file_does_not_invent_keys_of_its_own` enforces the "paths only" rule: a key only your
site knows is either a typo or a key the other sites silently lack.

`jobs/env/datarmor.sh` should end up with:

```sh
export OCEANML3D_DATA="${OCEANML3D_DATA:-$DATAWORK/oceanml3d/data}"
export OCEANML3D_PATHS=datarmor
export OCEANML3D_SIF="$DATAWORK/containers/oceanml3d-2026-09.sif"
export OCEANML3D_CONTAINER_CMD=singularity
export OCEANML3D_BIND="$DATAWORK:$DATAWORK,$SCRATCH:$SCRATCH"
export OCEANML3D_EXTRA="training.trainer.precision=16-mixed"   # V100: fp16 yes, bf16 no
```

---

## E. Check the container on a GPU node

**Do nothing heavy on the login node.** It is shared, and meant for editing and submitting.

> **Interactive or batch?** `qsub -I` gives you a shell once the resource is granted — good for
> checks and debugging, but the terminal stays blocked while queuing and the job dies if the
> connection drops. `qsub script.pbs` returns immediately, survives a closed terminal, and is what
> you want for anything long. Converting one to the other is mechanical: the same resources as
> `#PBS` directives, the same command in the body.
>
> **The csh batch trap: `module: Command not found`.** A batch job does not read your `~/.cshrc`, so
> `module` is undefined. Source the init first — the `.pbs` templates in `jobs/pbs/` already do:
>
> ```csh
> source /usr/share/Modules/init/csh   # [confirm: `ls /usr/share/Modules/init/` gives the right one]
> module load singularity
> ```
>
> **The routing trap.** The default queue `sequentiel` routes rather than runs: a large memory
> request can send you to rare big-RAM nodes and a long wait. Ask for what you need — `mem=16g`
> starts sooner than `mem=64g`. To diagnose: `qstat -f <jobid> | grep -i "job_state\|queue\|comment"`.

```bash
qsub -I -q gpuq -l select=1:ncpus=8:ngpus=1:mem=64g -l walltime=00:30:00
module load singularity          # [confirm the exact module name: module avail singularity]

nvidia-smi                       # Tesla V100-PCIE-32GB, driver 530.30.02, CUDA 12.1

singularity exec --nv --bind $DATAWORK,$SCRATCH \
  $DATAWORK/containers/oceanml3d-2026-09.sif \
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect `True Tesla V100-PCIE-32GB`. `False` is a missing `--nv`, nine times out of ten. Note that
`nvidia-smi` does not exist on the login node at all, and inside a container it only appears with
`--nv` — two ways to conclude wrongly that the GPU is broken.

Then the project itself:

`jobs/run.sh` runs **one** `oceanml3d` command — the one you give it — after sourcing
`jobs/env/datarmor.sh` and wrapping it in the container. It is not a pipeline, and it does not
replace the steps in F: those are several commands, in order, each its own job. It prints the
command it is about to run, so you can always see what it did with your arguments.

```bash
cd $DATAWORK/oceanml3d-core
jobs/run.sh --site datarmor command=list-models
jobs/run.sh --site datarmor command=validate experiment=osse3d_gs21_multivar_unet
exit
```

`validate` reports every problem at once. Run it before every submission; it costs a second, and a
queued job failing on a typo costs a day.

---

## F. Prepare the data

### F.1 GLORYS is already on Datarmor — do not download it

**This is the single biggest time saver on this platform, and the easiest to miss.** Datarmor mirrors
the central CMEMS products read-only under `/home/ref-<theme>/`. GLORYS12V1 (product 001-030) is
there:

```bash
ls /home/ref-ocean-reanalysis/
ls /home/ref-ocean-reanalysis/global-reanalysis-phy-001-030-daily/
ncdump -h <one-file>.nc | head -50
```

Check it covers the domain (32–44 °N, 66–54 °W), the period (2010–2020), and carries `thetao`, `uo`,
`vo`, `zos` on several depth levels. If it does, **skip the download entirely**: tens to hundreds of
gigabytes you do not fetch, do not store, and do not count against your quota.

Nothing in the code changes, and there is no recipe to write: `scripts/prepare/recipes/
glorys_gs_multidepth.datarmor.yaml` already reads from the mirror, with `${GLORYS_SRC}`, `${YEAR}`
and `${NEXT}` filled in by `jobs/prepare.sh`. Point `GLORYS_SRC` at the exact directory in
`jobs/env/datarmor.sh`:

```sh
export GLORYS_SRC=/home/ref-ocean-reanalysis/global-reanalysis-phy-001-030-daily
```

**One thing has to be right, and it fails misleadingly:** the mirror must be in `OCEANML3D_BIND`, or
the path resolves to nothing inside the container and looks like a missing dataset rather than a
missing mount. `jobs/env/datarmor.sh` binds it by default.

### F.2 Eleven years: a PBS job array, one year per sub-job

The job is in the repository — `jobs/pbs/prepare_glorys.pbs`. Submit it:

```csh
qsub -v OCEANML3D_SITE=datarmor jobs/pbs/prepare_glorys.pbs
qsub -v OCEANML3D_SITE=datarmor -J 2010-2012 jobs/pbs/prepare_glorys.pbs   # a subset
```

**One year per sub-job, not one monolithic job.** A single job chains thousands of file opens on one
core and saves nothing along the way, so one walltime overrun loses everything. A year that overruns
is relaunched alone — `qsub -J 2015-2015 …` — while the others stay done, and `regrid.py` skips
completed outputs, so relaunching the whole array is safe too.

**Resolution.** Native 1/12° by default. For another step — what `NOSC_TARGET_RES` did — set
`OCEANML3D_TARGET_RES`; the data root then becomes `$OCEANML3D_DATA/res<step>` for **every** step,
training included ([pipeline_3d.md §3.1a](../pipeline_3d.md)). A PBS job does not inherit your
shell's environment, so either give it to each `qsub`, or — simpler — write it once in
`~/.config/oceanml3d/env.sh`, which every job reads:

```bash
mkdir -p ~/.config/oceanml3d
echo 'export OCEANML3D_TARGET_RES="${OCEANML3D_TARGET_RES:-0.25}"' > ~/.config/oceanml3d/env.sh
```

(bash syntax on purpose: the jobs are bash scripts, whatever your login shell. Remove the file, or
pass `OCEANML3D_TARGET_RES=native`, for the native grid. Nothing creates this file for you; details
in [jobs/README.md](../../jobs/README.md).) Each log's first `[prepare]` line prints the
data directory and resolution it uses.

Then merge into the single file the configs expect:

```csh
qsub -v OCEANML3D_SITE=datarmor jobs/pbs/concat_glorys.pbs
```

Both are ten lines of directives around `jobs/prepare.sh`, which is what actually does the work and
which you can run interactively to debug:

```csh
qsub -I -l walltime=00:30:00 -l mem=32g
jobs/prepare.sh --site datarmor glorys 2014
```

Time one year that way before submitting eleven. It validates the output and sizes the walltime.

### F.3 ARGO is on Datarmor too — read the GDAC mirror

Ifremer hosts Coriolis, one of the two Argo global data centres, and the native GDAC is mirrored
read-only. Check it once:

```bash
ls /home/ref-argo/gdac/                                  # dac/ and ar_index_global_prof.txt
head -12 /home/ref-argo/gdac/ar_index_global_prof.txt
```

`scripts/prepare/recipes/argo_profiles_gs.datarmor.yaml` reads it (`source: gdac`,
`gdac_dir: ${ARGO_GDAC}`, set to `/home/ref-argo/gdac` in `jobs/env/datarmor.sh`, which also binds
`/home/ref-argo`). So this runs on a **CPU queue, not `ftp`**, with no `argopy`:

```csh
qsub -v OCEANML3D_SITE=datarmor jobs/pbs/prepare_argo.pbs
```

The global index is read first, so only the floats that crossed the box in the period are opened —
a few hundred files instead of the ~20 000 of the archive — and within each float only the cycles
inside the box and period are decoded. The log says which path it took and how far it got:

```
[argo] index /home/ref-argo/gdac/ar_index_global_prof.txt (… profiles, read in 40 s): 5123 in box/period, from 412 floats
[argo] 50/412 files, 610 profiles kept, 0 failed, 38 s
…
[argo] read: … points in … s
[argo] after QC: … points
```

The index is looked for in `gdac_dir` and its parent; if the mirror keeps it elsewhere, set `index:`
in the recipe. Without it the reader says so and opens every `<dac>/<wmo>/<wmo>_prof.nc` — slow, but
it no longer lists the millions of per-cycle files under `profiles/`. Then: QC on the standard flags,
vertical interpolation, and the coverage table — where and when a float was, and how deep. Delayed-mode
profiles use the `*_ADJUSTED` values, as argopy does. The values are discarded; F.4 replaces them with
the truth. Output: `$OCEANML3D_DATA/argo/argo_profiles_gs.csv`; an existing table is kept (delete it to
rebuild).

**Downloads, when you really need one** — a product the centre does not mirror, or GLORYS outside
the mirror's coverage — go on the only queue with outbound network, `ftp`
(`copernicusmarine login` once interactively first, and year by year).

### F.4 Simulate the observing system — no network needed

```csh
qsub -v OCEANML3D_SITE=datarmor jobs/pbs/prepare_obs.pbs
```

Writes `pseudo_obs_ssh_gs`, `pseudo_obs_sst_gs`, `argo_virtual_thetao_gs21`. **Those three keys must
already be in `config/paths/datarmor.yaml`** — they are outputs, but the simulator learns where to
write from the catalog.

### F.5 Keep the master copy off `$SCRATCH`

```bash
rsync -av $SCRATCH/oceanml3d/data/ $DATAWORK/oceanml3d/data/
```

---

## G. Train

```bash
cd $DATAWORK/oceanml3d-core
qsub -v OCEANML3D_SITE=datarmor,OCEANML3D_ARGS="experiment=osse3d_gs21_multivar_unet" \
     jobs/pbs/train.pbs
```

PBS does not forward trailing arguments the way `sbatch` does, which is why overrides travel in
`OCEANML3D_ARGS`.

**Write outputs to `$SCRATCH`, not to the clone.** Hydra writes `outputs/<date>/<time>/` relative to
the working directory by default — i.e. into `$DATAWORK`, or worse `$HOME`:

```bash
OCEANML3D_ARGS="experiment=osse3d_gs21_multivar_unet hydra.run.dir=$SCRATCH/oceanml3d/runs/$(date +%Y%m%d-%H%M%S)"
```

Compute the timestamp in the shell rather than with Hydra's `${now:…}`, which bash would try to
expand.

Resources for this task (144×144 patches, ~224M-parameter default trunk) — one V100 is enough:

```
#PBS -l select=1:ncpus=8:ngpus=1:mem=64g
#PBS -l walltime=20:00:00
```

Monitor:

```bash
qstat -u $USER
tail -f logs/o3d_train.o<jobid>
qstat -f <jobid> | grep -i "job_state\|comment"     # why is it still queued?
```

Multi-GPU: raise `ngpus` and add `training=ddp` to `OCEANML3D_ARGS`. Do not also wrap the command in
`mpiexec` — Lightning launches its own ranks.

---

## H. Inference, and getting results off `$SCRATCH`

```bash
qsub -v OCEANML3D_SITE=datarmor,OCEANML3D_ARGS="command=predict experiment=osse3d_gs21_multivar_unet ckpt=$SCRATCH/oceanml3d/runs/<run>/checkpoints/<best>.ckpt" \
     jobs/pbs/train.pbs
```

The normalisation statistics travel inside the checkpoint, so a prediction run cannot silently
denormalise with numbers the model never saw.

Then, before the purge:

```bash
rsync -av $SCRATCH/oceanml3d/runs/<run>/ $DATAWORK/oceanml3d/runs/<run>/
```

---

## Datarmor-specific pitfalls

| Symptom | Cause |
|---|---|
| `nvidia-smi: Command not found` | you are on `datarmor3`, a login node. Or inside a container without `--nv`. |
| `module: command not found` from `jobs/run.sh` | `module` is a shell *function*, and a script gets a fresh shell that lacks it. `load_modules` in `jobs/env/_lib.sh` sources the init; if it cannot find yours, its message says where it looked. |
| `module: Command not found` in a batch job | same cause; `.pbs` files source the modules init explicitly |
| `Connection timed out` on `datacopy` | that host is not routed from where you are, not a bad password |
| `530 Login incorrect` on `eftp` | the extranet account, which is not your Datarmor login |
| A job queued for hours | the `sequentiel` routing queue sent a large `mem=` request to rare nodes |
| Prepared data gone | `$SCRATCH`, ten days untouched |
| `torch.cuda.is_available()` is `False` | `--nv` missing |
| Quota exceeded before the first epoch | Hydra wrote `outputs/` into `$HOME` or `$DATAWORK`; set `hydra.run.dir` |
| `no file matches /home/ref-…` inside the container | the mirror is not in `OCEANML3D_BIND`. An unbound path is *empty*, not missing, so the error is about the pattern rather than the mount. |
| `FATAL: mount source … doesn't exist` | a bind path that does not exist on the host. `$OCEANML3D_DATA` is an output directory and is created for you now; anything else in `OCEANML3D_BIND` is skipped with a reason. |
| `Segmentation fault` from `singularity exec`, after `[regrid] N input file(s)` | netCDF4/HDF5 is not thread-safe and the C library dies rather than raising. `parallel` is off by default and the read runs on dask's synchronous scheduler; turn `parallel: true` on per recipe only once you have seen it work here. |
| `Killed` after `[regrid] writing … lat': 2041, 'lon': 4320 …` | the whole globe is being written: the recipe has no `domain:`, or a `regrid.py` older than 2026-09-19 ignored it. Now refused up front by `max_gb`. |
| `[regrid] N input file(s)` far from 365, with other years in the first/last names | the input glob matched the `_R<production date>` part of the names. Anchor the year on the validity date: `*_mean_${YEAR}????_R*.nc`. |
| `by_year/*.nc` files but the merge finds nothing | written before 2026-09-19, when the per-year step still wrote NetCDF. Every intermediate is Zarr now (`by_year/*.zarr`, then `glorys/*.zarr`); delete the `.nc` and resubmit the array. |
| `no file matches …/by_year/…zarr` at the merge, "They exist under …/res0.25" | the per-year step ran with a target resolution and the merge without: give both the same `OCEANML3D_TARGET_RES`, or set it once in `~/.config/oceanml3d/env.sh`. |
| ARGO job killed on walltime with nothing after `ARGO profiles and coverage table from …` | before 2026-09-20: output was buffered and the reader decoded every cycle of every float in Python loops, or recursed into `profiles/` when the index was not found. Now each stage and every 50 files is logged; check the `[argo] index …` line. |
| A `.tmp.<name>` directory in `by_year/` or `glorys/` | a write that was killed. Harmless: the next run removes it and starts that output again. |
| An error about HDF5 file locking, or a hang on the first open | Lustre does not implement the locking HDF5 wants. `regrid.py` sets `HDF5_USE_FILE_LOCKING=FALSE`; other tools may need it exported too. |
| `Could not find any nv files on this host!` | `--nv` on a CPU queue. Harmless, and no longer printed: `--nv` is passed only where a driver is present. Force it with `OCEANML3D_NV=1`. |
| `Unmatched "` from a `.pbs` job | csh cannot carry a multi-line double-quoted string. Call a `.py` file, never `python -c`. |
| A whole GLORYS job lost on walltime | one monolithic job saves nothing along the way. Use the `-J` array, one year per sub-job (F.2). |
