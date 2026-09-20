# oceanml3d on Jean Zay — step by step

From building the image to a first training run, in order. Validate each step before moving on.

Jean Zay is IDRIS's national cluster. This runbook covers the 3D multivariate gridded model;
`docs/pipeline_3d.md` describes what the model *is* and what the data means, and is worth reading
first.

**Two constraints of Jean Zay govern everything below:**

* **Compute nodes have no Internet, and login nodes do not run containers.** Every download —
  GLORYS through `copernicusmarine`, ARGO through `argopy` — goes through a **pre/post-processing**
  node (`--partition=prepost`), the only place that combines outbound network, plenty of memory, and
  permission to run containers.
* **Quotas are tight, in inodes as much as in bytes.** 500 000 inodes on `$WORK`, shared across the
  whole project. Hence the container, and hence data and outputs on `$SCRATCH`.

> **Sources.** Adapted from NOSC's `env/README_conteneur.md`, written against project `yrf`. Account
> names, logins and the exact QoS list are yours to substitute. IDRIS changes partition names and
> limits from time to time: **confirm against the current IDRIS documentation** rather than trusting
> this page alone. Points that most often move are marked **[confirm]**.

---

## A. Build the image, on your own machine

Jean Zay does not allow building from a definition file — that needs root. Build locally, transfer
the `.sif`, register it (step C).

```bash
# Ubuntu 22.04 or earlier
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt update && sudo apt install -y apptainer
# Ubuntu 24.04+: the .deb from https://github.com/apptainer/apptainer/releases
```

From a clone of this repository:

```bash
apptainer build oceanml3d-$(date +%Y-%m).sif container/oceanml3d.def
apptainer exec oceanml3d-*.sif python -c "import torch, monai, xarray; print(torch.__version__)"
```

Expect `2.6.0+cu124`, which covers V100 (sm_70), A100 (sm_80) and H100 (sm_90) — one image for every
partition.

---

## B. Transfer

Access is usually via a jump host. In `~/.ssh/config`:

```
Host imt
    HostName ssh.telecom-bretagne.eu
    User <your-imt-login>

Host jz
    HostName jean-zay.idris.fr
    User <your-idris-login>
    ProxyJump imt
    ServerAliveInterval 60
```

```bash
rsync -avP oceanml3d-2026-09.sif jz:$WORK/
```

`rsync -P` shows progress and resumes; `scp` is silent until it finishes, which on several gigabytes
is a long silence.

---

## C. Register the image

```bash
ssh jz
module load singularity
idrcontmgr cp $WORK/oceanml3d-2026-09.sif
idrcontmgr ls
rm $WORK/oceanml3d-2026-09.sif        # reclaim the WORK space and its inodes
```

`idrcontmgr` checks the image against IDRIS's security constraints and places it in
`$SINGULARITY_ALLOWED_DIR`, **the only directory it can be executed from**. That path is not
configurable and the area holds a limited number of images (20 at the time of writing, **[confirm]**).
Leave `$SINGULARITY_CACHEDIR` alone; it is created for you under `$SCRATCH`.

---

## D. Storage

| Space | Size | Purge | Backed up | Use |
|---|---|---|---|---|
| `$HOME` | 3 GB, 150 000 inodes | no | yes | configuration — **never data** |
| `$WORK` | 5 TB, 500 000 inodes (per project) | no | no | the clone |
| `$SCRATCH` | very large | **30 days without access** | no | **data, outputs, checkpoints** |
| `$STORE` | very large, few inodes | no | yes | `.tar` archives |
| `$JOBSCRATCH` | node-local disk | end of job | no | pre-staging; the fastest |
| `$DSDIR` | — | — | — | IDRIS public datasets, read-only |

```bash
git clone <this-repo> $WORK/oceanml3d-core
mkdir -p $SCRATCH/oceanml3d/data $SCRATCH/oceanml3d/runs
echo 'export OCEANML3D_DATA=$SCRATCH/oceanml3d/data' >> ~/.bashrc
```

**Guard against the purge.** `$SCRATCH` is wiped after thirty days without access. Archive the
master copy as a *single* tar, since `$STORE` gives volume but few inodes:

```bash
tar cf $STORE/oceanml3d_data_2026-09.tar -C $SCRATCH/oceanml3d data/
```

**Watch file counts, not just bytes.** One NetCDF per day per sensor reaches tens of thousands of
files fast, which degrades Lustre for everyone and burns your inodes. Aggregate: chunks of tens to
hundreds of megabytes, aligned on how the data is read — for this project, spatial patches traversed
in time.

```bash
idr_quota_user
idr_quota_project
```

Before downloading anything, check `$DSDIR`: if the reanalysis you need is already mirrored there,
you save both the download and the quota.

---

## E. Configure the site

Both files ship filled in; what is left to you is the account in `jobs/slurm/*.sbatch`.

```bash
cd $WORK/oceanml3d-core
$EDITOR config/paths/jeanzay.yaml     # the sixteen keys, rooted at OCEANML3D_DATA; paths only, never keys
$EDITOR jobs/env/jeanzay.sh
```

`$WORK`, not `$SCRATCH`: `$SCRATCH` is purged, and the Zarr stores cost inodes rather than volume —
`$WORK` allows 500 000 for the project and the whole GLORYS chain is about 1 400. If the `$WORK`
quota is the binding constraint, move `OCEANML3D_DATA` to `$SCRATCH` and re-prepare after a purge,
rather than splitting the catalog between the two.

```sh
export OCEANML3D_DATA="${OCEANML3D_DATA:-$WORK/oceanml3d}"
export OCEANML3D_PATHS=jeanzay
export OCEANML3D_SIF="$SINGULARITY_ALLOWED_DIR/oceanml3d-2026-09.sif"
export OCEANML3D_CONTAINER_CMD=singularity
export OCEANML3D_BIND="$WORK:$WORK,$SCRATCH:$SCRATCH"
export OCEANML3D_EXTRA="training.trainer.precision=bf16-mixed"   # A100/H100. On V100: 16-mixed
```

---

## F. Check the container on a compute node

Containers never run on login nodes. A V100 in the development QoS is enough, and it avoids loading
an `arch/` module while debugging.

```bash
srun -A <project>@v100 --qos=qos_gpu-dev --gres=gpu:1 --cpus-per-task=10 \
     --time=00:30:00 --hint=nomultithread --pty bash
module load singularity

singularity exec --nv --bind $WORK:$WORK,$SCRATCH:$SCRATCH \
  $SINGULARITY_ALLOWED_DIR/oceanml3d-2026-09.sif \
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect `True Tesla V100-...`. `False` is a missing `--nv`.

```bash
cd $WORK/oceanml3d-core
jobs/run.sh --site jeanzay command=list-models
jobs/run.sh --site jeanzay command=validate experiment=osse3d_gs21_multivar_unet
exit
```

`-A` is **mandatory** as soon as you have more than one allocation: Slurm otherwise refuses with
"Multiple accounts available".

---

## G. Prepare the data

### G.1 Downloads, on a pre/post node

Not a login node: CPU time is capped there, a multi-hour download will be cut, and **containers
cannot run there** — while `copernicusmarine` lives in the image. Not a compute node either: no
network. The pre/post node is the only one with all three properties, and it does not spend your GPU
hours.

Fill your account and `--partition=prepost` into the `##SBATCH` lines of
`jobs/slurm/prepare_glorys.sbatch` and `jobs/slurm/prepare_argo.sbatch` (they ship commented out,
since the account is yours), then:

```bash
srun -A <project>@cpu --partition=prepost --time=01:00:00 --pty bash
scontrol show partition prepost | grep -i maxtime      # confirm before trusting a long walltime

# once, so every year does not re-ask for credentials
jobs/prepare.sh --site jeanzay -- copernicusmarine login   # or run it inside the container directly
exit
```

Point `GLORYS_SRC` at wherever GLORYS should come from in `jobs/env/jeanzay.sh` — `$DSDIR` if IDRIS
already mirrors it, a download directory otherwise. **Look in `$DSDIR` first**: if the reanalysis is
already there you save the download, the space and the inodes.

### G.2 Eleven years: a Slurm array, one year per task

```bash
sbatch --export=ALL,OCEANML3D_SITE=jeanzay jobs/slurm/prepare_glorys.sbatch
sbatch --export=ALL,OCEANML3D_SITE=jeanzay --array=2015 jobs/slurm/prepare_glorys.sbatch   # one year
sbatch --export=ALL,OCEANML3D_SITE=jeanzay jobs/slurm/concat_glorys.sbatch
```

One year per task, for the same reason as everywhere: a monolithic job saves nothing along the way,
so a walltime overrun loses the lot. `regrid.py` skips completed outputs, so resubmitting the array
resumes rather than restarts.

**Resolution.** Native 1/12 deg by default. For another step, set `OCEANML3D_TARGET_RES`; the data
root then becomes `$OCEANML3D_DATA/res<step>` for every step, training included
([pipeline_3d.md section 3.1a](../pipeline_3d.md)). Rather than adding it to every `--export`, write
it once in `~/.config/oceanml3d/env.sh`, which every job reads ([jobs/README.md](../../jobs/README.md));
nothing creates that file for you:

```bash
mkdir -p ~/.config/oceanml3d
echo 'export OCEANML3D_TARGET_RES="${OCEANML3D_TARGET_RES:-0.25}"' > ~/.config/oceanml3d/env.sh
```

### G.3 ARGO coverage table — pre/post again, it downloads

```bash
sbatch --export=ALL,OCEANML3D_SITE=jeanzay jobs/slurm/prepare_argo.sbatch
```

### G.4 Simulate the observing system — anywhere

`prepare-obs` downloads nothing, so any node will do.

```bash
sbatch --export=ALL,OCEANML3D_SITE=jeanzay jobs/slurm/prepare_obs.sbatch
```

The three output keys must already be in `config/paths/jeanzay.yaml`.

### G.5 Archive

```bash
tar cf $STORE/oceanml3d_data_2026-09.tar -C $SCRATCH/oceanml3d data/
```

---

## H. Train

### H.1 Choosing the partition

`-A` sets the allocation charged and governs access; `-C` sets the node type. Figures below are from
NOSC's guide for project `yrf` — **[confirm] against current IDRIS documentation**, these move.

| | V100 (default) | A100 (`gpu_p5`) | H100 (`gpu_p6`) |
|---|---|---|---|
| `-A` | `<proj>@v100` | `<proj>@a100` | `<proj>@h100` |
| `-C` | none, or `v100-16g` / `v100-32g` | `a100` | `h100` |
| GPUs per node | 4 | 8 | 4 |
| Memory per GPU | 16 or 32 GB | 80 GB | 80 GB |
| `--cpus-per-task` | 10 | 8 | 24 |
| Max walltime | 100 h | **20 h** | 100 h |
| Module first | none | `arch/a100` | `arch/h100` |
| Dev QoS | `qos_gpu-dev` | `qos_gpu_a100-dev` | `qos_gpu_h100-dev` |
| Standard QoS | `qos_gpu-t3` | `qos_gpu_a100-t3` | `qos_gpu_h100-t3` |
| Long QoS | `qos_gpu-t4` | *(none)* | `qos_gpu_h100-t4` |

**H100 is the better default** where available: a long QoS exists, 80 GB per GPU, and more nodes. A100
caps at 20 hours, which for a multi-day campaign means chaining jobs.

**Precision follows the card.** A100 and H100 do bf16; **V100 does not** — use `16-mixed` there.
`jobs/env/jeanzay.sh` sets a default; override per partition.

### H.2 Smoke test, then the run

```bash
srun -A <proj>@v100 --qos=qos_gpu-dev --gres=gpu:1 --cpus-per-task=10 \
     --time=00:30:00 --hint=nomultithread --pty bash
cd $WORK/oceanml3d-core
jobs/run.sh --site jeanzay experiment=osse3d_smoke training=debug
```

Then:

```bash
sbatch --export=ALL,OCEANML3D_SITE=jeanzay jobs/slurm/train.sbatch \
       experiment=osse3d_gs21_multivar_unet \
       hydra.run.dir=$SCRATCH/oceanml3d/runs/$(date +%Y%m%d-%H%M%S)
```

**`hydra.run.dir` is not optional.** By default Hydra writes `outputs/<date>/<time>/` relative to the
working directory — i.e. into `$WORK`, whose inode quota is shared across the project, and the job
dies on quota before the first epoch with an error mentioning only `os.mkdir`. Compute the timestamp
in the shell, not with Hydra's `${now:…}`, which bash would try to expand.

Fill the account and QoS into `jobs/slurm/train.sbatch`:

```bash
#SBATCH --account=<proj>@h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH -C h100
#SBATCH --hint=nomultithread
```

Monitor with `squeue -u $USER` and `tail -f logs/oceanml3d-<jobid>.out`. For a full traceback inside
the container, `SINGULARITYENV_HYDRA_FULL_ERROR=1` — the prefix is what carries a variable across the
container boundary.

### H.3 Multi-GPU

`--gres=gpu:4`, `--ntasks-per-node=4`, and `training=ddp`. Lightning reads Slurm's variables itself:
do **not** also wrap the command in `srun`, or every rank relaunches the whole script.

---

## I. Inference

```bash
sbatch --export=ALL,OCEANML3D_SITE=jeanzay jobs/slurm/train.sbatch \
       command=predict experiment=osse3d_gs21_multivar_unet \
       ckpt=$SCRATCH/oceanml3d/runs/<run>/checkpoints/<best>.ckpt
```

Then archive the product before the thirty-day purge.

---

## Jean Zay-specific pitfalls

| Symptom | Cause |
|---|---|
| `Multiple accounts available` | `-A <proj>@<partition>` is missing |
| A download hangs or is killed | you are on a login node, or a compute node with no network. Use `--partition=prepost`. |
| `singularity: command not found`, or the image will not run | login nodes do not run containers; and the image must be in `$SINGULARITY_ALLOWED_DIR` via `idrcontmgr` |
| Quota exceeded before the first epoch | Hydra wrote `outputs/` into `$WORK`; set `hydra.run.dir` to `$SCRATCH` |
| Inode quota exceeded, plenty of bytes free | too many small files. Aggregate; archive to `$STORE` as one tar. |
| Data gone | `$SCRATCH`, thirty days without access |
| `bf16` errors on V100 | Volta has no bf16. `training.trainer.precision=16-mixed`. |
| A job capped at 20 hours | the A100 partition. Use H100, or chain jobs. |
