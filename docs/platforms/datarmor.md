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
* **Compute nodes have no Internet.** One queue does: **`ftp`**. Every download — GLORYS through
  `copernicusmarine`, ARGO through `argopy` — must be submitted there.

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

Transferring the image (several GB) depends on where you are, because the two dedicated hosts are
not routed from the same networks. **A `connect to host … port 22: Connection timed out` is not a
bad password — it means that host is not reachable from where you are**, typically `datacopy` from
outside even with the VPN.

| From | Host | Protocol | Credentials |
|---|---|---|---|
| An Ifremer building | `datacopy.ifremer.fr` | scp / sftp / rsync | intranet |
| Outside | `eftp.ifremer.fr` | FTP, exposes `$SCRATCH/eftp` | **extranet** (a different account) |
| Either, occasionally | `datarmor-access.ifremer.fr` | rsync | intranet |

```bash
rsync -avP oceanml3d-2026-09.sif <login>@datacopy.ifremer.fr:        # lands in $HOME
```

`530 Login incorrect` on `eftp` almost always means the extranet account: it is distinct from your
Datarmor login, Ifremer forbids sharing the password, and it may not be active. Created from
`teletravail.ifremer.fr`, password reset at `https://www.ifremer.fr/chpass/`. The third row avoids
it entirely — the access node is shared, so use it for occasional transfers, not routinely.

Park the image on `$DATAWORK`:

```bash
ssh datarmor
mkdir -p $DATAWORK/containers
mv ~/oceanml3d-2026-09.sif $DATAWORK/containers/
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

Before checking whether `/home/ref-ocean-reanalysis` already mirrors the GLORYS you need, look:
downloading fifty gigabytes that are already on the filesystem is a common waste.

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

```bash
cd $DATAWORK/oceanml3d-core
OCEANML3D_SITE=datarmor jobs/run.sh command=list-models
OCEANML3D_SITE=datarmor jobs/run.sh command=validate experiment=osse3d_gs21_multivar_unet
exit
```

`validate` reports every problem at once. Run it before every submission; it costs a second, and a
queued job failing on a typo costs a day.

---

## F. Prepare the data

### F.1 Downloads go on the `ftp` queue

The only nodes with outbound network. `jobs/pbs/prepare_download.pbs`:

```bash
#!/usr/bin/env bash
#PBS -N o3d_download
#PBS -q ftp
#PBS -l walltime=20:00:00
#PBS -l mem=16g
#PBS -j oe
#PBS -o logs/

source /usr/share/Modules/init/bash
module load singularity
cd $PBS_O_WORKDIR

# once, interactively, before the first run:
#   singularity exec $OCEANML3D_SIF copernicusmarine login
for y in $(seq 2010 2020); do
  echo "=== $y ==="
  sed "s/2010-01-01/${y}-01-01/; s/2020-01-11/${y}-12-31/" \
      scripts/prepare/recipes/glorys_gs_multidepth.yaml > /tmp/glorys_$y.yaml
  singularity exec --bind $DATAWORK,$SCRATCH $OCEANML3D_SIF \
      python scripts/prepare/download_copernicus.py --config /tmp/glorys_$y.yaml
done
```

`copernicusmarine login` once, interactively, or every year re-asks for credentials.

**Year by year, always.** Eleven years in one job is what exceeds walltime, and a per-year loop
resumes where it stopped.

### F.2 Subset and merge — GPU not needed, memory is

```bash
qsub -q sequentiel -l walltime=10:00:00 -l mem=64g jobs/pbs/prepare_regrid.pbs
```

The body runs `scripts/prepare/regrid.py` per year into `by_year/`, then once more with a glob to
merge. `regrid.py` is idempotent — a completed output is skipped — so a walltime kill is resumed by
resubmitting, not restarted. Its log is line-buffered, so `tail -f` on the `.o` file shows progress
rather than nothing until the end.

### F.3 ARGO coverage table — `ftp` queue again

```bash
qsub -q ftp -l walltime=06:00:00 -l mem=32g jobs/pbs/prepare_argo.pbs
```

Real profiles, QC on the standard flags, vertical interpolation, and the coverage table: where and
when a float was, and how deep. The values are discarded — F.4 replaces them with the truth.

### F.4 Simulate the observing system — no network

```bash
qsub -q sequentiel -l walltime=04:00:00 -l mem=64g -- \
     jobs/pbs/train.pbs   # with OCEANML3D_ARGS="command=prepare-obs experiment=osse3d_gs21_multivar_unet"
```

Writes `pseudo_obs_ssh_gs`, `pseudo_obs_sst_gs`, `argo_virtual_thetao_gs21`. **Those three keys must
already be in `config/paths/datarmor.yaml`** — they are outputs, but the simulator learns where to
write from the catalog.

### F.5 Keep the master copy

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
| `module: Command not found` in a job | batch does not read `~/.cshrc`; source the modules init first |
| `Connection timed out` on `datacopy` | that host is not routed from where you are, not a bad password |
| `530 Login incorrect` on `eftp` | the extranet account, which is not your Datarmor login |
| A job queued for hours | the `sequentiel` routing queue sent a large `mem=` request to rare nodes |
| Prepared data gone | `$SCRATCH`, ten days untouched |
| `torch.cuda.is_available()` is `False` | `--nv` missing |
| Quota exceeded before the first epoch | Hydra wrote `outputs/` into `$HOME` or `$DATAWORK`; set `hydra.run.dir` |
