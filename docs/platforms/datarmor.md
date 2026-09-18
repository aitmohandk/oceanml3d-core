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

Nothing in the code changes. `regrid.py` reads whatever `input:` points at, so a Datarmor variant of
the recipe is the whole difference:

```yaml
# scripts/prepare/recipes/glorys_gs_multidepth.datarmor.yaml
input: /home/ref-ocean-reanalysis/global-reanalysis-phy-001-030-daily/*/*.nc
output: ${OCEANML3D_DATA}/by_year/glorys_gs_multidepth_2010.nc
variables: {thetao: thetao, uo: uo, vo: vo, zos: zos}
keep_depth: true
method: none                 # native 1/12 deg kept; set a grid here to coarsen
time: ["2010-01-01", "2011-01-01"]
```

The `download:` block is simply absent. **One thing to get right:** the mirror must be bound into the
container, or the path resolves to nothing inside it. Add it in `jobs/env/datarmor.sh`:

```sh
export OCEANML3D_BIND="$DATAWORK:$DATAWORK,$SCRATCH:$SCRATCH,/home/ref-ocean-reanalysis"
```

Time one month interactively before launching eleven years. It validates the output *and* gives you
the scale for sizing the walltime:

```bash
qsub -I -l walltime=00:30:00 -l mem=32g
source /usr/share/Modules/init/bash && module load singularity
cd $DATAWORK/oceanml3d-core
OCEANML3D_DATA=$SCRATCH/oceanml3d/data \
  singularity exec --bind $OCEANML3D_BIND $DATAWORK/containers/oceanml3d-2026-09.sif \
  python scripts/prepare/regrid.py --config /tmp/glorys_one_month.yaml
```

### F.2 Eleven years: a PBS job array, one year per sub-job

**Not one monolithic job.** A single job chains thousands of file opens on one core and saves nothing
along the way: one walltime overrun and everything is lost. One sub-job per year, each independent,
then a merge. A year that overruns is relaunched on its own (`qsub -J 2015-2015 …`) while the others
stay done — and `regrid.py` skips a completed output, so relaunching the array is safe too.

`jobs/pbs/prepare_glorys.pbs`:

```csh
#!/bin/csh
#PBS -N o3d_glorys
#PBS -q omp
#PBS -l select=1:ncpus=8:mem=32g
#PBS -l walltime=03:00:00
#PBS -J 2010-2020
#PBS -j oe
#PBS -o logs/

source /usr/share/Modules/init/csh    # mandatory in batch csh: defines `module`
module load singularity
cd $PBS_O_WORKDIR
setenv OCEANML3D_DATA $SCRATCH/oceanml3d/data

set YEAR = $PBS_ARRAY_INDEX
@ NEXT = $YEAR + 1
sed -e "s/__YEAR__/$YEAR/g" -e "s/__NEXT__/$NEXT/g" \
    scripts/prepare/recipes/glorys_gs_multidepth.datarmor.yaml > /tmp/glorys_$YEAR.yaml

singularity exec --bind $DATAWORK,$SCRATCH,/home/ref-ocean-reanalysis \
  $DATAWORK/containers/oceanml3d-2026-09.sif \
  python scripts/prepare/regrid.py --config /tmp/glorys_$YEAR.yaml
```

Then merge the per-year files into the two the configs expect — same tool, `input:` a glob over
`by_year/`, `output:` on `$DATAWORK` so the result survives the purge:

```bash
qsub -q omp -l select=1:ncpus=4:mem=32g -l walltime=04:00:00 jobs/pbs/concat_glorys.pbs
```

> **Why `.pbs` files calling a `.py`, rather than `python -c "…"` inline.** These jobs run under
> **csh**, which cannot carry a multi-line double-quoted string: an inline program fails with
> `Unmatched "`. A file sidesteps the quoting entirely. NOSC learned this the hard way.

### F.3 Downloads, when you do need them — queue `ftp`

ARGO profiles are not mirrored, and neither is anything else you might add. Those go on the only
queue with outbound network:

```bash
qsub -q ftp -l walltime=06:00:00 -l mem=32g jobs/pbs/prepare_argo.pbs
```

The body runs `scripts/prepare/argo_profiles.py`: real profiles, QC on the standard flags, vertical
interpolation, and the coverage table — where and when a float was, and how deep. The values are
discarded; F.4 replaces them with the truth.

If you ever do need GLORYS from CMEMS (a domain or period the mirror does not cover), the same queue
applies, `copernicusmarine login` once interactively first, and year by year.

### F.4 Simulate the observing system — no network needed

```bash
qsub -v OCEANML3D_SITE=datarmor,OCEANML3D_ARGS="command=prepare-obs experiment=osse3d_gs21_multivar_unet" \
     jobs/pbs/train.pbs
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
| `no file matches /home/ref-…` inside the container | the mirror is not in `OCEANML3D_BIND`; the path exists on the host and not in the container |
| `Unmatched "` from a `.pbs` job | csh cannot carry a multi-line double-quoted string. Call a `.py` file, never `python -c`. |
| A whole GLORYS job lost on walltime | one monolithic job saves nothing along the way. Use the `-J` array, one year per sub-job (F.2). |
