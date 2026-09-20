# Running on a cluster

Three layers, so that adding a scheduler or a site is a small file rather than a fork of everything:

| Layer | File | Answers |
|---|---|---|
| Scheduler | `jobs/slurm/*.sbatch`, `jobs/pbs/*.pbs` | how do I ask for resources? |
| Site | `jobs/env/<site>.sh` | where is the environment and the data? |
| Work | `jobs/run.sh`, `jobs/prepare.sh` | what do I actually run? |
| Shared | `jobs/env/_lib.sh` | loading modules, wrapping a command in the container |

The wrappers hold directives and nothing else; the site files hold `module load`, paths and thread
counts; `run.sh` composes the Hydra command and, if `OCEANML3D_SIF` is set, runs it in a container.

This replaces `batch/`, which held 166 Slurm scripts — 113 hard-coding one user's repository path,
81 hard-coding that user's conda environment, and none usable under PBS. Adding a second scheduler
that way would have made 332 files to keep in step.

## What `run.sh` is, and is not

**It is not a pipeline.** It runs **one** `oceanml3d` command, exactly the one you give it, and does
three small things around it: source `jobs/env/<site>.sh` (modules, data root, thread count), append
`paths=<site>` so the catalog matches, and — if `OCEANML3D_SIF` is set — run the command inside the
container with the right binds and `--nv`.

That is all. These two are the same command:

```bash
jobs/run.sh --site datarmor command=prepare-obs experiment=osse3d_gs21_multivar_unet

singularity exec --nv --bind $DATAWORK:$DATAWORK,$SCRATCH:$SCRATCH,/home/ref-ocean-reanalysis \
    $DATAWORK/containers/oceanml3d.sif \
    oceanml3d command=prepare-obs experiment=osse3d_gs21_multivar_unet paths=datarmor
```

The first is the second with the site's details filled in from one file instead of retyped. Nothing
is hidden: `run.sh` prints the command it is about to run.

So a full preparation is **several** calls, in order, each its own job — that is why the runbooks walk
through numbered steps rather than handing you one command. `run.sh` is what each of those steps
calls.

**It is not the only way in either.** The preparation scripts under `scripts/prepare/` are plain
Python and take a `--config`; they are not `oceanml3d` sub-commands, so the runbooks call them
through `singularity exec` directly. `run.sh` covers the CLI: `validate`, `prepare-obs`, `eofs`,
`train`, `predict`.

## Quick start

```bash
# locally, no scheduler
jobs/run.sh --site local command=validate experiment=osse3d_gs21_multivar_unet

# Slurm (Jean Zay, Odyssey)
sbatch --export=ALL,OCEANML3D_SITE=jeanzay jobs/slurm/train.sbatch experiment=nosc_15m_duacs

# PBS Pro (Datarmor)
qsub -v OCEANML3D_SITE=datarmor,OCEANML3D_ARGS="experiment=nosc_15m_duacs" jobs/pbs/train.pbs
```

`--site` rather than `OCEANML3D_SITE=…` in front of the command, because the two are not equivalent
everywhere: `VAR=value command` is bash syntax, and Datarmor's default login shell is csh, which
reads the whole first word as a command name. The flag works in any shell. The environment variable
still works where the syntax does, and is what the job wrappers set.

## What is in `jobs/`

| File | Does |
|---|---|
| `run.sh` | one `oceanml3d` CLI command: `validate`, `prepare-obs`, `eofs`, `train`, `predict` |
| `prepare.sh` | one preparation step: `glorys <year>`, `concat`, `argo`, `obs` |
| `pbs/*.pbs`, `slurm/*.sbatch` | directives only; each is ~10 lines around one of the two above |
| `env/<site>.sh` | modules, data root, thread count, container path, precision default |
| `env/_lib.sh` | `load_modules`, `container_exec`, `load_site` — shared, so `--nv` and the bind list are decided once |

A file in `env/` whose name starts with `_` is a shared library, not a site. The convention is
load-bearing in two places: `load_site` filters it out of the "available sites" list, and
`test_every_site_env_file_has_a_matching_paths_file_or_is_a_template` skips it rather than
demanding it declare `OCEANML3D_PATHS`. Keep the prefix if you add another.

Every job file has a twin in the other scheduler and they differ only in their directives, because
the work is not in them. That is what makes a second scheduler cheap, and it is also why you can
debug any of them interactively:

```bash
qsub -I -l walltime=00:30:00 -l mem=32g       # or srun --pty bash
jobs/prepare.sh --site datarmor glorys 2014
```

## Your own settings, for every job: `~/.config/oceanml3d/env.sh`

A job started by the scheduler does not run your login shell's rc files, so a `setenv` in `~/.cshrc`
never reaches it -- and repeating the same `-v`/`--export` on every submission is how one step ends
up looking in another's directory. `load_site` (`jobs/env/_lib.sh`) therefore sources a per-user file
**before** the site file, in every job and every interactive `run.sh` / `prepare.sh`:

```bash
mkdir -p ~/.config/oceanml3d
echo 'export OCEANML3D_TARGET_RES="${OCEANML3D_TARGET_RES:-0.25}"' > ~/.config/oceanml3d/env.sh
```

- **Nothing creates it.** It is optional and personal: absent, everything runs at the native
  resolution with the site's defaults. It never goes in the repository -- it would impose your
  choices on everyone else.
- **Bash syntax**, whatever your login shell: the jobs are bash scripts. The single-quoted `echo`
  above works from csh as well.
- **Write each value as `${VAR:-value}`**, so a value given on the command line (`qsub -v ...`,
  `sbatch --export=...`, `--res`) still wins over the file.
- **Another location:** `OCEANML3D_USER_ENV=/path/to/file`.
- **Check what a job used:** its first log line prints it --
  `[prepare] site=datarmor  data=.../oceanml3d/res0.25  resolution=0.25` (`[run.sh] ...` for training).
- **Back to native:** delete the file, or pass `OCEANML3D_TARGET_RES=native`.

What typically goes there:

| Variable | Effect |
|---|---|
| `OCEANML3D_TARGET_RES` | grid step in degrees for the GLORYS preparation; the data root becomes `$OCEANML3D_DATA/res<step>` for every step, training included. `native` or unset: 1/12 deg. |
| `OCEANML3D_DATA` | data root, if not the site's default (the `res<step>` suffix is still added) |
| `OCEANML3D_SIF` | container image, if not the site's default |
| `OCEANML3D_EXTRA` | Hydra overrides appended to every `run.sh` command |

## Adding a site

1. `cp jobs/env/local.sh jobs/env/<site>.sh` and fill in the environment and `OCEANML3D_DATA`.
2. `cp config/paths/local.yaml config/paths/<site>.yaml` and change the **paths**, not the keys —
   `test_a_site_file_does_not_invent_keys_of_its_own` enforces that, because a key only one site
   knows is either a typo or a key the others silently lack.
3. `jobs/run.sh --site <site> command=validate experiment=<xp>` before submitting anything.

## The container

`container/oceanml3d.def` builds one image for every target: `torch 2.6.0+cu124` covers sm_70 (V100)
through sm_90 (H100), and CUDA minor-version compatibility means a cu124 build runs on any driver
supporting CUDA 12.0 or later — Datarmor's 530.30.02 included. torch 2.6 is the ceiling because
`monai>=1.5,<1.6` pins `torch<2.7`.

```bash
apptainer build oceanml3d.sif container/oceanml3d.def
```

`--nv` is not optional when running it. Without that flag the container sees no driver, and torch
reports `cuda.is_available() == False` with no other complaint. `run.sh` passes it.

The repository is bound rather than baked in: Hydra resolves `config_path` relative to the file
carrying `@hydra.main`, so the CLI runs from a clone even though the package is installed.

## Precision by GPU

| GPU | Sites | `training.trainer.precision` |
|---|---|---|
| V100 (sm_70) | Datarmor, Jean Zay V100 | `16-mixed` — **fp16 only, no bf16** |
| RTX 8000 (sm_75) | Odyssey | `16-mixed` |
| A100 / H100 (sm_80/90) | Jean Zay | `bf16-mixed` |

The site files set a default; override on the command line for a different partition.

## Memory, for the gridded tasks

`osse3d_gs21` (144x144 patches) is comfortable anywhere. `surface_currents_15m` is not: its patch is
560x1440, and with the default MONAI trunk at ~224M parameters the estimate is roughly 3.6 GB of
weights, optimiser state and gradients plus 15-20 GB of activations at `precision: 32` — tight on a
32 GB V100 and untested at the time of writing. If it does not fit, in order: `16-mixed`, then
`model.num_res_blocks=1`, then one fewer level in `model.widths`.

Never put level 0 in `model.attention_levels` for that task: attention over 806,400 positions is not
a memory problem to be tuned around.

## Multi-GPU

Add `training=ddp` and raise the resource request. Lightning reads Slurm's variables itself — do not
also wrap the command in `srun`, or each rank relaunches the whole script again.

Two things are already handled: `prepare-obs` runs on one rank while the others wait for the files
(`oceanml3d/io.py`), and the export runs on rank 0 with a single-device Trainer, because
`trainer.predict` shards the dataloader and a field stitched from one rank's share is wrong without
looking wrong.
