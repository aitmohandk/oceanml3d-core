# Running on a cluster

Three layers, so that adding a scheduler or a site is a small file rather than a fork of everything:

| Layer | File | Answers |
|---|---|---|
| Scheduler | `jobs/slurm/*.sbatch`, `jobs/pbs/*.pbs` | how do I ask for resources? |
| Site | `jobs/env/<site>.sh` | where is the environment and the data? |
| Work | `jobs/run.sh` | what do I actually run? |

The wrappers hold directives and nothing else; the site files hold `module load`, paths and thread
counts; `run.sh` composes the Hydra command and, if `OCEANML3D_SIF` is set, runs it in a container.

This replaces `batch/`, which held 166 Slurm scripts — 113 hard-coding one user's repository path,
81 hard-coding that user's conda environment, and none usable under PBS. Adding a second scheduler
that way would have made 332 files to keep in step.

## Quick start

```bash
# locally, no scheduler
OCEANML3D_SITE=local jobs/run.sh command=validate experiment=osse3d_gs21_multivar_unet

# Slurm (Jean Zay, Odyssey)
sbatch --export=ALL,OCEANML3D_SITE=jeanzay jobs/slurm/train.sbatch experiment=nosc_15m_duacs

# PBS Pro (Datarmor)
qsub -v OCEANML3D_SITE=datarmor,OCEANML3D_ARGS="experiment=nosc_15m_duacs" jobs/pbs/train.pbs
```

## Adding a site

1. `cp jobs/env/local.sh jobs/env/<site>.sh` and fill in the environment and `OCEANML3D_DATA`.
2. `cp config/paths/local.yaml config/paths/<site>.yaml` and change the **paths**, not the keys —
   `test_a_site_file_does_not_invent_keys_of_its_own` enforces that, because a key only one site
   knows is either a typo or a key the others silently lack.
3. `OCEANML3D_SITE=<site> jobs/run.sh command=validate experiment=<xp>` before submitting anything.

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
