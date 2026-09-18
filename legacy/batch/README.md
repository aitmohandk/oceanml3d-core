# Reserve: inherited SLURM scripts

**Not maintained, not runnable as they stand.** Kept for the record — several encode campaigns whose
results are written up in `reports/`, and reading the script is sometimes the only way to recover
exactly what was run.

166 job scripts (137 `.sbatch`, 29 `.slurm`) inherited from `4dvarnet-fm-opencode`. Measured:

| | |
|---|---|
| Hard-coding `/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode` | 113 |
| Hard-coding another user's conda environment (`.../miniforge3/envs/fdv/bin`) | 81 |
| Containing inline Python with the flat imports that stopped resolving at the namespace move | 4 |
| Calling `evaluation/sweep_qg_baselines.py`, deleted in lot 4 | several |
| Mentioning `oceanml3d` at all | 8 |

They also assume Slurm, which rules out Datarmor (PBS Pro) and any other centre that is not Odyssey.

## What replaces them

`jobs/`, and its README. A scheduler is ~15 lines of directives around `jobs/run.sh`; a site is a
handful of exports in `jobs/env/<site>.sh` plus a `config/paths/<site>.yaml`. Duplicating what is in
this directory for a second scheduler would have produced 332 files to keep in step.

## If you need one of these

Do not repair it in place. Read it for the *parameters* — the Hydra overrides, the resource request,
the sweep ranges — and express those through `jobs/`. The paths and the environment in these files
describe one machine and one user's account, and neither exists for you.
