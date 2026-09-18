#!/usr/bin/env bash
#SBATCH --job-name=o3d_glorys
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --array=2010-2020
#SBATCH --output=logs/%x-%A_%a.out
## Jean Zay: a pre/post node, since this reads a lot and may need network depending on GLORYS_SRC.
##SBATCH --account=xxx@cpu
##SBATCH --partition=prepost
#
# The Slurm twin of jobs/pbs/prepare_glorys.pbs: one year per array task, for the same reason.
#     sbatch --export=ALL,OCEANML3D_SITE=jeanzay jobs/slurm/prepare_glorys.sbatch
#     sbatch --array=2015 ...        # relaunch a single year that overran
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
export OCEANML3D_SITE="${OCEANML3D_SITE:-jeanzay}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
exec jobs/prepare.sh glorys "${SLURM_ARRAY_TASK_ID:?submit with --array=<first>-<last>}"
