#!/usr/bin/env bash
#SBATCH --job-name=o3d_obs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
##SBATCH --account=xxx@cpu
#
# Simulates the observing system. Downloads nothing, so any node will do.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
export OCEANML3D_SITE="${OCEANML3D_SITE:-jeanzay}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
exec jobs/prepare.sh obs
