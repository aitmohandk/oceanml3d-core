#!/usr/bin/env bash
#SBATCH --job-name=o3d_concat
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
##SBATCH --account=xxx@cpu
##SBATCH --partition=prepost
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
export OCEANML3D_SITE="${OCEANML3D_SITE:-jeanzay}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
exec jobs/prepare.sh concat
