#!/usr/bin/env bash
#SBATCH --job-name=oceanml3d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
## Site-specific, uncomment and fill in:
##SBATCH --account=xxx@v100        # Jean Zay
##SBATCH --partition=Odyssey_GPU   # Odyssey
##SBATCH --qos=qos_gpu-t3          # Jean Zay
#
# Directives, then hand over. Everything below this line is common to both schedulers.
#
#     sbatch --export=OCEANML3D_SITE=jeanzay jobs/slurm/train.sbatch experiment=nosc_15m_duacs
#
# For multi-GPU, raise --gres/--ntasks-per-node and add training=ddp to the overrides. Lightning
# reads Slurm's variables by itself; do not wrap the command in srun as well, or every rank
# relaunches the whole thing twice over.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
export OCEANML3D_SITE="${OCEANML3D_SITE:-odyssey}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "[slurm] job ${SLURM_JOB_ID:-?} on ${SLURMD_NODENAME:-?}, ${SLURM_GPUS_ON_NODE:-?} GPU(s)"
exec jobs/run.sh "$@"
