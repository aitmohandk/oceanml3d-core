#!/usr/bin/env bash
#SBATCH --job-name=o3d_argo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
## Jean Zay: this downloads, so it MUST be a pre/post node -- compute nodes have no outbound
## network and login nodes do not run containers.
##SBATCH --account=xxx@cpu
##SBATCH --partition=prepost
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
export OCEANML3D_SITE="${OCEANML3D_SITE:-jeanzay}"
# Exit code 75 means the recipe's `time_budget_s` ran out with the per-float cache written: resubmit
# this script and it continues where it stopped (`--dependency=afterany:$SLURM_JOB_ID` to queue it
# now). Nothing is re-read.
exec jobs/prepare.sh argo
