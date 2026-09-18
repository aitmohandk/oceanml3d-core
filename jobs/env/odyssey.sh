# Site: Odyssey (IMT Atlantique). Slurm, Quadro RTX 8000 (48 GB, sm_75), partition Odyssey_GPU.
#
# This is where the inherited batch/ scripts were written for, against a conda env belonging to one
# user. Either keep using that environment by activating it below, or build the container once and
# point OCEANML3D_SIF at it.

export OCEANML3D_DATA="${OCEANML3D_DATA:-/Odyssey/private/$USER/data/oceanml3d}"
export OCEANML3D_PATHS="${OCEANML3D_PATHS:-odyssey}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# Option A -- conda, built from environment.yml:
# source "$HOME/miniforge3/etc/profile.d/conda.sh"
# conda activate oceanml3d
#
# Option B -- container:
# module load singularity
# export OCEANML3D_SIF="${OCEANML3D_SIF:-$HOME/containers/oceanml3d.sif}"
# export OCEANML3D_CONTAINER_CMD=singularity

# RTX 8000 is Turing: fp16 yes, bf16 no.
export OCEANML3D_EXTRA="${OCEANML3D_EXTRA:-training.trainer.precision=16-mixed}"
