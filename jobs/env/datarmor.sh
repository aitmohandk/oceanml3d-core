# Site: Datarmor (Ifremer). PBS Pro, Tesla V100-PCIE-32GB, driver 530.30.02 / CUDA 12.1.
#
# TO FILL IN: OCEANML3D_DATA, and config/paths/datarmor.yaml with this centre's paths. The catalog
# keys are the sixteen in config/paths/local.yaml; copy that file and change the paths, not the keys
# (test_a_site_file_does_not_invent_keys_of_its_own enforces exactly that).

module purge
module load singularity          # or `apptainer`, depending on what the centre exposes

export OCEANML3D_DATA="${OCEANML3D_DATA:-$DATAWORK/oceanml3d}"
export OCEANML3D_PATHS="${OCEANML3D_PATHS:-datarmor}"
export OCEANML3D_SIF="${OCEANML3D_SIF:-$DATAWORK/containers/oceanml3d.sif}"
export OCEANML3D_CONTAINER_CMD="${OCEANML3D_CONTAINER_CMD:-singularity}"
export OCEANML3D_BIND="${OCEANML3D_BIND:-$DATAWORK:$DATAWORK,$SCRATCH:$SCRATCH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# V100 does fp16 but NOT bf16. `bf16-mixed` fails here; `16-mixed` is the one that works.
export OCEANML3D_EXTRA="${OCEANML3D_EXTRA:-training.trainer.precision=16-mixed}"
