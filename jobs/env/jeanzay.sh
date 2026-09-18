# Site: Jean Zay (IDRIS). Slurm, V100 / A100 / H100 depending on the partition.
#
# TO FILL IN: the account (`--account=xxx@v100`), and config/paths/jeanzay.yaml.
#
# Two IDRIS specifics to confirm against the current documentation before the first run, because
# both will stop a job dead and neither is a code problem:
#   * compute nodes have no outbound network, so nothing may download at run time. The container is
#     built elsewhere, and `scripts/prepare/` (which fetches from CMEMS/CDS) runs on a pre-post node.
#   * Singularity images are not run from an arbitrary path: they have to be registered in the
#     centre's image area first (`idrcontmgr`). Check the current procedure.

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
load_modules singularity || return 1

export OCEANML3D_DATA="${OCEANML3D_DATA:-$WORK/oceanml3d}"
export OCEANML3D_PATHS="${OCEANML3D_PATHS:-jeanzay}"
export OCEANML3D_SIF="${OCEANML3D_SIF:-$SINGULARITY_ALLOWED_DIR/oceanml3d.sif}"
export OCEANML3D_CONTAINER_CMD="${OCEANML3D_CONTAINER_CMD:-singularity}"
export OCEANML3D_BIND="${OCEANML3D_BIND:-$WORK:$WORK,$SCRATCH:$SCRATCH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"

# A100 and H100 support bf16, which is the better choice where available; on the V100 partition
# override this to 16-mixed.
export OCEANML3D_EXTRA="${OCEANML3D_EXTRA:-training.trainer.precision=bf16-mixed}"
