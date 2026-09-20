# Site: Datarmor (Ifremer). PBS Pro, Tesla V100-PCIE-32GB, driver 530.30.02 / CUDA 12.1.
#
# The catalog is config/paths/datarmor.yaml, rooted at OCEANML3D_DATA below. Check that root is
# where you want the data before the first preparation job: it is $DATAWORK by default, and
# $SCRATCH is purged after ten days.

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
load_modules singularity || return 1   # or `apptainer`, depending on what the centre exposes

export OCEANML3D_DATA="${OCEANML3D_DATA:-$DATAWORK/oceanml3d}"
export OCEANML3D_PATHS="${OCEANML3D_PATHS:-datarmor}"
export OCEANML3D_SIF="${OCEANML3D_SIF:-$DATAWORK/containers/oceanml3d.sif}"
export OCEANML3D_CONTAINER_CMD="${OCEANML3D_CONTAINER_CMD:-singularity}"
# /home/ref-ocean-reanalysis holds Datarmor's read-only CMEMS mirror, GLORYS12V1 included. Bound
# in so the preparation recipes can read it directly instead of downloading tens to hundreds of
# gigabytes that are already on the filesystem. Without the bind the path resolves to nothing
# inside the container, which looks like a missing dataset rather than a missing mount.
export OCEANML3D_BIND="${OCEANML3D_BIND:-$DATAWORK:$DATAWORK,$SCRATCH:$SCRATCH,/home/ref-ocean-reanalysis,/home/ref-argo}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# Where GLORYS comes from. Datarmor mirrors the central CMEMS products read-only, GLORYS12V1
# (product 001-030) included, so there is nothing to download. Confirm the exact directory with
# `ls /home/ref-ocean-reanalysis/` -- the product name is the part that moves.
export GLORYS_SRC="${GLORYS_SRC:-/home/ref-ocean-reanalysis/global-reanalysis-phy-001-030-daily}"
export GLORYS_YEARS="${GLORYS_YEARS:-2010:2020}"

# Argo: the native GDAC is mirrored too (Ifremer hosts Coriolis, one of the two Argo GDACs), so the
# coverage table is built locally on a CPU queue -- no argopy, no `ftp` queue.
export ARGO_GDAC="${ARGO_GDAC:-/home/ref-argo/gdac}"

# V100 does fp16 but NOT bf16. `bf16-mixed` fails here; `16-mixed` is the one that works.
export OCEANML3D_EXTRA="${OCEANML3D_EXTRA:-training.trainer.precision=16-mixed}"
