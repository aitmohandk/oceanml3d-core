#!/usr/bin/env bash
# The work. No scheduler directives, no site paths, no module loads -- those live in
# jobs/<scheduler>/*.{sbatch,pbs} and jobs/env/<site>.sh respectively.
#
# This separation is the point. There were 166 SLURM scripts in batch/, 113 of them hard-coding one
# user's repository path and 81 hard-coding that user's conda environment, and none of them usable
# on a PBS site. Adding a second scheduler by duplicating them would have made 332. Here a scheduler
# is ~15 lines of directives around this file, and a site is a handful of exports.
#
# Usage, directly or from a job wrapper:
#     OCEANML3D_SITE=datarmor jobs/run.sh experiment=nosc_15m_duacs
#     OCEANML3D_SITE=local    jobs/run.sh command=validate experiment=osse3d_gs21_multivar_unet
#
# Environment:
#   OCEANML3D_SITE   name of a file in jobs/env/ (default: local)
#   OCEANML3D_SIF    path to an Apptainer image; if set, the command runs inside it
#   OCEANML3D_DATA   data root, consumed by config/paths/<site>.yaml
#   OCEANML3D_EXTRA  extra Hydra overrides appended after "$@"
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SITE="${OCEANML3D_SITE:-local}"
ENV_FILE="$REPO_ROOT/jobs/env/$SITE.sh"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "no environment for site '$SITE'." >&2
    echo "available: $(cd "$REPO_ROOT/jobs/env" && ls *.sh | sed 's/\.sh$//' | tr '\n' ' ')" >&2
    echo "copy jobs/env/local.sh to jobs/env/$SITE.sh and fill it in." >&2
    exit 2
fi

# shellcheck source=/dev/null
source "$ENV_FILE"

cd "$REPO_ROOT"
# Hydra resolves config_path relative to the file carrying @hydra.main, so the CLI must run from a
# clone even when the package itself is installed. That is also why the container binds the repo.
OVERRIDES=("$@")
if [[ -n "${OCEANML3D_EXTRA:-}" ]]; then
    # shellcheck disable=SC2206
    OVERRIDES+=(${OCEANML3D_EXTRA})
fi
OVERRIDES+=("paths=${OCEANML3D_PATHS:-$SITE}")

echo "[run.sh] site=$SITE  repo=$REPO_ROOT"
echo "[run.sh] data=${OCEANML3D_DATA:-<unset>}  sif=${OCEANML3D_SIF:-<none>}"
echo "[run.sh] oceanml3d ${OVERRIDES[*]}"

if [[ -n "${OCEANML3D_SIF:-}" ]]; then
    BINDS=("--bind" "$REPO_ROOT:$REPO_ROOT")
    [[ -n "${OCEANML3D_DATA:-}" ]] && BINDS+=("--bind" "$OCEANML3D_DATA:$OCEANML3D_DATA")
    [[ -n "${OCEANML3D_BIND:-}" ]] && BINDS+=("--bind" "$OCEANML3D_BIND")
    # --nv is not optional: without it the container sees no driver, and torch reports
    # cuda.is_available() == False without any other complaint.
    exec "${OCEANML3D_CONTAINER_CMD:-apptainer}" exec --nv "${BINDS[@]}" --pwd "$REPO_ROOT" \
        "$OCEANML3D_SIF" oceanml3d "${OVERRIDES[@]}"
else
    exec oceanml3d "${OVERRIDES[@]}"
fi
