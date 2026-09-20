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
#     jobs/run.sh --site datarmor experiment=nosc_15m_duacs
#     jobs/run.sh --site local command=validate experiment=osse3d_gs21_multivar_unet
#
# Environment:
#   OCEANML3D_SITE   name of a file in jobs/env/ (default: local)
#
# `--site <name>` does the same as OCEANML3D_SITE and does not depend on the shell. That matters:
# Datarmor's default login shell is csh, where `VAR=value command` is not a thing -- it reads the
# whole first word as a command name and answers `OCEANML3D_SITE=datarmor: Command not found.`
# In csh the alternatives are `setenv OCEANML3D_SITE datarmor` first, or `env VAR=value ...`;
# `--site` avoids having to know which shell you are in.
#   OCEANML3D_SIF    path to an Apptainer image; if set, the command runs inside it
#   OCEANML3D_DATA   data root, consumed by config/paths/<site>.yaml
#   OCEANML3D_TARGET_RES  (or --res) resolution the data was prepared at: data root becomes
#                    $OCEANML3D_DATA/res<step>, the directory jobs/prepare.sh wrote
#   OCEANML3D_EXTRA  extra Hydra overrides appended after "$@"
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --site <name> before any Hydra override; everything else is passed through untouched.
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --site) OCEANML3D_SITE="$2"; shift 2 ;;
        --site=*) OCEANML3D_SITE="${1#--site=}"; shift ;;
        --res) export OCEANML3D_TARGET_RES="$2"; shift 2 ;;
        --res=*) export OCEANML3D_TARGET_RES="${1#--res=}"; shift ;;
        --) shift; ARGS+=("$@"); break ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

SITE="${OCEANML3D_SITE:-local}"
# shellcheck source=jobs/env/_lib.sh
source "$REPO_ROOT/jobs/env/_lib.sh"
# Same loader as jobs/prepare.sh: user settings, site file, then the per-resolution data dir -- so
# training reads exactly the directory the preparation wrote.
load_site "$SITE" "$REPO_ROOT" || exit 2

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
echo "[run.sh] data=${OCEANML3D_DATA:-<unset>}  resolution=${OCEANML3D_TARGET_RES:-native}  sif=${OCEANML3D_SIF:-<none>}"
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
