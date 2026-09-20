#!/usr/bin/env bash
# One preparation step, whatever the scheduler. The scheduler files under jobs/pbs/ and
# jobs/slurm/ hold directives and call this; everything they used to carry inline lives here.
#
#     jobs/prepare.sh --site datarmor glorys 2014     # one year of GLORYS truth
#     jobs/prepare.sh --site datarmor concat          # merge the per-year files
#     jobs/prepare.sh --site jeanzay  argo            # ARGO coverage table (needs network)
#     jobs/prepare.sh --site datarmor obs             # simulate the observing system
#
# Why a shared script rather than the body of each .pbs: a job file that carries its work cannot be
# run interactively to debug it, cannot be reused by the other scheduler, and -- under csh, which is
# Datarmor's default -- cannot even hold a multi-line quoted command. This file is bash, runs
# anywhere, and both schedulers call it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=jobs/env/_lib.sh
source "$REPO_ROOT/jobs/env/_lib.sh"

usage() {
    cat >&2 <<'USAGE'
usage: jobs/prepare.sh [--site <name>] [--res <deg>] <step> [args]

steps:
  glorys <year>   subset one year of GLORYS truth from the site's source into by_year/
  concat          merge the per-year files into the consolidated file the configs expect
  argo            ARGO coverage table: from the site's GDAC mirror if it has one
                  (argo_profiles_gs.<site>.yaml), else downloaded   (then needs network)
  obs             simulate the observing system: pseudo-obs and virtual ARGO
  all             argo, then obs -- the steps that do not need a year loop

environment:
  OCEANML3D_SITE     site name (or --site)
  OCEANML3D_DATA     data root, from the site file
  GLORYS_SRC         where GLORYS comes from. On Datarmor the read-only mirror
                     /home/ref-ocean-reanalysis/global-reanalysis-phy-001-030-daily -- there is no
                     reason to download what the centre already stores.
  ARGO_GDAC          local Argo GDAC mirror, read by argo_profiles_gs.<site>.yaml
  GLORYS_YEARS       first:last, used by `concat` for the output label (default 2010:2020)
  OCEANML3D_TARGET_RES  target grid step in degrees (or --res). Unset: native grid. Set, the data
                     root becomes $OCEANML3D_DATA/res<step> for every step, training included.
                     Give it to every job, or once in ~/.config/oceanml3d/env.sh.
USAGE
    exit 2
}

ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --site) OCEANML3D_SITE="$2"; shift 2 ;;
        --site=*) OCEANML3D_SITE="${1#--site=}"; shift ;;
        --res) export OCEANML3D_TARGET_RES="$2"; shift 2 ;;
        --res=*) export OCEANML3D_TARGET_RES="${1#--res=}"; shift ;;
        -h|--help) usage ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
set -- "${ARGS[@]+"${ARGS[@]}"}"
[[ $# -ge 1 ]] || usage

STEP="$1"; shift
SITE="${OCEANML3D_SITE:-local}"
load_site "$SITE" "$REPO_ROOT"
cd "$REPO_ROOT"

: "${OCEANML3D_DATA:?the site file must set OCEANML3D_DATA}"
RECIPES="scripts/prepare/recipes"
YEARS="${GLORYS_YEARS:-2010:2020}"

# The recipes are read through os.path.expandvars, so a template needs no sed: export the variables
# and the same file serves every year. That is also why GLORYS_SRC can point at a local mirror or at
# a download directory without the recipe changing.
export OCEANML3D_DATA

say() { echo "[prepare] $*"; }
say "site=$SITE  data=$OCEANML3D_DATA  resolution=${OCEANML3D_TARGET_RES:-native}"

step_glorys() {
    local year="${1:?glorys <year>}"
    export YEAR="$year" NEXT="$((year + 1))"
    local recipe="$RECIPES/glorys_gs_multidepth.$SITE.yaml"
    [[ -f "$recipe" ]] || recipe="$RECIPES/glorys_gs_multidepth.yaml"
    say "GLORYS $YEAR from ${GLORYS_SRC:-the input: in the recipe} -> $OCEANML3D_DATA/by_year" \
        "(resolution ${OCEANML3D_TARGET_RES:-native})"
    container_exec python scripts/prepare/regrid.py --config "$recipe"
}

step_concat() {
    export YEAR="${YEARS%%:*}" NEXT="${YEARS##*:}"
    local recipe="$RECIPES/glorys_gs_concat.yaml"
    say "merging $OCEANML3D_DATA/by_year/*.zarr -> $OCEANML3D_DATA/glorys (label $YEAR-$NEXT)"
    container_exec python scripts/prepare/regrid.py --config "$recipe"
}

step_argo() {
    local recipe="$RECIPES/argo_profiles_gs.$SITE.yaml"
    if [[ -f "$recipe" ]]; then
        say "ARGO profiles and coverage table from ${ARGO_GDAC:-the gdac_dir in $recipe}"
    else
        recipe="$RECIPES/argo_profiles_gs.yaml"
        say "ARGO profiles and coverage table, downloaded with argopy (needs outbound network)"
    fi
    container_exec python scripts/prepare/argo_profiles.py --config "$recipe"
}

step_obs() {
    say "simulating the observing system"
    container_exec oceanml3d command=prepare-obs \
        experiment="${OCEANML3D_EXPERIMENT:-osse3d_gs21_multivar_unet}" "paths=${OCEANML3D_PATHS:-$SITE}"
}

case "$STEP" in
    glorys) step_glorys "$@" ;;
    concat) step_concat ;;
    argo)   step_argo ;;
    obs)    step_obs ;;
    all)    step_argo; step_obs ;;
    *) echo "unknown step '$STEP'" >&2; usage ;;
esac
say "done: $STEP"
