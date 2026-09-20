# Shared helpers for the site files. Sourced by jobs/env/<site>.sh, never run on its own.

# `module` is a shell function, not a program. Your login shell defines it by sourcing the Modules
# init from its rc file; a script started with `#!/usr/bin/env bash` gets a fresh shell where the
# function does not exist, and `module load singularity` dies with:
#
#     jobs/env/datarmor.sh: line 7: module: command not found
#
# Same cause as the csh batch trap already documented for .pbs jobs -- it just bites interactively
# too. This finds the init for the running shell and sources it, and is a no-op when `module` is
# already defined (an interactive login shell, or a job that sourced the init itself).
load_modules() {
    if ! type module >/dev/null 2>&1; then
        local init
        for init in "${MODULESHOME:+$MODULESHOME/init/bash}" \
                    /usr/share/Modules/init/bash \
                    /usr/share/lmod/lmod/init/bash \
                    /etc/profile.d/modules.sh \
                    /etc/profile.d/lmod.sh; do
            if [[ -n "$init" && -r "$init" ]]; then
                # shellcheck source=/dev/null
                source "$init" && break
            fi
        done
    fi
    if ! type module >/dev/null 2>&1; then
        echo "[oceanml3d] the 'module' function is not available and no Modules init was found." >&2
        echo "[oceanml3d] looked in: \$MODULESHOME/init/bash, /usr/share/Modules/init/bash," >&2
        echo "[oceanml3d]            /usr/share/lmod/lmod/init/bash, /etc/profile.d/{modules,lmod}.sh" >&2
        echo "[oceanml3d] find yours with: ls /usr/share/Modules/init/ ; ls /etc/profile.d/*module*" >&2
        echo "[oceanml3d] then set MODULESHOME, or source it in jobs/env/$SITE.sh before load_modules." >&2
        return 1
    fi
    [[ $# -gt 0 ]] && module load "$@"
}

# Run a command in the site's container, or directly when no image is configured. Shared by
# jobs/run.sh and jobs/prepare.sh so the bind list and `--nv` are decided in exactly one place -- a
# forgotten --nv is the most common way to conclude the GPU is broken.
container_exec() {
    if [[ -z "${OCEANML3D_SIF:-}" ]]; then
        "$@"
        return
    fi
    local root="${REPO_ROOT:-$PWD}"
    local binds=()

    # OCEANML3D_DATA is an *output* directory: on a first run it does not exist yet, and Singularity
    # refuses to start at all rather than creating it --
    #   FATAL: mount source /.../oceanml3d doesn't exist
    # which reads like a configuration error when it is only a missing mkdir.
    if [[ -n "${OCEANML3D_DATA:-}" ]]; then
        mkdir -p "$OCEANML3D_DATA" || {
            echo "[oceanml3d] cannot create OCEANML3D_DATA=$OCEANML3D_DATA" >&2
            return 1
        }
    fi

    # Every bind must exist on the host. A path that does not is skipped with a reason, because one
    # typo in OCEANML3D_BIND otherwise takes the whole job down with a message about mounting.
    local spec src
    for spec in "$root:$root" "${OCEANML3D_DATA:+$OCEANML3D_DATA:$OCEANML3D_DATA}" \
                ${OCEANML3D_BIND:+${OCEANML3D_BIND//,/ }}; do
        [[ -n "$spec" ]] || continue
        src="${spec%%:*}"
        if [[ -e "$src" ]]; then
            binds+=("--bind" "$spec")
        else
            echo "[oceanml3d] skipping bind '$spec': $src does not exist on this host" >&2
        fi
    done

    # --nv only where there is a driver to expose. On a CPU queue -- Datarmor's `omp`, used for the
    # preparation jobs -- it prints "Could not find any nv files on this host!", which looks like a
    # failure in a log that is otherwise about data.
    local nv=()
    if [[ "${OCEANML3D_NV:-auto}" == "1" ]] || { [[ "${OCEANML3D_NV:-auto}" == "auto" ]] &&
            { [[ -e /dev/nvidiactl ]] || command -v nvidia-smi >/dev/null 2>&1; }; }; then
        nv=("--nv")
    fi

    "${OCEANML3D_CONTAINER_CMD:-apptainer}" exec "${nv[@]+"${nv[@]}"}" "${binds[@]}" --pwd "$root" \
        "$OCEANML3D_SIF" "$@"
}

# One data directory per resolution, derived rather than typed. File names do not carry the
# resolution, so native and 0.25-degree products must not share a directory -- and asking for
# `OCEANML3D_DATA=.../res0.25` on every qsub meant one forgotten step looked in the wrong place:
#     FileNotFoundError: no file matches .../oceanml3d/by_year/glorys_gs_multidepth_*.zarr
# (the per-year stores were in .../oceanml3d/res0.25/by_year). Now OCEANML3D_TARGET_RES alone
# decides: every step -- preparation, merge, ARGO, obs, training -- appends /res<step> to the site's
# data root. Idempotent: a root that already ends in /res<step> is left alone.
resolve_data_dir() {
    local res="${OCEANML3D_TARGET_RES:-}"
    [[ -n "$res" && "$res" != "native" && -n "${OCEANML3D_DATA:-}" ]] || return 0
    case "${OCEANML3D_DATA%/}" in
        */res"$res") ;;
        *) export OCEANML3D_DATA="${OCEANML3D_DATA%/}/res$res" ;;
    esac
}

# Settings that should hold for every job without repeating them on each qsub/sbatch line --
# typically OCEANML3D_TARGET_RES. A job started by the scheduler does not inherit your shell's
# environment, so a `setenv` in ~/.cshrc is not enough. Write them with a default, so a value
# given on the command line (`qsub -v ...`) still wins:
#     export OCEANML3D_TARGET_RES="${OCEANML3D_TARGET_RES:-0.25}"
OCEANML3D_USER_ENV="${OCEANML3D_USER_ENV:-$HOME/.config/oceanml3d/env.sh}"

# Source a site file by name: the user's settings, the site's defaults, then the derived data dir.
load_site() {
    local site="${1:?load_site <name> <repo_root>}" root="${2:?load_site <name> <repo_root>}"
    local env_file="$root/jobs/env/$site.sh"
    if [[ ! -f "$env_file" ]]; then
        echo "no environment for site '$site'." >&2
        echo "available: $(cd "$root/jobs/env" && ls ./*.sh | sed 's|.*/||; s/\.sh$//' | grep -v '^_' | tr '\n' ' ')" >&2
        echo "copy jobs/env/local.sh to jobs/env/$site.sh and fill it in." >&2
        return 2
    fi
    if [[ -f "$OCEANML3D_USER_ENV" ]]; then
        # shellcheck source=/dev/null
        source "$OCEANML3D_USER_ENV"
    fi
    # shellcheck source=/dev/null
    source "$env_file"
    resolve_data_dir
}
