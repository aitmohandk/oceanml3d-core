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
    local binds=("--bind" "$root:$root")
    [[ -n "${OCEANML3D_DATA:-}" ]] && binds+=("--bind" "$OCEANML3D_DATA:$OCEANML3D_DATA")
    [[ -n "${OCEANML3D_BIND:-}" ]] && binds+=("--bind" "$OCEANML3D_BIND")
    "${OCEANML3D_CONTAINER_CMD:-apptainer}" exec --nv "${binds[@]}" --pwd "$root" \
        "$OCEANML3D_SIF" "$@"
}

# Source a site file by name.
load_site() {
    local site="${1:?load_site <name> <repo_root>}" root="${2:?load_site <name> <repo_root>}"
    local env_file="$root/jobs/env/$site.sh"
    if [[ ! -f "$env_file" ]]; then
        echo "no environment for site '$site'." >&2
        echo "available: $(cd "$root/jobs/env" && ls ./*.sh | sed 's|.*/||; s/\.sh$//' | grep -v '^_' | tr '\n' ' ')" >&2
        echo "copy jobs/env/local.sh to jobs/env/$site.sh and fill it in." >&2
        return 2
    fi
    # shellcheck source=/dev/null
    source "$env_file"
}
