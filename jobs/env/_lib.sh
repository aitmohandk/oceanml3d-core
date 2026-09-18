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
