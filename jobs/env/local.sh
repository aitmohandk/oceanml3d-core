# Site: a workstation or a laptop. The template for every other file here.
#
# A site file answers three questions and nothing else:
#   1. how do I get a working environment?   (module load / conda activate / a .sif)
#   2. where is the data?                    (OCEANML3D_DATA, consumed by config/paths/<site>.yaml)
#   3. how many threads per process?         (centres size cores differently)
# Anything about *what to run* belongs in a Hydra override, not here.

export OCEANML3D_DATA="${OCEANML3D_DATA:-$HOME/data/oceanml3d}"
export OCEANML3D_PATHS="${OCEANML3D_PATHS:-local}"   # which config/paths/<name>.yaml to use
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# Either activate an environment built from environment.yml / environment-cpu.yml...
# conda activate oceanml3d
# ...or point at a container and let run.sh use it:
# export OCEANML3D_SIF="$HOME/containers/oceanml3d.sif"
