"""4DVarNet and flow-matching core for 3D multivariate ocean data assimilation.

The scientific code lives under this namespace so that the project is installable and its modules
cannot collide with the generic names they used to occupy at the repository root (`models`,
`data`, `training`, `evaluation`, `conf` — `data` in particular shadowed any other `data` package
on the path).

Subpackages:
- `models`  — networks, solvers and the Lorenz-63/96 and quasi-geostrophic dynamics
- `data`    — datasets, dataloaders and normalisation
- `training` — Lightning modules, losses and the staged training pipeline
- `evaluation` — baselines (EnKF/ETKF/4D-Var), metrics and the evaluation drivers
- `conf`    — the typed configuration schema (the Hydra YAML tree stays in `config/`)
"""

__all__ = ["conf", "data", "evaluation", "models", "training"]
