# oceanml3d-core

4DVarNet and flow matching for 3D multivariate ocean data assimilation and reconstruction.

Derived from [`4dvarnet-fm-opencode`](https://github.com/CIA-Oceanix/4dvarnet-fm-opencode) and
[`NOSC`](https://github.com/aitmohandk/NOSC), and independent of both — see `PROVENANCE.md`.

## Install

```bash
pip install -e .             # library + CLI
pip install -e '.[dev]'      # + pytest, ruff
pip install -e '.[plots]'    # + matplotlib, for the report scripts
pip install -e '.[zarr]'     # + zarr, if your catalog points at .zarr stores
pip install -e '.[prepare]'  # + copernicusmarine, cdsapi, xesmf, for scripts/prepare/
pip install -e '.[monai]'    # the MONAI backbone; pulls torch forward, give it its own environment
```

The wheel ships the `oceanml3d/` package and the `oceanml3d` console script. The Hydra YAML tree
(`config/`) and the driver scripts (`train.py`, `eval_*.py`, `run_experiment*.py`) stay at the
repository root, because `@hydra.main(config_path=…)` resolves relative to the file carrying the
decorator: an installed distribution gives you the library, run the drivers from a clone.

## Layout

```
oceanml3d/
  catalog.py, variables.py, registry.py, config_schema.py, cli.py
  data/        Lorenz-63/96 and QG simulation, datasets, dataloaders, normalisation,
               and the lazy xarray layer (open, patches, datamodule, transforms, augment)
  dynamics/    Dynamics base + registry (RK4 Lorenz-63/96, two-scale)
  models/      the toy / Lorenz / QG family, built through factory.py
    ocean/     the gridded-ocean family, built through the @register_model registry:
               nosc/, fourdvarnet/, baselines/, assimilation/, nn/
  obs/         OSSE observation operators — repeat orbits, per-mission masks, virtual Argo
  training/    Lightning modules, losses, staged training, loss grouping, vertical modes
  evaluation/  EnKF / ETKF / strong- and weak-4D-Var baselines, joint variants, metrics, drivers
  inference/   prediction, export, and the shared product_contract.py
  conf/        typed Hydra schemas for the toy path
config/        YAML presets, two roots: config.yaml (train.py) and main.yaml (the CLI)
train.py, eval_*.py, run_experiment*.py         drivers
tests/         the suite            reports/, batch/, scripts/, demos/, notebooks/
```

Import by full path: `from oceanml3d.models.solver import TweedieSolver`.

**Two model families, deliberately.** `oceanml3d/models/*.py` is the toy / Lorenz / QG code, built
by `models/factory.py` and trained by `train.py`. `oceanml3d/models/ocean/` is the gridded-ocean
code, built by the `@register_model` registry and trained by the `oceanml3d` CLI. They are not
interchangeable — `models/fourdvarnet.py` and `models/ocean/fourdvarnet/` are different models that
share a name. `docs/feature_inventory.md` maps every capability to its file.

## Run

```bash
python train.py experiment=<name>          # Hydra, see config/experiment/ (78 presets)
oceanml3d --help                           # the registry-based gridded path
pytest -q -m "not slow"                    # 779 passed, 11 skipped, ~13 min
ruff check .
```

## Companion repository

[`oceanml3d-eval`](https://github.com/aitmohandk/oceanml3d-eval) scores the products this repository
produces. The two share `product_contract.py` byte-for-byte, sealed by a SHA-256 pin on both
sides — change it in one repository only and the cross-repo contract test fails in the other.

`AGENTS.md` holds the working conventions, `PLAN.md` the state of the work,
`docs/feature_inventory.md` what came from where, `docs/worktrees.md` the topic-worktree workflow.
