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
```

MONAI is a plain dependency (`monai>=1.5,<1.6`): `DiffusionModelUNet` is the default trunk of the
gridded models. The `<1.6` ceiling is deliberate — monai 1.6 requires torch>=2.8, while 1.5.x
installs next to torch 2.4 without moving it.

The wheel ships the `oceanml3d/` package and the `oceanml3d` console script. The Hydra YAML tree
(`config/`) and the driver scripts stay in the clone, because `@hydra.main(config_path=…)` resolves
relative to the file carrying the decorator: an installed distribution gives you the library, run
the drivers from a clone.

## Layout

```
oceanml3d/
  catalog.py, variables.py, registry.py, config_schema.py, cli.py
  data/        the lazy xarray layer: open, patches, datamodule, transforms, augment
  dynamics/    Dynamics base + registry (RK4 Lorenz-63/96, two-scale)
  models/
    ocean/     the gridded-ocean family, built through the @register_model registry:
               nosc/, fourdvarnet/, baselines/, assimilation/, nn/
  obs/         OSSE observation operators — repeat orbits, per-mission masks, virtual Argo
  training/    losses, loss grouping, optimisers, patch weights, callbacks, vertical modes
  inference/   prediction, export, and the shared product_contract.py
  legacy/      the reserved toy / Lorenz / QG family — models/, data/, training/,
               evaluation/, conf/ — see oceanml3d/legacy/README.md
config/        YAML presets for the CLI, rooted at main.yaml; config/legacy/ is the toy root
legacy/        the toy driver scripts: train.py, eval_*.py, run_experiment*.py, ...
tests/         the suite (gridded *and* reserve)   reports/, batch/, scripts/, demos/, notebooks/
```

Import by full path: `from oceanml3d.models.ocean.nosc.model import NOSCUNet`.

**Two model families, deliberately.** `oceanml3d/models/ocean/` is the gridded-ocean code, built by
the `@register_model` registry and trained by the `oceanml3d` CLI — that is what this repository is
about. `oceanml3d/legacy/` is the toy / Lorenz / QG code it grew out of (flow matching, CFM, SDA,
the L96 4DVarNet), built by `legacy/models/factory.py` and trained by `legacy/train.py`. They are
not interchangeable — `legacy/models/fourdvarnet.py` and `models/ocean/fourdvarnet/` are different
models that share a name. The reserve is set aside, not removed: its tests still run, and
`oceanml3d/legacy/README.md` says how to use it and how to undo the move.
`docs/feature_inventory.md` maps every capability to its file.

## Run

```bash
oceanml3d --help                           # the registry-based gridded path
oceanml3d experiment=smoke training=debug  # see config/experiment/ (11 presets)
python legacy/train.py experiment=<name>   # the reserve, see config/legacy/experiment/ (67)
pytest -q -m "not slow"                    # 811 passed, 9 skipped, ~7 min
ruff check .
```

## Companion repository

[`oceanml3d-eval`](https://github.com/aitmohandk/oceanml3d-eval) scores the products this repository
produces. The two share `product_contract.py` byte-for-byte, sealed by a SHA-256 pin on both
sides — change it in one repository only and the cross-repo contract test fails in the other.

`AGENTS.md` holds the working conventions, `PLAN.md` the state of the work,
`docs/feature_inventory.md` what came from where, `docs/worktrees.md` the topic-worktree workflow.
