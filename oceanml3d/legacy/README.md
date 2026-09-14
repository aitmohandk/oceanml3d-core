# The reserve — toy / Lorenz / QG family

This directory holds the flow-matching and L96-4DVarNet work that `oceanml3d-core` grew out of.
Nothing here is imported by the gridded pipeline, and nothing here is deleted: it was set aside on
2026-09-14 so the repository shows only what the project is now about — **gridded ocean models**
(`oceanml3d/models/ocean/`, the `@register_model` registry, the `oceanml3d` CLI, `config/main.yaml`).

The two families never shared code (`docs/gridded_models.md` §1): separate model construction,
separate Hydra root, separate config schema. Moving one out of the way was a pure rename.

## What is where

| | |
|---|---|
| `oceanml3d/legacy/models/` | 19 models: `solver` (Tweedie), `vanilla_cfm`, `fourdvarnet` (the L96 one), `direct_unet`, `sda`, `param_head`, `unet`, `residual`, `interpolant`, `qg*`, `lorenz*_dynamics`, `monai_unet_adapter`, and `factory.py` (the `model_type` dispatch) |
| `oceanml3d/legacy/data/` | `lorenz63`, `lorenz96`, `qg`, `qg_neural`, `dataloader`, `normalization`, `random_{bias,param}_dataset` |
| `oceanml3d/legacy/training/` | `lightning_module`, `pipeline`, `stage1`, `stage2` (the two-stage recipe) |
| `oceanml3d/legacy/evaluation/` | 13 modules: `baselines` (4D-Var / EnKF / ETKF), `neural_inference`, `metrics`, `run*` |
| `oceanml3d/legacy/conf/` | `schema.py` — the second config schema, the toy one |
| `legacy/` (repo root) | the 20 driver scripts: `train.py`, `train_qg_neural.py`, the `eval_*.py`, `evaluate_all*.py`, `run_experiment*.py`, `precompute_*.py`, `rerun_*.py` |
| `config/legacy/` | the toy Hydra root: `config.yaml`, `lorenz63_default.yaml`, `lorenz96_default.yaml`, `baselines/`, `case_study/`, and the 67 `experiment/` presets |

**The tests did not move.** The 47 toy test files stay in `tests/` and still run on every `pytest`
invocation, including the CI gate. That is deliberate: the reserve is covered, so a later change to
the core that breaks it fails loudly instead of rotting quietly.

## Using it

```bash
python legacy/train.py --config-name=experiment/L1b_direct_unet_s0s1_norm
python legacy/eval_neural_l96.py --help
python -c "from oceanml3d.legacy.models.solver import TweedieSolver"
```

`legacy/train.py` composes against `config/legacy/`; `hydra.initialize(config_path=...)` in any new
script must point there too, not at `config/`, which is the gridded root.

## What the reserve still imports from the core

Two modules, and nobody should garbage-collect them as orphans:

- `oceanml3d/training/losses.py` — merged by hand during the NOSC graft; both families use it.
- `oceanml3d/models/ocean/nn/unet_monai.py` — `patch_diffusion_resblock()`, the MONAI
  `DiffusionUNetResnetBlock` monkeypatch that adds the `spatial_dims == 1` branch. It lives on the
  gridded side because the 2-D trunk needs it too; `legacy/models/monai_unet_adapter.py` imports it.

The dependency runs one way only (reserve → core). Nothing in `oceanml3d/models/ocean/`,
`oceanml3d/cli.py` or `oceanml3d/registry.py` imports `oceanml3d.legacy`.

## Undoing the whole thing

It was one commit, and it was a pure move:

```bash
git log --diff-filter=A --format='%H %s' -1 -- oceanml3d/legacy/__init__.py
git revert <that sha>
```

That restores `oceanml3d/models/*.py`, `oceanml3d/{data,training,evaluation,conf}`, the root
drivers and `config/`'s toy tree to exactly where they were. To bring back one piece only, an
inverse `git mv` plus the matching import rewrite is enough — there is no shim layer and no
registry entry to undo.
