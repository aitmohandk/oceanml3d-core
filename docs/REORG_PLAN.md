# Phase 1–2: restructure, then mutualise — with the measurements that size the work

Your two-phase plan is the right shape. One amendment on ordering, then the concrete target and the
measured list of what can actually be merged.

## The ordering amendment: interleave per domain, do not do two global passes

Reorganising first and deduplicating second means moving N copies into the new tree, at which point
the structure has blessed them: `ETKF` and `EnKF` both land in `assimilation/`, both get an
`__init__.py` entry, both get imported by the configs — and merging them afterwards is now a
breaking change to a structure people have started depending on. Deduplicating first is worse: you
cannot decide where a shared helper belongs before the target tree exists.

What works is **move-and-merge, one domain at a time**: for each domain (dynamics, models,
assimilation, data, metrics, config), decide the target location, then move the files *and* merge
their duplicates in the same pull request. Each PR is small, guarded by the existing tests plus the
golden files of step 0, and reviewable as one idea: "this is what the assimilation code looks like
now". The global tree emerges after the last domain, instead of existing empty for weeks.

The one exception is **step 1 of your plan applied to configuration**, which must come first and
globally: `paths / data / model / training / ablation / experiment` groups plus `validate_config`.
Everything else is arranged relative to the config tree, and every later PR needs somewhere to put
its options.

## Target tree

```
oceanml3d-core/
  oceanml3d/
    catalog.py  variables.py  registry.py  config_schema.py
    dynamics/          base.py (Dynamics + @register_dynamics), lorenz.py, qg.py
    data/              open.py (lazy xarray), patches.py, datamodule.py, transforms.py,
                       augment.py, trajectories.py      # toy/trajectory datasets keep their own reader
    obs/               missions.py orbits.py sampling.py pseudo_obs.py argo.py argo_virtual.py
    models/            base.py (BaseOceanModel + @register_model)
                       nn/         unet2d.py heads.py blocks.py     # ONE U-Net for the whole repo
                       direct/     nosc.py direct_unet.py
                       variational/ fourdvarnet.py solver.py prior.py
                       generative/ cfm.py sda.py interpolant.py param_head.py
                       assimilation/ filters.py (EnKF/ETKF), variational.py (3D/4D-Var),
                                     joint.py (state+parameter mixin), localisation.py, oi.py
                       baselines/  passthrough.py linear.py climatology.py
    training/          losses.py loss_grouping.py weights.py optim.py callbacks.py
                       pipeline.py (stages)
    inference/         predict.py export.py product_contract.py
    cli.py
  config/              paths/ data/ model/ training/ ablation/ experiment/
  scripts/             prepare/ make_toy_data.py make_synthetic_data.py import_nosc_config.py
  tests/               unit/ legacy/(golden files) integration/
oceanml3d-eval/          product.py regions.py reference/ metrics/{eulerian,gridded,spectral,
                       lagrangian,ensemble}.py benchmarks/ report.py plots.py regression.py
                       benchmarks/*.yaml products/*.yaml regions/*.json third_party/
```

Two placement rules that carry most of the value:

* **`models/nn/` holds every reusable block.** Today there are at least three U-Nets
  (`fm/models/unet.py`, the MONAI adapter, the reanalyses `backbone_unet2d.py`) plus NOSC's. One
  survives, parameterised; the others become configs.
* **`assimilation/` is a model family like any other**, not a corner of `evaluation/`. That is what
  lets an EnKF and a U-Net be compared by the same benchmark through the same `product.yaml`.

## Measured mutualisation targets

Duplicate-function analysis over `models/ evaluation/ training/ data/` + root scripts: 638 functions,
318 of ≥12 lines, **22 pairs ≥85 % identical, ~580 duplicated lines**. The largest, by domain:

| Target | Evidence | Approach | Order of magnitude |
|---|---|---|---|
| **`evaluation/baselines.py`** — 3 014 lines, 14 classes | `JointEnKFL96.assimilate` ≡ `JointETKFL96.assimilate` (100 %, 52 l); `assimilate_batch` 99 % over 94 l; `ETKF.__init__` vs `EnKF.__init__` 92 %; `Joint*` = **1 646 of 3 014 lines** against 1 081 for the base filters | The `Joint*` classes are the base filters plus a state-augmentation step. Extract a `JointEstimationMixin` (augment state, split, observation operator) and a shared `EnsembleFilter` base with the analysis step as the only difference | **−800 to −1 000 lines**, the single biggest win in the repo |
| **QG dynamics** — `qg_dynamics.py` vs `qg1l_dynamics.py` | `generate_wind_state` 100 % (28 l), `wind_curl_field` 100 % (19 l), `step` 90 %, `rollout_steps` 91 %, `generate_batch_trajectories` 91 %, `generate_full_trajectory` 88 % | One `QGBase` with the layer count as a parameter; `QGPsiDynamics`/`QG1LPsiDynamics` likewise (`step` 90 %, `rollout_trajectory` 92 %) | **−150 to −200 lines** |
| **Model config dataclasses** — 21 classes, 123 fields | 13 fields appear in ≥3 classes (`hidden_channels`, `dropout`, `param_dim`, `time_emb_dim`, `N_outer`…), ~66 redundant field declarations | A base `NeuralModelConfig` + per-family additions; most of them then collapse into `config/model/*.yaml` | **−60 to −100 lines**, and 21 classes → ~6 |
| **Root eval scripts** — 12 files, 4.5 k lines | `eval_fdv1_fdv1cfm_hybrid_l96` vs `eval_sda_fdv1_hybrid_l96` 78 %; two more pairs at 62–65 %; `print_table` identical across `evaluate_all*.py` | They dissolve into benchmark YAML + metric during the evaluation extraction — do not refactor them in place first | **−3 000 to −4 000 lines** moved to ~15 YAML files |
| **Inline metrics in `reports/`** | 25 inline computations vs 5 importing `evaluation.metrics` | Converge on the module (see the safety-net step); a disagreement is a finding, not a merge conflict | small in lines, large in correctness |
| **Misc.** | `sda.py::Unconditional/ConditionalPriorCFM.sample` 100 % (13 l); `RandomParam/RandomBiasLorenz96Dataset.__init__` 92 %; `FlowMatchingDataset/ConcatFMDataset.__getitem__` 86 %; MONAI vs direct U-Net `forward` 94 % | ordinary base-class extraction | **−100 lines** |

Realistic total: **the trunk loses roughly 4 000–5 000 lines** (≈15 % of the non-test code), most of
it by converting scripts into declarative benchmarks, and about 1 200 lines by merging genuinely
copy-pasted classes. That is worth doing — but note what it is *not*: there is no dead code to
delete, and the repo is 40 MB. This is consolidation, not cleanup.

## Sequence

1. **Config groups + `validate_config`** (global, first). Old `DataConfig` kept as a deprecated
   adapter so the 76 YAML files keep loading.
2. **Registry** (`@register_model`, `@register_dynamics`, entry points) — small: one factory and ~15
   direct instantiations in `evaluation/run*.py`.
3. **Domain by domain, move-and-merge**: dynamics (QG merge) → `models/nn` (one U-Net) → assimilation
   (the `Joint*` merge, the big one) → generative → data (add the lazy xarray/patches layer, keep the
   trajectory reader) → training.
4. **Extract `oceanml3d-eval`**, define the `product.yaml` contract with its pinned hash, then convert
   the 12 root scripts into benchmarks one at a time, deleting each only when its numbers reproduce.
5. **Fold in NOSC and the reanalyses case studies**, then delete the deprecated adapters.

Every merge in step 3 is guarded by an equivalence test against the golden file recorded before the
work starts. For the `Joint*` merge specifically, record the analysis of all four filters on a fixed
window first: it is the step where a subtle change would be both easiest to make and hardest to see.
