# Documentation

The entry point is the repository's [`README.md`](../README.md): what the project does, the complete
list of its features, how a pipeline runs and which files each step reads and writes. This directory
holds the detail it points to.

**To build a pipeline end to end, start with [`tutorial.md`](tutorial.md).**

| Document | Read it when |
|---|---|
| [`tutorial.md`](tutorial.md) | you build a pipeline end to end, from the sources to a scored product: a rehearsal on your machine with the real tools, then the same steps on a cluster, with the check after each step |
| [`gridded_models.md`](gridded_models.md) | you configure or extend the gridded ocean models: every key of a task config, what selects a checkpoint, what is validated and refused |
| [`pipeline_3d.md`](pipeline_3d.md) | you want to understand the OSSE-3D task: what each stage does and why, the observing-system simulation, the ablations, the pitfalls |
| [`platforms/datarmor.md`](platforms/datarmor.md), [`platforms/jeanzay.md`](platforms/jeanzay.md) | you are on that machine: the image, storage, queues, the exact commands, and every failure met so far with its fix |
| [`../jobs/README.md`](../jobs/README.md) | you run jobs, add a site, set personal defaults, or build the container |
| [`data_preparation.md`](data_preparation.md) | you need to know which recipe produces which catalog key |
| [`adding_a_model.md`](adding_a_model.md) | you add a model to the registry |
| [`product_format.md`](product_format.md) | you read the exported product, or change the contract shared with `oceanml3d-eval` |
| [`migration_nosc.md`](migration_nosc.md) | you are coming from the NOSC repository and look for a module |
| [`feature_inventory.md`](feature_inventory.md) | you look for a capability of the two upstream repositories and where it went |
| [`porting_guide.md`](porting_guide.md) | you port one of the remaining capabilities |
| [`upstream_triage.md`](upstream_triage.md) | you wonder what to do with what has been published upstream since the import |

At the repository root: [`PLAN.md`](../PLAN.md) for the open work, [`CHANGELOG.md`](../CHANGELOG.md)
for what changed and why, [`LICENSING.md`](../LICENSING.md), [`NOTICE`](../NOTICE) and
[`PROVENANCE.md`](../PROVENANCE.md) for the licence and the provenance, [`AGENTS.md`](../AGENTS.md)
for the working conventions.

[`legacy/`](legacy/README.md) holds the working notes of the toy / Lorenz-96 / QG family, which is in
reserve. They are kept for provenance and are not maintained.

## Keeping it true

Two tests hold this documentation to the code:

* `tests/test_tutorial.py` runs the tutorial's Part 1 command by command — the preparation in the
  fast suite, training and inference in the slow one.
* `tests/test_docs.py` checks that every relative link and anchor in the maintained documents
  resolves, and that the README's feature list names every task, experiment, model, ablation, site
  and CLI command the repository ships — so adding one without documenting it fails.
