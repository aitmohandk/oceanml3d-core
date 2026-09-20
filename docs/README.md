# Documentation

Start with the one that matches what you are doing.

| Document | Read it when |
|---|---|
| [`gridded_models.md`](gridded_models.md) | you configure or extend the gridded ocean models: every key of a task config, what selects a checkpoint, what is validated and refused |
| [`pipeline_3d.md`](pipeline_3d.md) | you run the OSSE-3D task end to end: prepare, train, follow the run, predict, export |
| [`platforms/datarmor.md`](platforms/datarmor.md), [`platforms/jeanzay.md`](platforms/jeanzay.md) | you are on that machine: the runbook, step by step, with its queues, quotas and traps |
| [`data_preparation.md`](data_preparation.md) | you need to know which recipe produces which catalog key |
| [`adding_a_model.md`](adding_a_model.md) | you add a model to the registry |
| [`product_format.md`](product_format.md) | you read the exported product, or change the contract shared with `oceanml3d-eval` |
| [`migration_nosc.md`](migration_nosc.md) | you are coming from the NOSC repository and look for a module |
| [`feature_inventory.md`](feature_inventory.md) | you look for a capability of the two upstream repositories and where it went |
| [`porting_guide.md`](porting_guide.md) | you port one of the remaining capabilities |
| [`upstream_triage.md`](upstream_triage.md) | you wonder what to do with what has been published upstream since the import |

At the repository root: [`README.md`](../README.md) for the layout and the commands,
[`PLAN.md`](../PLAN.md) for the open work, [`CHANGELOG.md`](../CHANGELOG.md) for what changed and
why, [`LICENSING.md`](../LICENSING.md), [`NOTICE`](../NOTICE) and [`PROVENANCE.md`](../PROVENANCE.md)
for the licence and the provenance, [`AGENTS.md`](../AGENTS.md) for the working conventions.

[`legacy/`](legacy/README.md) holds the working notes of the toy / Lorenz-96 / QG family, which is in
reserve. They are kept for provenance and are not maintained.
