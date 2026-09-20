# Reserve documentation — the toy / Lorenz-96 / QG family

Working notes of the family this repository grew out of: flow matching and CFM variants, joint
state-parameter estimation, the SDA sampler, the L96 and QG experiments. The code they describe is
in reserve under [`oceanml3d/legacy/`](../../oceanml3d/legacy/README.md), `legacy/` and
`config/legacy/`; it is still tested and runnable, and these notes are what make it usable again.

**They are not maintained.** Each was written at a point in time, most of them before this
repository became about the gridded ocean models, and they are kept for provenance and for whoever
picks the reserve up. Dates and numbers inside them refer to their own moment. What is current
lives in [`docs/`](..) one level up.

| File | What it holds |
|---|---|
| `phase_B_l96_cfm_variants.md` | the V2/V3 CFM variants (TweedieCFM, PredictStateCFM) and their naming |
| `phase_C_l96_joint_da.md`, `phase_C_l96_joint_neural.md` | joint state-parameter estimation: ETKF, then the neural variants |
| `joint_estimation_progress.md`, `joint_additional_metrics_plan.md` | progress log and the metrics planned for that work |
| `experiment_G_tau0_cfm.md` | experiment G, the tau=0 CFM ablation |
| `cond_extra_dim_plan.md` | conditioning through an extra dimension |
| `fdv_torch_compile_and_jax_notes.md` | `torch.compile` and JAX notes for the 4DVarNet solver |
| `research_notes_cfm_da_originality_and_benchmarking.md` | literature and originality notes; referenced by `oceanml3d/legacy/evaluation/sda_sampler.py` |
| `L96_FAST_WEIGHTS_PROGRESS.md`, `L96_NEURAL_TRAINING_PROGRESS.md` | training progress logs of the L96 runs |
| `PLAN_case_study_refactoring.md` | the plan that abstracted L63 / L96 / shallow water behind one interface |
| `TEST_SUITE_SUMMARY.md`, `VANILLA_EXPERIMENT_TEST_PLAN.md` | test notes for the L63 baselines and the vanilla experiments |
| `IMPLEMENTATION_SUMMARY.md` | the DA-baseline visualisation work behind `demos/` |

Three documents were removed rather than moved here, because they described a state that no longer
exists and the history is in [`CHANGELOG.md`](../../CHANGELOG.md): `docs/ROADMAP.md` and
`docs/AUDIT_ca5277c.md` (the September audit and its lots, all closed) and `docs/worktrees.md` (a
worktree layout belonging to the upstream repository, never valid here).
