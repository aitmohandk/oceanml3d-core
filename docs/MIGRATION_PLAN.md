# Refactor-in-place migration plan

**Decision.** Stop treating `oceanml3d-core` / `oceanml3d-eval` as the destination. Restart from the
existing repositories, restructure them in place, then transplant the improvements that currently
live in the oceanml3d prototypes. The prototypes become a *specification and a donor of code*, not a
target.

**Why this is the right way round.** Functionality is preserved by construction: the code that
implements it is never deleted, only moved. The burden of proof inverts — instead of proving that a
reimplementation reproduces 39 000 lines, you prove only that each mechanical transformation left
behaviour unchanged, and the 49 existing test files are the safety net. Git history (blame, bisect)
survives. And the pruning of dead weight becomes an explicit, reviewable step rather than an
accident of what someone remembered to port.

---

## 0. Measured starting point

| | `4dvarnet-fm-opencode` | `4dvarnet-ocean-reanalyses` | `NOSC` |
|---|---|---|---|
| Python | ≈29 k lines (excl. tests) | 1.3 k lines | ≈4.6 k lines (`code/multivar_drifter_`) |
| Tests | 49 files / 9.9 k lines, 42 need torch, 16 slow-or-GPU | 5 files | none |
| Configs | 76 YAML | 6 YAML | ≈30 xp + 80 archived |
| Peripheral (not imported by the core, but importing it) | `reports/` 7.8 k + `demos/` 1.4 k + `batch/` 0.8 k + 20 root scripts 4.5 k — 45 files, ~25 core symbols | `reports/`, notebooks | 33 notebooks, 90 archived configs, committed `.nc`/`.png`, 838 MB tree |
| Core to keep | `models/` 4.3 k, `evaluation/` 7.3 k, `data/` 2.6 k, `training/` 0.4 k | `data/`, `models/`, `conf/schema.py` | model, data pipeline, `env/velocity_metrics` |

**Age changes the whole framing.** The repositories are a few months old at most. There is no legacy
to excavate: the peripheral directories are almost certainly live work, not archaeology; git history
has little diagnostic value yet; and the structure has not set. Three consequences drive the plan
below:

* **Do it now, or it gets more expensive weekly.** ~39 k lines in a few months means the current
  patterns replicate fast. Two root eval scripts already overlap at 78 %; that is not debt left by
  someone who has moved on, it is the template the next script will copy.
* **Ask, do not infer.** Every conclusion I could reach from `git log` or import graphs is worse than
  half a day of interviews with the authors, who remember what each file is for. Use the
  measurements below to prepare the questions, not to answer them.
* **The risk is collision, not loss.** Refactoring an actively developed repo means conflicts across
  the 45 peripheral files. The mitigation is a short announced window, mechanical renames applied in
  one commit, and steps of a few days each — not a two-month restructuring.

Two corrections to the comparison table, from measurement:

* **`get_dynamics()` is not the coupling problem.** It has a single call site. The real coupling is
  ~15 direct `Lorenz96Dynamics(...)` / `Lorenz63Dynamics(...)` instantiations inside
  `evaluation/run*.py` and `tune_*.py`. The registry is still the right target, but the work is in
  the scripts, not in the factory.
* **`DataConfig` is genuinely monolithic** — 50 fields mixing L63 parameters, two-scale L96
  parameters (`NO, J, h, hx, eps, fast_weights`), observation settings and case-study switches
  (`case`, `train_mix`, `randomize`), plus `device`. Splitting it is the highest-value config step.

---

## 1. Repository mapping

| Target | Trunk | How |
|---|---|---|
| `oceanml3d-core` | **`4dvarnet-fm-opencode`**, renamed | it holds the most code, the most tests and the conventions; keep the git history untouched |
| `oceanml3d-eval` | **new repo, history grafted** | the evaluation code has three sources: `fm/evaluation/` (7.3 k), `NOSC/env/velocity_metrics/`, and the reanalyses stub. Use `git subtree split -P evaluation` on fm and `git subtree add` for velocity_metrics so blame survives; the reanalyses `baselines.py` (8-line OI) is superseded, not merged |
| ocean data layer | `4dvarnet-ocean-reanalyses` | its `data/` (GLORYS, ERA5, grid, observations) and `conf/schema.py` land in core via `git subtree add`, then get reconciled with the NOSC/xarray layer |
| NOSC | model plugin inside core | `git subtree add -P models/nosc` on `code/multivar_drifter_`, then the cleanups already scripted (`import_nosc_config.py`) |

Rule for every move: `git mv` or `git subtree`, never copy-paste. A move that loses blame makes the
next bisect impossible, which is the whole reason for choosing this strategy.

---

## 2. Safety net before touching anything (step 0, do not skip)

1. **Measure the suite.** Run the 49 test files, record wall time and which need GPU. The plan below
   assumes a fast CPU subset exists; if the honest number is "40 minutes on a GPU box", the
   refactoring cadence changes and you decide that now, not in week three.
2. **Freeze behaviour.** Extend `tests/legacy/generate_golden.py` (donated from the prototype) to
   cover, at minimum: one L63 and one two-scale L96 trajectory, one EnKF and one ETKF analysis, one
   Weak4DVar and one Strong4DVar analysis, the CFM sampler at a fixed seed, `_gaspari_cohn`, and the
   metrics on a fixed ensemble. This is cheap now — you are *inside* the repo with its environment,
   so no cross-repo import problem. Commit the `.npz` files.
3. **Freeze scores.** For 2–3 reference experiments per system, run the current pipeline and store
   the metrics; later they become `oceanml3d-eval baseline --check` gates.
4. **Interview the authors.** Half a day: which directories are live, which experiments must keep
   running during the window, which results are considered settled. On a months-old codebase this is
   strictly better information than anything inferable from the history.
5. **Add an import-smoke test.** A test that imports every module in the repo, so a rename that
   breaks a peripheral script fails CI instead of failing silently.
6. **Tag.** `git tag pre-refactor` on all three repos.

Steps 3 onward are only safe because of this step.

---

## 3. Sequence

Each step is one reviewable PR, ends green on the existing suite, and changes one thing.

**Step 1 — Packaging, no renames.** Add `pyproject.toml`, move the 20 root scripts into
`scripts/` (still working), make the repo `pip install -e .`-able. Nothing else moves. *Sizing: a
day. Risk: import paths — the existing tests catch it immediately.*

**Step 2 — Reduce the refactoring surface (nothing is deleted, nothing is frozen).** Measurement
contradicts the usual assumptions: `reports/`, `demos/` and `batch/` are **not imported by the core**
(1 hit, a false positive), the repo is 40 MB, and the 12 root `eval_*.py` scripts overlap only in
three pairs (62–78 %). There is no dead weight. Given the age of the code, there is also no
quarantine to apply: these directories are live work, and freezing them would block their authors.
What the measurement does show is two real problems.

1. **Diverged metrics — fix first, while everything still runs.** 25 metric computations are inlined
   in `reports/`, against only 5 scripts importing `evaluation.metrics`. On a months-old codebase the
   divergence is recent and the authors can still say which formula is authoritative. Diff each
   inline computation against the module; where they agree, replace with the import; where they
   disagree, that is a finding — decide, fix the module, and note it in the changelog. Doing this
   before any rename means the comparison is a pure diff, with no moved symbols in the way.
2. **45 peripheral files import ~25 core symbols.** Every rename in steps 3–6 breaks them. They stay
   in the trunk and stay working: each renaming PR carries the mechanical update of its call sites
   (a codemod plus a green import-smoke test that simply imports every module in the repo). Add that
   smoke test now — it is ten lines and it turns "45 files silently broken" into a red CI.
3. **Leave the binaries alone.** 84 committed PDFs/PNGs in a 40 MB repo cost nothing; rewriting
   history to remove them would cost every open branch.

The duplication among the eval scripts is not addressed here: it dissolves in step 7, when each is
converted into a benchmark. Converting them is urgent for a different reason — until the benchmark
path exists, every new experiment adds another near-copy.

`NOSC` is the only repo needing genuine cleanup, and for a reason unrelated to age: an 838 MB working
tree with committed `.nc` and `.png`, 33 notebooks and 90 `xp/old/` configs. Strip the data files and
the archived configs *at the subtree import*, so they never enter the trunk of `oceanml3d-core`; the
notebooks follow their authors' call.

**Step 3 — Registry.** `@register_dynamics` / `@register_model` + entry points; keep
`get_dynamics()` as a thin deprecated wrapper. Replace the ~15 direct instantiations in
`evaluation/run*.py`. *Sizing: half a day. Donor: `oceanml3d/registry.py`, `dynamics/base.py`.*

**Step 4 — Config groups.** Split `DataConfig` into Hydra groups
`paths / data / model / training / ablation / experiment`; keep the old dataclass as a deprecated
adapter so the 76 YAML files keep loading; migrate them with a script (the shape of
`import_nosc_config.py`); add `validate_config`. *Sizing: the biggest config step, ~3 days. Donor:
`oceanml3d/config_schema.py`, `config/**`. Guard: `test_hydra_config.py`, `test_config_persistence.py`
already exist.*

**Step 5 — Gridded data layer (purely additive).** Bring in `VariableSet`, `Catalog`,
`open_variable_set`, `PatchArray`, `OceanDataModule`, `augment`, `obs/` from the prototype and the
reanalyses `data/`. Nothing existing is touched: the trajectory datasets keep working side by side.
*Donor code transplants almost verbatim — it is numpy/xarray and torch-free.*

**Step 6 — One model contract.** Introduce `BaseOceanModel`; migrate the existing models one at a
time, each in its own PR with its equivalence test against the step-0 golden file. Order:
`direct_unet` → `fourdvarnet` → `vanilla_cfm` → `sda` → `param_head`. *This is the long pole,
~2 weeks, and the only step where a silent science change is possible — hence one model per PR.*

**Step 7 — Extract the evaluation repo (strangler fig).** Create `oceanml3d-eval` with the grafted
history; define the `product.yaml` contract (donor: `product_contract.py`, hash-pinned in both
repos); add the export step to core. Then convert the 15 root `eval_*.py` **one at a time** into
benchmark YAML + metric, keeping the old script until its replacement reproduces its numbers. Delete
each script only when its `baseline --check` passes. *Sizing: ~2 weeks, but incremental and always
shippable.*

**Step 8 — Fold in NOSC and the reanalyses case studies.** Subtree, then the migration already
documented (`docs/migration_nosc.md`). CS1–CS4 become `config/data/*.yaml`.

**Step 9 — Remove the scaffolding.** Delete the deprecated `get_dynamics` wrapper and the
`DataConfig` adapter, fail the build on their reappearance.

---

## 4. What the prototypes contribute, and what of them is thrown away

**Transplanted almost as-is** (additive, torch-free, tested): `variables.py`, `catalog.py`,
`data/open.py`, `data/patches.py`, `data/augment.py`, `obs/**` (orbits, masks, pseudo-obs, Argo),
`config_schema.py`, `inference/export.py` + `product_contract.py`, `training/{loss_grouping,
vertical_modes,weights}.py`, `scripts/prepare/**`, `scripts/import_nosc_config.py`, the whole of
`oceanml3d-eval` (products, regions, references, metrics, benchmarks, runner, report, plots,
regression gate), plus the docs (`feature_inventory.md`, `porting_guide.md`, `product_format.md`)
and `tests/legacy/`.

**Discarded in favour of the originals** — this is the point of the inversion:
`oceanml3d/dynamics/lorenz.py` (fm's is richer: torch, parameter estimation, fast weights),
`models/fourdvarnet` (fm's solver is more complete), `models/assimilation` (fm's EnKF/ETKF/4D-Var
with localisation and inflation supersede my EnKF and OI), `models/nn/unet2d` (reconcile with fm's
`unet.py` and the reanalyses backbone — keep one).

**Kept as an addition to the originals:** the ensemble metrics (`ensemble_scores`: fm has CRPS and
energy score, the spread-skill ratio with the `sqrt((m+1)/m)` correction and the rank histogram are
new), the non-neural controls (`passthrough`, `linear`, `climatology`), and the multi-stage hook if
fm's `pipeline.py` proves too specific.

---

## 5. Honest costs of this route

* **Collision with ongoing work is the main risk.** The repos are young and active: 45 peripheral
  files import core symbols and new experiments land continuously. Mitigations: announce a window,
  keep each renaming PR mechanical and same-day, add the import-smoke test in step 2, and agree that
  new experiments during the window follow the new config shape rather than copying an old script.
* **The safety net may be slower than hoped.** 42 of 49 test files import torch and 16 are slow or
  GPU-bound. If the CPU subset is thin, add fast surrogate tests *before* step 6, or the model
  migration proceeds blind.
* **Two incompatible data models cohabit for a while.** Trajectory datasets and gridded datasets
  live side by side from step 5 to step 8. That is deliberate (additive = safe), but it must be
  written in `AGENTS.md` so nobody "unifies" them prematurely.
* **The train/eval split is the one genuine rewrite.** Extraction cannot be a pure move: the 15 root
  scripts have no equivalent in the new shape. The strangler-fig rule (old script stays until its
  benchmark reproduces its numbers) is what keeps it safe.
* **Timeline.** Steps 1–5 ≈ 1.5 weeks; step 6 ≈ 2 weeks; step 7 ≈ 2 weeks; steps 8–9 ≈ 1 week.
  Roughly 6–7 focused weeks — but on a months-old codebase with no legacy to excavate, the earlier
  steps are likely faster than that and the estimate should be revised after step 0 measures the test
  suite. The prototypes save perhaps 2 weeks of design and writing, and have already been used to
  find the design mistakes (the `_self_` ordering bug, the wrong spread-skill definition, the
  single-scale vs two-scale L96).
* **Every week of delay costs more than the last.** The patterns being fixed are the ones the next
  experiment will copy: 12 root eval scripts today, three of them near-duplicates.

---

## 6. Definition of done

* `pre-refactor` tag and post-refactor HEAD produce the same numbers on every golden case and every
  frozen score, or every difference is documented in `docs/feature_inventory.md`.
* `docs/feature_inventory.md` has no *planned* line left without a `PLAN.md` item.
* `oceanml3d-eval` reproduces what the 15 deleted `eval_*.py` scripts produced.
* Both repos: `pytest` green, `ruff` clean, `product_contract.py` byte-identical and hash-pinned.
