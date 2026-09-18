# 4DVarNet-FM Project Guidelines

## Session Workflow

Every opencode session in this repository MUST follow this workflow:

1. **Read PLAN.md first** — Before making any changes, read `PLAN.md` to understand the current design plan.
2. **Implement changes** — Make the requested code modifications.
3. **Verify** — Run the relevant test/lint commands (see below).
4. **Log changes** — Append a dated entry to `CHANGELOG.md` describing what was implemented, why, and any notable design decisions.

### Working tree selection (IMPORTANT)

This project uses one physical worktree per topic branch:

| Worktree dir | Branch | Topic |
|---|---|---|
| `/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode` | `master` | Integration / PR landing |
| `.../4dvarnet-fm-opencode/4dvarnet-fm-joint-da` | `feature/l96-joint-da-benchmark` | Joint DA (ETKF) topic |
| `.../4dvarnet-fm-opencode/4dvarnet-fm-cfm-v2v3` | `feature/l96-v2v3-pure` | V2/V3 (TweedieCFM + PredictStateCFM) topic |

**Every opencode session MUST start in the appropriate working tree.** Branch-pinning guarantees you cannot work on the wrong branch, but this only works if you `cd` into the topic directory first:

- **Joint-DA session**: `cd .../4dvarnet-fm-opencode/4dvarnet-fm-joint-da && opencode`
- **V2/V3 session**: `cd .../4dvarnet-fm-opencode/4dvarnet-fm-cfm-v2v3 && opencode`
- **Main repo changes**: `cd .../4dvarnet-fm-opencode && opencode`

**Worktree bootstrap rule:** When starting a topic session, if `AGENTS.md`, `PLAN.md`, or `CHANGELOG.md` were last updated on the master branch **before the topic was forked** (check their timestamps), manually sync them from master first:

```bash
git pull origin master --dry-run  # fast check: origin/master newer?
# If newer: git pull origin master    # sync merged docs first
```

Commit-before-switch rule: **Never leave uncommitted edits when switching worktrees.** Always commit or stash your unfinished work before checking out a different branch.

## Git / PR Workflow

These general requirements apply to code changes in **every** session (not just L96).
The canonical, current version of these rules lives here; PLAN.md may carry
L96-specific details and pointers.

### Branching

- New work branches use the `feature/<topic>` prefix (e.g. `feature/session-workflow-automation`).
- `feat/*` names are **reserved for integration branches** (e.g. `feat/l96-neural-eval-fix`).
  A repo ruleset **blocks direct pushes of new `feat/l96-*` branches**, so always create
  `feature/...` branches for new work.
- Check the active integration branch (remote / PLAN.md) and use it as the PR base.

### Run-to-completion policy (IMPORTANT)

Once the user gives a go-ahead (e.g. "go", "proceed", "approved"), the implementing agent
drives a subtask **all the way to a merged PR without pausing for another approval between
the create → CI → review → merge steps**. The loop is fully automated (approval is done by
the `rfablet-review` identity and the `pytest` CI check is the merge gate), so nothing a human
must decide sits in the middle. Concretely, after pushing and opening the PR, continue:
wait for the `pytest` check to pass, run the reviewer approval, then verify and squash-merge.
Do **NOT** treat "PR created" as a natural stopping point.

Stop for user input only on a genuine external blocker:

- a reviewer **request-changes** (must be fixed with a new commit)
- a **non-informational CI failure** (informational ruff `continue-on-error` does not block)
- a **merge conflict**
- a **checkpoint the user explicitly asked to review**

### Review + merge

- Approvals come from the second GitHub account **`rfablet-review`** — not the author
  (GitHub blocks self-approval). Use `scripts/open_pr.sh review <PR#>` with the reviewer
  token at `~/.config/opencode/reviewer-token`.
- Pipeline: `scripts/open_pr.sh create "<msg>"` → wait for CI → `review <PR#>` →
  `verify <PR#>` (squash-merge).
- **Merge gate = `pytest -m "not slow"`** on the required test files. Ruff is informational
  (`continue-on-error: true`) so `mergeStateStatus: UNSTABLE` does not block the merge.
- Before merging, run the fast tests and `ruff check` on touched files locally.

### Hygiene

- Never commit artifacts/checkpoints or untracked scratch files (`experiments/` is gitignored).
- Add a CHANGELOG.md entry (see format below) for every merged change.

## Changelog Format

Each entry in `CHANGELOG.md` should follow this format:

```
## YYYY-MM-DD: Short Title

**Summary:** 1-2 sentence description of changes.
**Files modified:** `path/to/file.py` — brief note
**Rationale:** Why this change was made.
**Verification:** Test command run and result.
```

## Build, Lint, and Test Commands

- **Lint:** `ruff check .` — check code quality
- **Type check:** `mypy .` — static type analysis
- **Tests:** `pytest tests/ -v` — run full test suite
- **Quick test:** `pytest tests/ -v -m "not slow"` — skip slow tests
- **Coverage:** `pytest tests/ --cov=oceanml3d --cov-report=term`

Always run tests after making changes.

## Project Structure

All importable code lives under the `oceanml3d/` package. Import it by its full path
(`from oceanml3d.models.ocean.nosc.model import NOSCUNet`); the bare top-level names (`models`,
`data`, …) no longer exist.

The repository is about the **gridded ocean models**. The toy / Lorenz / QG family it grew out of
is in reserve under `oceanml3d/legacy/` + `legacy/` + `config/legacy/` — still tested, still
runnable, just out of the way. `oceanml3d/legacy/README.md` is the only document about it.

- `oceanml3d/data/` — the lazy xarray layer: `open`, `patches`, `datamodule`, `transforms`, `augment`
- `oceanml3d/models/ocean/` — the gridded models, built through the `@register_model` registry
- `oceanml3d/training/` — losses, loss grouping, optimisers, patch weights, callbacks, vertical modes
- `oceanml3d/inference/` — prediction, export, `product_contract.py`
- `oceanml3d/obs/`, `oceanml3d/dynamics/` — OSSE observation operators; dynamics base + registry
- `oceanml3d/legacy/` — the reserved toy family: `models/`, `data/`, `training/`, `evaluation/`, `conf/`
- `config/` — YAML presets for the CLI, rooted at `main.yaml`; `config/legacy/` is the toy root
  (stays at the repository root: `@hydra.main(config_path=…)` resolves relative to the driver
  script that carries the decorator)
- `legacy/` — the toy driver scripts: `train.py`, `eval_*.py`, `run_experiment*.py`, …
- `reports/` — Report generation scripts
- `jobs/` — how to run on a cluster: `run.sh` (scheduler-agnostic), `slurm/` and `pbs/` wrappers,
  `env/<site>.sh` per centre. `container/oceanml3d.def` builds the image.
- `legacy/batch/` — the inherited SLURM scripts. Reserve, not runnable as they stand (they
  hard-code one user's paths and environment, and assume Slurm). See `legacy/batch/README.md`.
- `tests/` — Unit and integration tests, for both families

## Key Conventions

- **Python 3.10+** with `torch`, `numpy`, `hydra-core`, `pytorch-lightning`
- **Configuration** uses Hydra/OmegaConf (`oceanml3d/config_schema.py` for the gridded path,
  `oceanml3d/legacy/conf/schema.py` for the reserve)
- **No comments** in code unless absolutely necessary (prefer self-documenting names)
- **Type hints** should be used for all function signatures
- **Training** uses PyTorch Lightning — `BaseOceanModel` for the gridded path, the `LitModel`
  wrapper in `oceanml3d/legacy/training/lightning_module.py` for the reserve
- **Two-stage training** pattern (reserve): Stage 1 trains the mean estimator, Stage 2 freezes it and trains the residual
- **Data** is generated on-the-fly; no large data files committed to git
- **Tests** use `pytest` with markers (`@pytest.mark.slow`) for expensive tests

## When Making Model Changes

- Update the corresponding config in `config/experiment/` if training parameters change
- Register the new gridded model with `@register_model` (`oceanml3d/registry.py`) — `docs/adding_a_model.md`
- Reserve models instead register in `oceanml3d/legacy/models/factory.py` — `legacy/train.py` and
  `oceanml3d/legacy/evaluation/neural_inference.py` both build through it, so a model added in only
  one of them cannot be evaluated from a checkpoint
- Add tests for any new model, loss, or dataset in `tests/`
- Document the change in `CHANGELOG.md`

