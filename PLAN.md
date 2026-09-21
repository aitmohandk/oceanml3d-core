# oceanml3d-core — Plan

Derived from `4dvarnet-fm-opencode` and now independent of it (see `PROVENANCE.md`).

**How to read this file.** §A is the open work on the gridded ocean path — what this repository is
about. Everything after it is the backlog of the reserve (toy / Lorenz-96 / QG), kept because the
reserve is still tested and runnable. What has been *done* is in `CHANGELOG.md`, entry by entry,
with the reasoning; this file only carries what is not.

## A. Gridded path — open

Ordered by what blocks the next result.

- [ ] **The acceptance run.** One full run of `osse3d_gs21_multivar_unet` and one of
      `nosc_15m_duacs` on real data, with time and memory recorded in `CHANGELOG.md`. Everything
      else in this section is small; this is the one that tells us the pipeline holds at scale.
      Blocked only on the data preparation finishing on Datarmor.
- [ ] **`bathy_gs` — deferred, not dropped.** The static input is off (`bathy: null` in
      `config/data/osse3d_gs21.yaml`): the first thing to validate is the simple model, and the key
      has no producer. Once there is a recipe — GEBCO regridded onto the prepared truth with
      `regrid.py`, `reference:` pointing at the truth (`docs/data_preparation.md`) — `ablation=bathy`
      puts it back with no other change. `test_every_source_of_the_osse_task_is_produced_by_something`
      is what stops a source without a producer coming back in.
- [x] **`config/paths/datarmor.yaml` and `config/paths/jeanzay.yaml`** now exist, carrying local's
      sixteen keys with each centre's root; `test_every_site_env_file_has_a_matching_paths_file`
      keeps `jobs/env/<site>.sh` and the catalog in step.
- [ ] **Record the licence permission** in `LICENSING.md` (who granted it, when, reference), and
      put SPDX headers on the Python files — here and in `oceanml3d-eval`.
- [ ] **`cache: true` for the OSSE-3D configs**: decide on the evidence of the full run.
- [ ] **Order and border tests on `reconstruct`** (patch permutation, insufficient overlap).
- [ ] **The numpy binary-compatibility warning** (`numpy.ndarray size changed`) comes from a
      C extension built against another numpy; it is harmless but noisy. Identify which
      (netCDF4 or cftime, by elimination) and pin it in `environment.yml`.
- [ ] **`reports/` out of the repository** (a GitHub release or an artefact store): 39 MB of PNG
      and PDF for a 86 MB clone.
- [ ] **The crop and the domain's outer rim.** `rec_weight.crop` is in cells and nothing covers the
      domain's outer edge, so the product is NaN over `crop x step` (2° at 0.5°). `check_export_rim`
      refuses it when it reaches into `eval_domain`, but the design question is open: a crop in cells
      (the network's border artefacts are in cells), a crop in degrees, or a truth prepared over a box
      larger than the task domain so the rim falls outside it — the classical answer.
- [ ] Depth as a patch dimension — `PatchArray` is already generic over `DIMS`.

The evaluation side has its own open list in `oceanml3d-eval/PLAN.md`: an in-situ ARGO reference
(the OSSE → OSE step), along-track altimetry, significance testing between two models, and the
conversion of the last `legacy_from_fm/` drivers.

## B. Reserve backlog

### 0. The blocking question — RESOLVED, no GPU needed
- [x] **The published S0/S1 numbers reproduce.** The constants in `test_equiv_report.py` were stale
      (pre-bugfix 0.78/0.77 instead of the report's current 0.88/0.88) and the protocol used 1 window
      against a 200-window mean. Device RNG was ruled out by measurement first: six CPU seeds give
      +/-0.02 against a 0.085-0.106 gap. The science never moved. See `tests/KNOWN_FAILURES.md`.
- [x] run the 20-window comparison (`pytest tests/test_equiv_report.py`) and record it as the
      reference gate for the `Joint*` merge — **done on CPU, 26 passed in 11 min**, and re-measured
      on an independent sample (`EQUIV_REPORT_WINDOWS=16`, 26 passed). Fixing it turned up three more
      protocol defects, none of them in the science: the run was resuming a cache written by the
      earlier one-window investigation (the window count is not in the cache key), the sample was
      silently 10 windows rather than 20 (`make_mixed_datasets` sizes the test sets from
      `num_test_windows`, not `cfg.num_windows`), and the assertions used unpaired means of a
      heavy-tailed quantity — the published S1 Strong-vs-EnKF margin, 0.17, is smaller than the
      0.20 standard error of the mean measuring it. Now paired per window, level compared on the
      median. See `tests/KNOWN_FAILURES.md`.
- [ ] **`Strong4DVar` diverges on a minority of windows** — found by the above. On S1 window 5 the
      analysis blows up mid-window to X=516 against a normal truth (RMSE 23.3 vs a 1.93 median) while
      the other three filters sit at their medians; the minimisation over the initial condition fails
      to converge and returns the blown-up trajectory without raising. Needs a divergence guard.
      Until then the report comparison uses the median rather than the mean.

### 1. Safety net — mostly done
- [x] record the suite at `pre-refactor`: 585 passed, 27 failed, 2 collection errors
- [x] quarantine with reasons (`tests/KNOWN_FAILURES.md`)
- [x] golden files for QG and Lorenz-63, replayed at 1e-10
- [x] repair the 24 filter failures (dynamics injection) — adjudicated against external references
- [x] restore the two collection-error scripts as real tests
- [x] restore both collection-error scripts as real tests (6 + 10 passing)
- [x] **the quarantine is empty** — the last 11 xfail markers are removed and the causes fixed:
      a production bug in `JointCFM.compute_cfm_loss` (5 tests), two NaN comparisons that made
      bit-for-bit identical datasets look 94% different (2 tests), one stale key set, two genuine
      `JointCFM` API drifts, and `test_observations_noise`, which was a one-sigma band at n=25 and
      so failed on about a third of seeds while the code was correct. `pytest -q -m "not slow"`:
      **609 passed, 7 skipped, 0 xfailed**. See `tests/KNOWN_FAILURES.md`.
- [x] **the quarantine is still empty after the transplant** — the graft brought one red test
      (`test_enkf_beats_the_raw_observations`). It reproduces bit-identically in the donor, so it was
      inherited, not caused; adjudicated rather than marked. The EnKF is correct — the assertion was
      being measured over a ten-step window, which at this observation density is spin-up and
      nothing else (the filter converges to 0.04 against an observation error of 0.089, but needs
      ~40 steps to get there). The test now runs on an 80-step window and also asserts that the
      second half beats the first by 2x. `pytest -q -m "not slow"`: **779 passed, 11 skipped,
      45 deselected, 0 xfailed** in 13 min 26 s. See `tests/KNOWN_FAILURES.md`, 2026-09-14.
- [ ] **every window of `Lorenz63Dataset` shares one observation-noise draw** — found by the above.
      `oceanml3d/data/lorenz63.py:160` hoists `obs_seed = cfg.seed + 1` out of the window loop, so all windows
      get an identical noise realisation (measured: agreement to 1.9e-6, pure float cancellation).
      Every other dataset in the repo uses a per-window `cfg.seed + i * 100 + 1`. Effective sample
      size for anything observation-driven is one, whatever `num_windows` says — including the
      20-window S0/S1 gate. Deliberately **not** fixed alongside the test repair: the one-line change
      moves every L63 observation, invalidating the golden files and the published S0/S1 reference
      values, so it needs its own re-baselining run.
- [x] add the import-smoke test (imports every module) before any rename —
      `tests/test_import_smoke.py`, dynamic discovery via `pkgutil.walk_packages`, one test per
      module plus a guard against a wrong root making them all vacuously pass. Written and proven
      green *before* the namespace move, which is what it was for.

### 2. Correctness audit of the scientific core
- [x] `crps` returned the negation of the CRPS — fixed, `tests/test_crps.py`
- [x] `Strong4DVar`'s energy-score path had never executed — fixed (`_as_numpy`)
- [x] `JointCFM.compute_cfm_loss` crashed on any batch without `true_params`, the exact case its
      own guard four lines later exists to handle; `JointCFMCoupled` had the same shape of bug and
      now raises a `ValueError` naming why that model cannot train without parameters
- [ ] `energy_score` — pin against the closed form
- [ ] the four filters — EnKF beats the observations, spread-skill ≈ 1, EnKF ≈ ETKF on a linear
      Gaussian case
- [ ] two-scale Lorenz-96 — energy bounds, exponential divergence, the clamp
- [ ] `estimate_metrics.py`, `explained_variance`, SW component metrics

### 3. Structure (after 0-2)
- [x] **the `oceanml3d/` namespace and `pyproject.toml`** — `models/`, `data/`, `training/`,
      `evaluation/` and `conf/` moved under one package (first step of the target tree in
      `docs/REORG_PLAN.md`), 455 import statements rewritten across 128 files. The move made the
      project installable and killed the one import that could never survive an install:
      `evaluation/neural_inference.py` imported `model_factory` from the root script `train.py`,
      which only resolves when the repository root is on `sys.path`. It now lives in
      `oceanml3d/models/factory.py` and `train.py` re-exports it. `config/` and the driver scripts
      stay at the root — `@hydra.main(config_path=…)` resolves relative to the decorated file — so
      the wheel ships the library, not the CLI. Moving them in is the next packaging step.
- [x] `models/qg_base.py`: the 5 identical methods (−138 lines, proven neutral)
- [x] Hydra groups `paths/data/model/training/ablation/experiment` + `validate_config` — arrived
      with the transplant (§4). Caveat: they compose under `config/main.yaml`, a *second* Hydra root
      alongside `config/config.yaml`, and `validate_config` lives in the second of two config
      schemas. Unifying them is §5.
- [x] `@register_model` / `@register_dynamics` + entry points — `oceanml3d/registry.py` and
      `oceanml3d/dynamics/base.py`, seven models declared in `pyproject.toml` and verified to
      resolve. They cover the gridded family only; the toy/L96/QG models still go through
      `legacy/models/factory.py`, now in reserve (`oceanml3d/legacy/README.md`).
- [ ] one U-Net in `oceanml3d/models/nn/` — now **four**: `legacy/models/unet.py`, `legacy/models/direct_unet.py`,
      `models/ocean/nn/unet_nosc.py` (heads and attention; the pre-MONAI trunk, kept for
      reproduction) and `models/ocean/nn/unet_monai.py` (MONAI `DiffusionModelUNet`, the default
      gridded trunk since 2026-09-14). The MONAI wrapper is the plausible convergence point — the
      1-D adapter (`legacy/models/monai_unet_adapter.py`) already shares its resblock patch — but
      merging still needs the toy and gridded paths to agree on a channel convention.
- [ ] `oceanml3d/legacy/evaluation/baselines.py`: `Joint*` = 1 646 of 3 014 lines → `EnsembleFilter` +
      `JointEstimationMixin`, −800/−1 000 lines. **Unblocked**: filter tests green, reference values
      trustworthy. This is now the next structural task.
- [ ] the QG methods at 88-91% similarity (needs a judgement call on the formulations)

### 4. Transplants from the prototype — DONE 2026-09-14
- [x] `variables.py`, `catalog.py`, lazy xarray data layer, `patches.py`, `datamodule.py`, `augment.py`
- [x] `obs/`, `inference/export.py` + `product_contract.py`, `config_schema.py`, `scripts/prepare/`
- [x] **the port is in the tree, not in a sibling directory.** 120 files copied from
      `donor-prototype`, 36 import paths rewritten, ~77 tests added. The prototype is snapshotted at
      `../donor-prototype-snapshot-2026-09-14.tar.gz` and is no longer a dependency of anything.
      Four collisions, each resolved deliberately:
      * `models/fourdvarnet.py` (620-line L96 solver) vs the port's `models/fourdvarnet/` package
        (149-line gridded model) — a module cannot coexist with a package of the same name, and the
        classes are not interchangeable. The whole registry-based family moved to
        `oceanml3d/models/ocean/`, which removed every other model-level collision at once and left
        `models/__init__.py` empty, so no pre-existing import changed behaviour. The entry-point
        *group* is still `oceanml3d.models` — that string is a published contract, not a path.
      * `training/losses.py` — merged by hand; `StateMSELoss` and the NaN-aware weighted losses are
        two scales of one concept and the module docstring now says which path uses which.
      * `tests/conftest.py` — merged by hand; the port's synthetic-ocean session fixtures appended.
      * `tests/test_hydra_config.py` → the port's copy is `test_hydra_config_ocean.py` (the two test
        different config roots: `config/config.yaml` vs `config/main.yaml`).
      `lightning` was normalised to `pytorch_lightning` throughout, including the
      `pytest.importorskip` guards: both are installed and are the same code, but they are distinct
      module objects and a `lightning.LightningModule` fails an isinstance check in a
      `pytorch_lightning.Trainer`. One flavour per process.
- [x] `pyproject.toml` completed: `xarray`/`pandas`/`netCDF4`/`dask` promoted to hard dependencies,
      `zarr` and `prepare` extras added, the `oceanml3d` console script and the seven
      `oceanml3d.models` entry points declared. A wheel builds, ships only `oceanml3d/`, and every
      entry-point target resolves to the class the registry registers.
- [x] `product_contract.py` survived with its pinned SHA-256 intact
      (`dd44f0e198d0122b88207509801ed5c60a7e2fa36172151d89d78a622fe88846`), so the cross-repo
      contract with `oceanml3d-eval` still holds.
- [x] the one red test the graft brought in was adjudicated, not quarantined: it reproduces
      bit-identically in the donor, and the EnKF is correct — the window was ten steps, which is
      spin-up and nothing else. See `tests/KNOWN_FAILURES.md`, 2026-09-14.
- [ ] ensemble metrics additions (spread-skill correction, rank histogram) — these live in
      `oceanml3d-eval`, not here

### 5. Fold in the other repositories, then remove the scaffolding
- [x] NOSC — done via §4; it lives at `oceanml3d/models/ocean/nosc/` with its data pipeline,
      OSSE chain, weights and ablations. `docs/migration_nosc.md` is the module-by-module map.
- [ ] reanalyses data layer + CS1-CS4 — the *capabilities* are present and reimplemented (see
      `docs/feature_inventory.md` §2); what is missing is CS1-CS4 and the depth-embedding interior
      model
- [ ] deduplicate what the transplant doubled: two config schemas (`legacy/conf/schema.py` /
      `config_schema.py`, disjoint importers, two Hydra roots) and two dynamics abstractions
      (`legacy/models/dynamics.py` if/elif / `dynamics/base.py` registry). `docs/feature_inventory.md`
      §4 is the list. **Halved in cost, not solved**, by the 2026-09-14 move of the toy family to
      `oceanml3d/legacy/`: the two of each are now on opposite sides of a visible boundary
      (`oceanml3d/legacy/README.md`) instead of interleaved, so either can be retired without
      archaeology.
- [ ] move the evaluation drivers to `oceanml3d-eval`, one benchmark at a time
- [ ] delete the deprecated adapters
