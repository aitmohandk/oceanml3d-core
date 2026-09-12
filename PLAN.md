# oceanml3d-core — Plan

Continuation of `4dvarnet-fm-opencode` (see `PROVENANCE.md`). The upstream plan is kept as
`PLAN_upstream.md`.

## 0. The blocking question — RESOLVED, no GPU needed
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

## 1. Safety net — mostly done
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
- [ ] **every window of `Lorenz63Dataset` shares one observation-noise draw** — found by the above.
      `data/lorenz63.py:160` hoists `obs_seed = cfg.seed + 1` out of the window loop, so all windows
      get an identical noise realisation (measured: agreement to 1.9e-6, pure float cancellation).
      Every other dataset in the repo uses a per-window `cfg.seed + i * 100 + 1`. Effective sample
      size for anything observation-driven is one, whatever `num_windows` says — including the
      20-window S0/S1 gate. Deliberately **not** fixed alongside the test repair: the one-line change
      moves every L63 observation, invalidating the golden files and the published S0/S1 reference
      values, so it needs its own re-baselining run.
- [ ] add the import-smoke test (imports every module) before any rename

## 2. Correctness audit of the scientific core
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

## 3. Structure (after 0-2)
- [x] `models/qg_base.py`: the 5 identical methods (−138 lines, proven neutral)
- [ ] Hydra groups `paths/data/model/training/ablation/experiment` + `validate_config`
- [ ] `@register_model` / `@register_dynamics` + entry points
- [ ] one U-Net in `models/nn/`
- [ ] `evaluation/baselines.py`: `Joint*` = 1 646 of 3 014 lines → `EnsembleFilter` +
      `JointEstimationMixin`, −800/−1 000 lines. **Unblocked**: filter tests green, reference values
      trustworthy. This is now the next structural task.
- [ ] the QG methods at 88-91% similarity (needs a judgement call on the formulations)

## 4. Transplants from the prototype
- [ ] `variables.py`, `catalog.py`, lazy xarray data layer, `patches.py`, `datamodule.py`, `augment.py`
- [ ] `obs/`, `inference/export.py` + `product_contract.py`, `config_schema.py`, `scripts/prepare/`
- [ ] ensemble metrics additions (spread-skill correction, rank histogram)

## 5. Fold in the other repositories, then remove the scaffolding
- [ ] reanalyses data layer + CS1-CS4, NOSC as `models/nosc` (strip its committed data first)
- [ ] move the evaluation drivers to `oceanml3d-eval`, one benchmark at a time
- [ ] delete the deprecated adapters
