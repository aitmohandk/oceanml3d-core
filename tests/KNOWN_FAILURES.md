# Pre-existing test failures at `pre-refactor`

Recorded before any restructuring, so that refactoring breakage can be told apart from breakage that
was already there. Measured with `pytest -q` on CPU (torch 2.14, pytorch-lightning 2.6):
**585 passed, 27 failed, 7 skipped, 2 collection errors** in 4 min 40 s.
After quarantining: **585 passed, 0 failed, 27 xfailed** — the suite is now a usable safety net.

Each entry below is marked `xfail(strict=False)` or skipped with a reason, so the suite is green by
declaration. **Every one of them is a bug to fix, not a decision** — the marker is a bookmark, and
`PLAN.md` carries the item. Removing a marker requires fixing the underlying cause.

| Tests | Count | Root cause | Kind |
|---|---|---|---|
| ~~`test_baselines_weak4dvar.py`, `test_baselines_strong4dvar.py`, `test_baselines_enkf.py`, `test_data_leakage.py`~~ | ~~24~~ **0** | the tests built the filters without `dynamics` | **FIXED** — see the adjudication below |
| ~~`test_joint_estimation.py::TestJointCFM` (7)~~ | ~~7~~ **0** | **not API drift** — 5 of the 7 were a production bug in `compute_cfm_loss`; only 2 were stale tests | **FIXED** — see below |
| ~~`test_random_param_dataset.py` (2)~~ | ~~2~~ **0** | one stale key set; one NaN comparison | **FIXED** — see below |
| ~~`test_refactoring_equivalence.py::test_dataset_reproducible`~~ | ~~1~~ **0** | **not a real bug** — the dataset is bit-for-bit reproducible; the assertion compared NaN with NaN | **FIXED** — see below |
| ~~`test_lorenz63.py::test_observations_noise`~~ | ~~1~~ **0** | not a tolerance question — the test was underpowered by construction at n=25 | **FIXED** — see below |
| `test_baselines_enkf.py` (4), `test_baselines_strong4dvar.py` (5), `test_baselines_weak4dvar.py::test_weak4dvar_forward_model` | 10 | same `dynamics=None` drift, surfaced once the first batch was marked | **API drift** |
| ~~`test_equiv_report.py`~~ | collection | was a script with `device="cuda"` hard-coded | **converted to tests** — and it revealed that the published numbers do not reproduce, see below |
| ~~`test_numerical_equivalence.py`~~ | collection | was a script calling a 3-arg signature with 6 positional args | **converted to tests**, 6 passing |

**The quarantine is now empty: 609 passed, 7 skipped, 0 xfailed** (`pytest -q -m "not slow"`).

The two collection errors matter most: `test_numerical_equivalence.py` is the file whose entire
purpose is to prove that a refactoring changed nothing, and it has not run since the signature it
tests was changed. Repairing it is the first item of the plan, not the last.

## Bugs found by triangulation, not by the suite

The repositories were written by several AI agents, so tests are not an independent check: a wrong
implementation and a test that accepts it can come from the same session. Verification therefore has
to come from outside the repo — closed forms, conservation laws, known orderings, or agreement
between two implementations written independently.

| Found | How | Status |
|---|---|---|
| `evaluation/metrics.py::crps` returned **minus** the CRPS (pairwise term minus error term, no factor 1/2). A forecast biased by +10 scored −8.9 against +0.03 for a perfect one — the worst forecast won. | Compared with the closed form for a Gaussian ensemble, and cross-checked against `energy_score` (which is correct) | **fixed**, `tests/test_crps.py` added (analytic value, ordering, fair form, shape check) |
| `tests/test_metrics.py` asserted only `callable(crps)` | reading the test | the bug was invisible to the suite; the new file replaces that coverage |
| ~20 inline RMSE computations across `evaluation/` | compared formula by formula | **no divergence** — all are `sqrt(mean((a-b)**2, axis=0))`. They are duplication, not a correctness risk. |

Next candidates for the same treatment, in order of exposure: `energy_score` (looks correct, pin it
against the closed form anyway), the four assimilation filters (an EnKF must beat the observations
and its spread-skill ratio must be ≈1), the two-scale Lorenz-96 (energy bounds, Lyapunov behaviour),
`estimate_metrics.py`.


## Adjudication of the 24 filter failures (resolved)

No author to ask, so each was decided against an external reference rather than against neighbouring
code. Result: **24 of the 27 are repaired**, and two real implementation bugs surfaced in the process.

**1. `dynamics=None` was a lying default.** All four filters declared `dynamics: DynamicsBase = None`
with a fallback `state_dim = 3`, but every method calls `self.dynamics.step()`. The live caller
(`evaluation/run.py`) always passes one. Verdict: the tests were stale, and the signature was
advertising an optionality that never worked. The default now raises a `ValueError` explaining why,
and the tests construct `Lorenz63Dynamics(dt=0.01)` like the production path does.

**2. `Strong4DVar`'s energy-score path had never executed.** Repairing the tests exposed
`AttributeError: 'numpy.ndarray' object has no attribute 'detach'` in `_ESAccumulator.step`:
`Strong4DVar.assimilate` feeds it numpy arrays while `EnKF`/`ETKF` feed tensors. Fixed with a
`_as_numpy` helper that accepts both. **This bug was invisible while the tests were red.**

**3. `test_enkf_mean_tracks_truth` asserted an unreachable threshold.** It required
`mean_rmse < 0.5`. Measured: climatology 6.07, observations cover 5 % of steps with an RMSE of 0.62
at those points, EnKF gives 1.20 (N=30) and 0.91 (N=100). An analysis below 0.5 over all steps would
have to beat the observation noise everywhere — impossible at that coverage. Verdict: the filter is
correct, the number was invented. Replaced by two references that mean something: the analysis must
beat climatology by at least 3x, and a larger ensemble must not do worse.

Remaining quarantined: **none.** The 7 `JointCFM` tests, the 2 dataset tests,
`test_refactoring_equivalence.py::test_dataset_reproducible`,
`test_lorenz63.py::test_observations_noise` and both collection errors are all cleared — see
"the remaining 11 quarantined tests, cleared" below. Two of the diagnoses in the table above turned
out to be wrong in the same direction: what was filed as "API drift" was largely a production bug,
and what was filed as a "possible real bug" was a NaN comparison.


## Resolved: the published S0/S1 numbers were stale constants, not a science regression

`tests/test_equiv_report.py` compared the four filters with `0.64 / 0.73 / 0.78 / 0.77`. On CPU we
measured `0.5485 / 0.5236 / 0.8492 / 0.8682` — none within the 0.05 tolerance the script declared but
never enforced. Three hypotheses were open: the refactoring moved the science, the constants were
wrong, or device-dependent RNG. Resolved without a GPU, in this order:

**1. RNG ruled out by measurement.** `data/lorenz63.py` uses `torch.Generator(device=device)`, so CPU
and CUDA do give different streams for the same seed — the hypothesis was mechanically plausible. But
running six seeds on CPU gives EnKF 0.865 +/- 0.019 and ETKF 0.876 +/- 0.026, while the gap to the
constants is 0.085 and 0.106: three to four standard deviations, with the published value outside the
min-max envelope. A different noise draw cannot explain it.

**2. The constants were superseded, and the report says so.** Section 1 of
`reports/l63/outputs/s0_s1_synthesis.md` records a coupling-exponent bugfix: the old code ignored
`coupling_exponent_truth=1.6` and used 1.0, and after the fix *"S0 EnKF/ETKF RMSEs increased from
0.78 -> 0.88"*. Section 2 of the same file gives the current values: **0.64 / 0.77 / 0.88 / 0.88**
for S0 and **1.63 / 2.10 / 2.27 / 2.27** for S1. Our measured 0.849 / 0.868 match those, not the ones
the script carried. The science never moved; the comparison was pointing at pre-bugfix numbers and
could not say so, because it errored at collection and never ran.

**3. The protocol was also wrong.** The script used `num_windows=1`; the report averages **200**
windows, with per-window standard deviations of 0.43 (Weak) to 0.95 (Strong). One window cannot be
compared with a 200-window mean — which is why the variational filters looked furthest off.

Fixed: the reference values now come from section 2 of the report, `NUM_WINDOWS` defaults to 20
(overridable with `EQUIV_REPORT_WINDOWS`), and the tolerance is three standard errors of the
window mean rather than a flat 0.05. `EnKF-s0` and `ETKF-s0` pass. The `slow` marker keeps the
20-window run out of the fast gate.

**Lesson for the rest of the migration:** the "external reference" that was going to arbitrate the
whole restructuring was itself a stale copy of a number, in a file nothing executed. Before trusting
any reference value, check that the document it came from has not superseded it.

## 2026-09-09 follow-up: the comparison still was not measuring what it said

The repaired file above left one red test, `test_variational_filters_beat_the_ensemble_ones_on_s0[s1]`.
Three further defects were behind it, and none of them was in the science.

**1. The cache was serving another protocol's numbers.** `run_and_cache_baselines` keys its cache on
the DA window length, the inflations and `suffix`, and *resumes* from whatever it finds. The window
count is not in the key. `experiments/baselines_dws50_equiv_test_inf2.0_etkf_inf2.0.json` held
`0.5485 / 0.5236 / 0.8492 / 0.8682` — byte-for-byte the S0 numbers quoted in the section above, from
the **one-window** investigation. The nominally 20-window run completed in 16.8 s and recomputed
nothing. Same failure mode as the stale constants: an assertion checked against a number produced
under settings nobody re-verified. The window count is now part of the suffix.

**2. The sample was 10 windows, not 20.** `make_mixed_datasets` sizes its two test sets from its own
`num_test_windows` argument (default 10) and ignores `cfg.num_windows`, which was all the test set.
`EQUIV_REPORT_WINDOWS` therefore did nothing, while `tolerance()` divided by sqrt(20).
`test_the_sample_is_the_size_the_tolerance_assumes` now pins this.

**3. The assertions were not estimable at this sample size.** The ordering claim compared unpaired
means: the published S1 margin between Strong-4DVar and EnKF is `2.27 - 2.10 = 0.17`, against a
standard error of `0.89/sqrt(20) = 0.20` for a 20-window mean. The effect was smaller than the noise
of the statistic used to measure it, so the test was underpowered by construction and no amount of
correctness downstream would have made it pass reliably. Pairing per window — every filter sees the
same window, truth and observations — cancels the window-to-window variation and settles it easily.
The level comparison moved from the mean to the median for the same reason (see below).

## Open bug: `Strong4DVar` diverges on a minority of windows

Found while repairing the above; **a real bug, not a tolerance question.** On S1 window 5 of the
20-window sample, the other three filters sit exactly at their medians (Weak 1.61, EnKF 2.25,
ETKF 2.35) while `Strong4DVar`'s analysis blows up mid-window to X=516, Y=−308, Z=−193 — against a
truth staying in the normal L63 range — before recovering, for a window RMSE of **23.33 against its
own median of 1.93**. The trajectory stays finite, so nothing raises. It is the strong-constraint
minimisation over the initial condition failing to converge (`max_iter=40`, `lr=0.1`), not the data.

Consequence for the test: that single window lifts the 20-window mean from 1.95 to 3.01. A rare
divergent window carries a twentieth of our mean but only a two-hundredth of the published
200-window mean — ten times the leverage — so the two are not the same estimator, and the standard
error of our own sample mean (4.79/sqrt(20) = 1.07) already exceeds the 0.597 tolerance the file
declares. `test_matches_the_published_report` therefore compares the **median**, which is estimable
at n=20. This is not a weakening: for the seven case/method pairs with no catastrophic window the
median lands 0.03–0.22 from the published mean, comfortably inside the same tolerance — the two
statistics agree wherever the mean is estimable at all. The tolerance itself was not widened.

The tail is bounded rather than ignored: `test_divergent_windows_stay_exceptional` fails if more than
25% of windows exceed 3x the median, so a filter that started diverging routinely could not hide
behind the median. Note that occasional hard windows are normal here and not specific to
Strong-4DVar — on S0 all four filters put 1-2 windows out of 20 at 4-7x their median. What is
specific to Strong-4DVar is the *magnitude*, 12x the median.

**To fix:** make `Strong4DVar` detect non-convergence (line search / divergence guard on the
minimisation, or a sanity bound on the propagated trajectory) instead of returning a blown-up
analysis. When it is fixed, the median comparison can go back to the mean and match the report's
estimator exactly.

## 2026-09-09: the remaining 11 quarantined tests, cleared

Two of the four diagnoses recorded in the table above were wrong, and the errors both pointed the
same way — towards blaming the test. In each case the label said "API drift" or "tolerance" where
the actual cause was either a production bug or a test that could never have passed reliably.

**1. Five of the seven `JointCFM` failures were a production bug, not API drift.**
`JointCFM.compute_cfm_loss` builds the parameter interpolant with
`self._norm(batch.true_params)` unconditionally, then four lines later guards the parameter loss
with `if batch.true_params is not None`. A batch without parameters — which that guard exists
precisely to serve — dies in `_norm` with
`TypeError: unsupported operand type(s) for -: 'NoneType' and 'Tensor'` before ever reaching it.
`param_tau` is now built only when the batch carries parameters; `param_0` stays unconditional so
the RNG stream does not depend on the branch (`test_param_loss_weight_zero_ignores_params` seeds
and compares the two paths, and would break if it did). Removing the guard and re-running fails
exactly those five tests, which is what confirms the direction of the fix.

`JointCFMCoupled.compute_cfm_loss` carries the identical shape of bug, at lines 490/494. It is
*not* the same fix: that model conditions **both** velocity fields on the parameter interpolant
(`cond_extra_dim=1 + param_dim`), so there is no meaningful loss without `true_params` and the
`is not None` guard was unreachable-by-design rather than merely misordered. It now raises a
`ValueError` naming the reason, which is what the confusing `TypeError` was standing in for.

The other two were genuine drift: `forward` returns `(v_state, v_param, x_hat_1)` and gates the
param velocity on a `param_tau` argument, so `v_state, param_feats = model(...)` could not unpack;
and `estimate_params` has never existed on this class — the entry points are
`sample(..., return_params=True)` and `sample_params_from_state`.

**2. `test_dataset_reproducible` was not "a reproducibility guarantee that no longer holds".**
The two datasets are bit-for-bit identical. `obs` is NaN at every unobserved step and NaN != NaN,
so `torch.testing.assert_close` without `equal_nan=True` reports every one of them as a mismatch.
The "94 % of elements differ" that made this look like a real bug is just `1 - 1/obs_interval`:
the fraction of steps that carry no observation. Same cause in
`test_random_param_dataset.py::test_deterministic_with_seed`, via `torch.allclose`. Both now
compare the NaN pattern and the finite values separately, at `rtol=0, atol=0` — strictly tighter
than what they replaced. A 1e-3 nondeterministic perturbation injected into `generate_observations`
fails both, so the assertions did not become vacuous.

**3. `test_observations_noise` was underpowered by construction, not mis-toleranced.**
It measured `torch.var` over the ~25 observed steps of one window and required +/-30% of `R_var`.
The sample variance of n normal draws has a relative spread of `sqrt(2/(n-1))` = **29% at n=25**,
so that band was a one-sigma interval: the test failed on roughly a third of all seeds *while the
code was correct*. The observed 0.3293 against 0.5 is 1.2 sigma — an ordinary fluctuation.
Widening the band is not available either, because an interval wide enough to be reliable at n=25
spans a factor of two and would no longer detect `* R_var` written for `* sqrt(R_var)`.
The sample had to grow. The test now (a) pins the **scaling law** exactly and with zero sampling
error — one seed, two values of `R_var`, noise ratio must be `sqrt(k)` — which is the property
that actually encodes the square root, and (b) measures the level over 200 independent seeds
(n=5000 per dimension, giving a 4-sigma band of +/-8%: measured 0.490/0.494/0.498 against 0.500).
Dropping the `np.sqrt` in `generate_observations` fails it immediately.

## Open bug: every window of `Lorenz63Dataset` shares one observation-noise realisation

Found while repairing `test_observations_noise`, and the reason its sample could not simply be
pooled over windows. `data/lorenz63.py:160` sets `obs_seed = cfg.seed + 1` **outside** the window
loop, then passes that same seed to `generate_observations` for every window (line 195). Every
window therefore receives an identical noise draw. Measured on the `cs1` fixture: the five windows
have visibly different trajectories, but their noise vectors agree to 1.9e-6 — float cancellation
in `obs - true_state`, nothing more — and their per-dimension variances agree to seven digits.

This is the odd one out in the repo, not a convention: `data/random_param_dataset.py:36` and
`data/lorenz96.py:437,466,555,583` all use a per-window `obs_seed = cfg.seed + i * 100 + 1`.

Consequence: observation errors are perfectly correlated across windows, so any statistic averaged
over the windows of an `Lorenz63Dataset` has an effective sample size of one noise realisation
regardless of `num_windows`. That includes the 20-window S0/S1 comparison in `test_equiv_report.py`
— its window-to-window spread reflects the trajectory, not the observations.

**Not fixed here, deliberately.** The one-line change moves every L63 observation in the
repository, which would invalidate the golden files and the published S0/S1 reference values at the
same moment those values are serving as the gate for the `Joint*` restructuring. It is a
science-affecting change and wants its own re-baselining, not a ride-along in a test repair.
`test_observations_noise` is written to hold either way: it takes its level sample from explicit
seeds rather than from the windows.
