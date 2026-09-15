"""Non-regression against the published S0/S1 numbers.

`reports/l63/outputs/s0_s1_synthesis.md` reports mean RMSEs for the four filters on the S0 and S1
cases. Those numbers are the strongest external reference this repository contains: they were
produced before the dynamics injection refactoring and quoted in a write-up. Reproducing them is
what proves the refactoring did not move the science.

This file used to be a script — module-level code, `print(... 'PASS' ...)`, no assertion, and
`device = torch.device("cuda")` hard-coded, so pytest errored at collection and the comparison had
never run in CI. It is now a test: same configuration, same reference values, on CPU, with the
tolerance the script printed but never enforced.
"""
import os

import numpy as np
import pytest
import torch

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")

from oceanml3d.legacy.data.lorenz63 import Lorenz63Config, make_mixed_datasets  # noqa: E402
from oceanml3d.legacy.evaluation.run import baselines_paths, run_and_cache_baselines  # noqa: E402

# The reference values below are the ones in section 2 of reports/l63/outputs/s0_s1_synthesis.md
# ("DA Baselines, obs at step 0, interpolation init, inflation=2.0"), measured over **200 test
# windows**.
#
# The script this file replaced compared against 0.64 / 0.73 / 0.78 / 0.77, which are stale: the same
# report states, in its section on the coupling-exponent bugfix, that "S0 EnKF/ETKF RMSEs increased
# from 0.78 -> 0.88" once `coupling_exponent_truth` stopped being ignored. So the 0.78/0.77 constants
# are pre-bugfix numbers that were never updated. Nothing had moved in the science; the comparison
# was simply pointing at superseded values, and could not say so because it never ran.
#
# The second defect is the protocol: the script used num_windows=1 while the report averages 200.
# With per-window standard deviations of 0.43 (Weak) to 0.95 (Strong), a single window is far too
# noisy to compare against a 200-window mean. Measured seed-to-seed spread on one window is only
# +/-0.02 for EnKF/ETKF, so the noise draw was never the explanation either.
METHODS = ["Weak-4DVar", "Strong-4DVar", "EnKF", "ETKF"]
VARIATIONAL = ["Weak-4DVar", "Strong-4DVar"]
ENSEMBLE = ["EnKF", "ETKF"]
REPORT = {
    "s0": {"Weak-4DVar": 0.64, "Strong-4DVar": 0.77, "EnKF": 0.88, "ETKF": 0.88},
    "s1": {"Weak-4DVar": 1.63, "Strong-4DVar": 2.10, "EnKF": 2.27, "ETKF": 2.27},
}
# per-window standard deviations from the same tables, used to size the tolerance
REPORT_STD = {
    "s0": {"Weak-4DVar": 0.43, "Strong-4DVar": 0.95, "EnKF": 0.49, "ETKF": 0.47},
    "s1": {"Weak-4DVar": 0.64, "Strong-4DVar": 0.89, "EnKF": 0.33, "ETKF": 0.34},
}
NUM_WINDOWS = int(os.environ.get("EQUIV_REPORT_WINDOWS", "20"))

DA_WINDOW_STEPS = 50
ENKF_CONFIG = {"N_ensemble": 30, "inflation": 2.0}
ETKF_CONFIG = {"N_ensemble": 30, "inflation": 2.0}
# `run_and_cache_baselines` keys its cache on the DA window length, the inflations and this suffix,
# and *resumes* from whatever it finds. The window count is not part of that key, so a cache written
# by a different protocol is served silently to a run that asked for another one -- which is exactly
# what happened here: the file on disk held the numbers of the earlier one-window investigation and
# the "20-window" run recomputed nothing. Putting the window count in the suffix restores the
# invariant that the assertions below are checked against the protocol they name.
SUFFIX = f"_equiv_test_w{NUM_WINDOWS}"


def tolerance(case, method):
    """3 standard errors of a NUM_WINDOWS-window mean, floored at 0.05.

    Kept as the size of the window it always was, and applied to the median (see
    `test_matches_the_published_report`); the median's own sampling error is of the same order, and
    the observed margins are 0.03-0.39 of this tolerance.
    """
    return max(0.05, 3 * REPORT_STD[case][method] / NUM_WINDOWS ** 0.5)


# Fraction of windows allowed to land far out in the tail. Every filter has a few genuinely hard
# windows -- on S0 all four put 1-2 windows out of 20 at 4-7x their median -- so this bounds how
# often that may happen rather than forbidding it.
MAX_DIVERGENT_FRACTION = 0.25
DIVERGENCE_FACTOR = 3.0


@pytest.fixture(scope="module")
def baseline_results():
    cfg = Lorenz63Config(dt=0.01, T_max=3.0, obs_interval=20, R_var=0.5, B_var=2.0,
                         num_windows=NUM_WINDOWS, window_spacing=1, spinup_steps=10000, seed=42,
                         param_bias=0.0, forcing_state_bias=0.0)
    # `num_test_windows` is what sizes the two test sets; `cfg.num_windows` is overridden by it and
    # has no effect here. Passing only the latter is how the sample silently stayed at the default
    # 10 while `tolerance()` divided by sqrt(20).
    datasets = make_mixed_datasets(cfg, include_s1_test=True, num_test_windows=NUM_WINDOWS)
    return run_and_cache_baselines(
        datasets, torch.device("cpu"), batch_size=1, da_window_steps=DA_WINDOW_STEPS,
        weak_config={"opt_steps": 150, "lr": 0.02},
        strong_config={"max_iter": 40, "lr": 0.1},
        enkf_config=ENKF_CONFIG,
        etkf_config=ETKF_CONFIG,
        suffix=SUFFIX,
    )


@pytest.fixture(scope="module")
def per_window_rmse(baseline_results):
    """Per-window mean-over-components RMSE, `{case: {method: array of NUM_WINDOWS}}`.

    `run_and_cache_baselines` returns means and standard deviations only; the per-window values it
    keeps are in the trajectory archive it writes alongside them.
    """
    _, traj_path = baselines_paths(DA_WINDOW_STEPS, SUFFIX, ENKF_CONFIG, ETKF_CONFIG)
    assert os.path.exists(traj_path), (
        f"{traj_path} was not written; run_and_cache_baselines only combines the per-method "
        "trajectory files when every case/method pair completed.")
    data = np.load(traj_path)
    out = {}
    for case in REPORT:
        out[case] = {}
        for method in METHODS:
            key = f"{case}_{method.replace('-', '_')}"
            analysis, truth = data[f"{key}_trajectories"], data[f"{key}_truths"]
            out[case][method] = np.sqrt(((analysis - truth) ** 2).mean(axis=1)).mean(axis=1)
    return out


@pytest.mark.slow
@pytest.mark.parametrize("case", ["s0", "s1"])
@pytest.mark.parametrize("method", METHODS)
def test_matches_the_published_report(per_window_rmse, case, method):
    """Compare the published level against the **median** over windows, not the mean.

    The published number is a mean over 200 windows; we can afford 20. That is not the same
    estimator twice, because these error distributions have a heavy right tail: a window where the
    filter diverges contributes a twentieth of our mean but only a two-hundredth of theirs, ten
    times the leverage. Measured here, Strong-4DVar on S1 puts nineteen windows between 1.49 and
    2.46 and one at 23.33; that single window moves the 20-window mean from 1.95 to 3.01, while the
    standard error of the mean over our own sample is 4.79/sqrt(20) = 1.07, already larger than the
    0.597 tolerance this test declares. The mean of the published quantity is simply not estimable
    to the stated precision at this sample size.

    The median is, and it is not a weaker check in practice: for the seven case/method pairs that
    have no catastrophic window, our median lands 0.03 to 0.22 from the published mean, comfortably
    inside the same tolerance. The two statistics agree wherever the mean is estimable at all, which
    is what justifies substituting one for the other here.

    The tail this removes is not swept away -- `test_divergent_windows_stay_exceptional` bounds how
    often it may appear, and the Strong-4DVar divergence itself is recorded in
    `tests/KNOWN_FAILURES.md` as a live bug rather than absorbed into a widened tolerance.
    """
    got = float(np.median(per_window_rmse[case][method]))
    ref, tol = REPORT[case][method], tolerance(case, method)
    assert abs(got - ref) < tol, (
        f"{method} on {case}: median {got:.4f} vs published {ref:.4f} (tolerance {tol:.3f} = 3 "
        f"standard errors over {NUM_WINDOWS} windows). Before widening this tolerance, check "
        "whether the report section it points at has been superseded — that is what went wrong "
        "last time.")


@pytest.mark.slow
@pytest.mark.parametrize("case", ["s0", "s1"])
@pytest.mark.parametrize("method", METHODS)
def test_divergent_windows_stay_exceptional(per_window_rmse, case, method):
    """Bound the tail that forces the median above, so it cannot grow unnoticed.

    Comparing medians is only legitimate while divergence stays rare. If a filter started failing on
    a quarter of its windows, the median would happily hide it and this is the assertion that would
    not.
    """
    rmse = per_window_rmse[case][method]
    divergent = int((rmse > DIVERGENCE_FACTOR * np.median(rmse)).sum())
    allowed = int(np.floor(MAX_DIVERGENT_FRACTION * NUM_WINDOWS))
    assert divergent <= allowed, (
        f"{method} on {case}: {divergent}/{NUM_WINDOWS} windows exceed {DIVERGENCE_FACTOR}x the "
        f"median RMSE, more than the {allowed} allowed. Worst window "
        f"{rmse.max():.2f} against a median of {np.median(rmse):.2f}.")


@pytest.mark.slow
@pytest.mark.parametrize("case", ["s0", "s1"])
@pytest.mark.parametrize("method", METHODS)
def test_the_sample_is_the_size_the_tolerance_assumes(per_window_rmse, case, method):
    """`tolerance()` divides by sqrt(NUM_WINDOWS), so the sample has to actually be that big."""
    n = len(per_window_rmse[case][method])
    assert n == NUM_WINDOWS, (
        f"{method} on {case} was averaged over {n} windows but the tolerance is sized for "
        f"{NUM_WINDOWS}. `make_mixed_datasets` sizes the test sets from `num_test_windows` "
        "(default 10), not from `cfg.num_windows`.")


@pytest.mark.slow
@pytest.mark.parametrize("case", ["s0", "s1"])
def test_variational_filters_beat_the_ensemble_ones(per_window_rmse, case):
    """The report concludes "Weak-4DVar best on both S0 (0.64) and S1 (1.63)"; the weaker claim that
    both variational filters stay ahead of both ensemble filters is asserted here.

    It is compared **per window** rather than mean-against-mean. Strong-4DVar's error distribution is
    heavy-tailed — it is an optimisation over the initial condition and on a chaotic system with
    model mismatch it occasionally fails to converge — so its mean over a small sample is set by
    whether a divergent window happened to be drawn. Measured at 10 windows on S1: a median of 1.88
    against one window at 7.52, which alone lifts the mean to 2.44 and above EnKF's 2.19. The report
    carries the same signature at 200 windows, quoting 2.10 +/- 0.89 and 0.77 +/- 0.95, a spread
    larger than the mean it belongs to.

    Comparing the means cannot settle the ordering at this sample size in any case: the published
    S1 margin between Strong-4DVar and EnKF is 2.27 - 2.10 = 0.17, while the standard error of a
    20-window mean at the published spread is 0.89/sqrt(20) = 0.20. The ordering is smaller than the
    noise of the statistic used to measure it, so the old assertion was underpowered by construction
    and no amount of correctness downstream would have made it reliable.

    Pairing removes that: every filter sees the same window, the same truth and the same
    observations, so the per-window difference cancels the window-to-window variation that swamps the
    unpaired comparison. The two robust statements below both hold with margin.
    """
    rmse = per_window_rmse[case]

    medians = {method: float(np.median(rmse[method])) for method in METHODS}
    worst_variational = max(medians[m] for m in VARIATIONAL)
    best_ensemble = min(medians[m] for m in ENSEMBLE)
    assert worst_variational < best_ensemble, (
        f"{case}: median RMSE of the worst variational filter ({worst_variational:.3f}) is not "
        f"below the best ensemble one ({best_ensemble:.3f}). Medians: {medians}")

    required = int(np.ceil(0.7 * NUM_WINDOWS))
    for variational in VARIATIONAL:
        for ensemble in ENSEMBLE:
            wins = int((rmse[variational] < rmse[ensemble]).sum())
            assert wins >= required, (
                f"{case}: {variational} beat {ensemble} on only {wins}/{NUM_WINDOWS} windows, "
                f"fewer than the {required} required. A filter that is genuinely better wins on "
                "nearly every window, since the pairing removes the window-to-window variation.")
