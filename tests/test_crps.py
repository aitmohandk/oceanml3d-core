"""CRPS: checked against its closed form, not against another implementation.

The previous version returned the negation of the CRPS (pairwise term minus error term, and no
factor 1/2), which ranked a forecast biased by +10 as far better than a perfect one. It went
unnoticed because the only assertion on the function was `callable(crps)`. These tests pin the
value against the analytic expression and the ordering against forecasts whose ranking is known
a priori — an external source of truth rather than agreement with neighbouring code.
"""
import numpy as np
import pytest

from evaluation.metrics import crps

# CRPS of a N(0,1) predictive distribution against y = 0.
# Closed form: sigma * [ w (2 Phi(w) - 1) + 2 phi(w) - 1/sqrt(pi) ] with w = (y - mu)/sigma = 0,
# i.e. 2 phi(0) - 1/sqrt(pi) = 2/sqrt(2 pi) - 1/sqrt(pi) = 0.23369.
CRPS_STANDARD_NORMAL = 2 / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi)


def test_matches_the_closed_form_for_a_gaussian_ensemble():
    rng = np.random.default_rng(0)
    ens = rng.normal(0.0, 1.0, (2000, 50, 1))
    truth = np.zeros((50, 1))
    assert crps(ens, truth)[0] == pytest.approx(CRPS_STANDARD_NORMAL, abs=0.01)


def test_single_member_reduces_to_absolute_error():
    ens = np.full((1, 10, 2), 3.0)
    truth = np.full((10, 2), 1.0)
    assert crps(ens, truth) == pytest.approx([2.0, 2.0])


def test_perfect_forecast_scores_zero():
    truth = np.random.default_rng(1).standard_normal((20, 3))
    ens = np.repeat(truth[None], 5, axis=0)
    assert crps(ens, truth) == pytest.approx(np.zeros(3), abs=1e-12)


def test_ordering_accurate_beats_biased_beats_wild():
    rng = np.random.default_rng(2)
    truth = np.zeros((100, 1))
    accurate = rng.normal(0, 0.1, (200, 100, 1))
    calibrated = rng.normal(0, 1.0, (200, 100, 1))
    biased = calibrated + 10.0
    s = [crps(e, truth)[0] for e in (accurate, calibrated, biased)]
    assert s[0] < s[1] < s[2]
    assert all(v > 0 for v in s)          # CRPS is non-negative
    # a +10 bias costs the bias minus the spread term: 10 - 0.5 E|xi - xj| = 10 - 0.564
    assert s[2] == pytest.approx(10.0 - 1 / np.sqrt(np.pi), abs=0.1)


def test_fair_form_removes_the_ensemble_size_bias():
    rng = np.random.default_rng(3)
    truth = np.zeros((200, 1))
    small = rng.normal(0, 1, (5, 200, 1))
    large = rng.normal(0, 1, (500, 200, 1))
    biased_gap = abs(crps(small, truth)[0] - crps(large, truth)[0])
    fair_gap = abs(crps(small, truth, fair=True)[0] - crps(large, truth, fair=True)[0])
    assert fair_gap < biased_gap


def test_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="truth must have shape"):
        crps(np.zeros((4, 10, 2)), np.zeros((10, 3)))
