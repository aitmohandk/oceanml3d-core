"""Ported from NOSC first_implementation tests/test_loss_and_modes.py (numpy only)."""
import numpy as np
import pytest

from oceanml3d.training.loss_grouping import combine_grouped_losses
from oceanml3d.training.vertical_modes import eofs_from_profiles

NAMES = ["zos", "t_d00", "t_d01", "t_d02", "u_d00", "u_d01", "u_d02"]
GROUPS = {"zos": "ssh", "t_d00": "T", "t_d01": "T", "t_d02": "T", "u_d00": "U", "u_d01": "U", "u_d02": "U"}


def test_flat_sum_is_plain_sum():
    losses = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    total, per = combine_grouped_losses(losses, NAMES, GROUPS, "flat_sum")
    assert total == 28.0 and per == {"all": 28.0}


def test_group_mean_balances_families():
    losses = [1.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0]
    total, per = combine_grouped_losses(losses, NAMES, GROUPS, "group_mean")
    assert per == {"ssh": 1.0, "T": 2.0, "U": 4.0} and total == 7.0
    total_w, _ = combine_grouped_losses(losses, NAMES, GROUPS, "group_mean", {"ssh": 3.0})
    assert total_w == 9.0


def test_group_mean_needs_groups():
    with pytest.raises(ValueError):
        combine_grouped_losses([1.0], ["a"], None, "group_mean")


def test_eofs_orthonormal_ordered_reconstruct():
    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.standard_normal((6, 6)))[0]
    coeffs = rng.standard_normal((500, 6)) * np.array([5, 3, 2, 1, 0.5, 0.1])
    prof = coeffs @ basis.T
    comp, evr = eofs_from_profiles(prof, 4)
    assert comp.shape == (4, 6)
    assert np.allclose(comp @ comp.T, np.eye(4), atol=1e-8)
    assert np.all(np.diff(evr) <= 0) and evr.sum() > 0.95
    rec = (prof - prof.mean(0)) @ comp.T @ comp + prof.mean(0)
    assert np.sqrt(((rec - prof) ** 2).mean()) < 0.6


def test_eofs_reject_underdetermined():
    with pytest.raises(ValueError):
        eofs_from_profiles(np.random.rand(3, 6), 2)
