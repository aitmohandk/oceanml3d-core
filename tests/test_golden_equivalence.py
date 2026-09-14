"""Behaviour recorded before the restructuring must survive it.

Golden files are produced by `python tests/golden/generate.py` on the pre-refactor code and
committed. Any merge of duplicated classes is proven here, not asserted in a commit message.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

GOLDEN = Path(__file__).resolve().parent / "golden"
TOL = 1e-10   # identical code path expected: float64 bit-for-bit up to reassociation


def load(case):
    p = GOLDEN / f"{case}.npz"
    if not p.exists():
        pytest.skip(f"no golden file for {case}: run python tests/golden/generate.py")
    return dict(np.load(p, allow_pickle=False))


def _check_qg(dyn, g, keys):
    s0 = torch.as_tensor(g["state0"])
    traj = dyn.rollout_trajectory(s0, steps=g["trajectory"].shape[0] - 1)
    assert np.abs(traj.detach().numpy() - g["trajectory"]).max() < TOL
    wind = torch.as_tensor(g["wind_state"])
    assert np.abs(dyn.wind_curl_field(wind).detach().numpy() - g["wind_curl"]).max() < TOL
    for method, key in keys:
        got = getattr(dyn, method)(traj).detach().numpy()
        assert np.abs(got - g[key]).max() < TOL, key


def test_qg_2layer_unchanged():
    from oceanml3d.legacy.models.qg_dynamics import QGDynamics

    g = load("qg_2layer")
    _check_qg(QGDynamics(nx=32, dt=7200.0), g, [("kinetic_energy", "ke"), ("enstrophy", "enstrophy")])


def test_qg_1layer_unchanged():
    from oceanml3d.legacy.models.qg1l_dynamics import QG1LDynamics

    g = load("qg_1layer")
    _check_qg(QG1LDynamics(nx=32, dt=7200.0), g, [("kinetic_energy", "ke"), ("enstrophy", "enstrophy")])


def test_lorenz63_unchanged():
    from oceanml3d.legacy.models.lorenz63_dynamics import Lorenz63Dynamics

    g = load("lorenz63")
    d = Lorenz63Dynamics(dt=0.01, c1=0.0, clip_range=None)
    s = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)
    W = torch.zeros(1, dtype=torch.float64)
    out = [s.numpy().copy()]
    for _ in range(g["trajectory"].shape[0] - 1):
        s = d.step(s, W, sigma=10.0, rho=28.0, beta=8.0 / 3.0)
        out.append(s.numpy().copy())
    assert np.abs(np.concatenate(out) - g["trajectory"]).max() < TOL
