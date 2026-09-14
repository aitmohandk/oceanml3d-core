"""The dynamics classes must reproduce the inline formulas they replaced.

This file used to be a script: module-level code, `print(... 'PASS' if diff < 1e-10 ...)`, no
assertion, and a call `dynamics.step(s, W, sigma, rho, beta)` against a signature that takes
`(state, forcing, **kwargs)`. pytest executed it at collection time, so it errored out and its
verdicts were never checked by anyone. Since its whole purpose is to prove that a refactoring changed
nothing, it is now real tests: the same comparisons, with assertions and a fixed seed.
"""
import numpy as np
import pytest
import torch

from oceanml3d.data.lorenz63 import generate_long_trajectory, generate_observations
from oceanml3d.evaluation.baselines import ETKF, EnKF, Strong4DVar, Weak4DVar
from oceanml3d.models.lorenz63_dynamics import Lorenz63Dynamics, _apply_coupling

DT = 0.01
NUM_STEPS = 60
SIGMA, RHO, BETA = 10.0, 28.0, 8 / 3
C1 = 1.0
COUPLING_EXPONENT = 1.6
R_VAR = 0.5
TOL = 1e-10


@pytest.fixture(scope="module")
def trajectory():
    device = torch.device("cpu")
    full = generate_long_trajectory(
        num_steps=NUM_STEPS + 2000, dt=DT, seed=42, sigma=SIGMA, rho=RHO, beta=BETA,
        gamma=0.05, W_L_bar=0.0, c1=C1, c2=0.1, sigma_0=0.08, sigma_L=0.20,
        coupling_exponent=COUPLING_EXPONENT, device=device)
    state = full[-NUM_STEPS:, :3]
    forcing = full[-NUM_STEPS:, 3]
    obs, obs_mask = generate_observations(state, obs_interval=1, R_var=R_VAR, seed=43, device=device)
    return {"state": state, "forcing": forcing, "obs": obs, "obs_mask": obs_mask,
            "dynamics": Lorenz63Dynamics(dt=DT, coupling_exponent=COUPLING_EXPONENT)}


def _inline_step(s, w):
    """The formula the class replaced, kept here as the reference."""
    X, Y, Z = s[..., 0], s[..., 1], s[..., 2]
    coupling = _apply_coupling(w, C1, COUPLING_EXPONENT)
    dX = SIGMA * (Y - X) + coupling
    dY = X * (RHO - Z) - Y
    dZ = X * Y - BETA * Z
    return torch.stack([X + dX * DT, Y + dY * DT, Z + dZ * DT], dim=-1)


def test_single_step_matches_the_inline_formula(trajectory):
    s = trajectory["state"][0].unsqueeze(0)
    w = trajectory["forcing"][0].unsqueeze(0)
    from_class = trajectory["dynamics"].step(s, w, sigma=SIGMA, rho=RHO, beta=BETA)
    assert (from_class - _inline_step(s, w)).abs().max().item() < TOL


def test_rollout_matches_the_inline_formula(trajectory):
    s = trajectory["state"][0].unsqueeze(0)
    forcing = trajectory["forcing"]
    from_class, inline = [s], [s]
    a = b = s
    for t in range(1, NUM_STEPS):
        a = trajectory["dynamics"].step(a, forcing[t - 1:t], sigma=SIGMA, rho=RHO, beta=BETA)
        b = _inline_step(b, forcing[t - 1:t])
        from_class.append(a)
        inline.append(b)
    diff = (torch.cat(from_class) - torch.cat(inline)).abs().max().item()
    assert diff < TOL


@pytest.mark.parametrize("name", ["weak4dvar", "strong4dvar", "enkf", "etkf"])
def test_filters_produce_finite_analyses(trajectory, name):
    """Each filter, given the shared dynamics, must return a finite trajectory and finite RMSE."""
    common = dict(dt=DT, device=torch.device("cpu"), coupling_exponent=COUPLING_EXPONENT,
                  dynamics=trajectory["dynamics"])
    filters = {
        "weak4dvar": lambda: Weak4DVar(da_window_steps=NUM_STEPS, opt_steps=5, **common),
        "strong4dvar": lambda: Strong4DVar(da_window_steps=NUM_STEPS, max_iter=5, **common),
        "enkf": lambda: EnKF(N_ensemble=20, inflation=1.0, **common),
        "etkf": lambda: ETKF(N_ensemble=20, inflation=1.0, **common),
    }
    result = filters[name]().assimilate(
        trajectory["obs"], trajectory["obs_mask"], trajectory["forcing"], trajectory["state"],
        sigma=SIGMA, rho=RHO, beta=BETA, c1=C1)
    assert np.isfinite(result.trajectory).all(), f"{name} produced non-finite states"
    assert np.isfinite(result.rmse).all() and (np.asarray(result.rmse) >= 0).all()
