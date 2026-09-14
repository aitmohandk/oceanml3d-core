"""Toy dynamics + classical DA: the capabilities inherited from 4dvarnet-fm-opencode.

These also prove the framework is not ocean-grid-specific: a Lorenz-96 state is a 1 x K 'grid' and
goes through the same catalog / datamodule / patches / model / export path.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from oceanml3d.dynamics import get_dynamics, list_dynamics  # noqa: E402


def test_registry_and_lorenz63_attractor():
    assert {"lorenz63", "lorenz96"} <= set(list_dynamics())
    traj = get_dynamics("lorenz63").simulate(5000, spinup=1000)
    assert traj.shape == (5000, 3)
    assert 20 < traj[:, 2].mean() < 30 and np.isfinite(traj).all()      # z hovers around rho-ish values


def test_lorenz96_is_chaotic_and_energy_bounded():
    dyn = get_dynamics("lorenz96", state_dim=40)
    a = dyn.simulate(2000, spinup=500, seed=0)
    b = dyn.simulate(2000, x0=a[0] + 1e-6, seed=0)
    assert np.abs(a - b)[-1].max() > 1.0                                # divergence from a tiny perturbation
    assert np.abs(a).max() < 30                                          # but the attractor stays bounded


def test_rk4_matches_a_finer_integration():
    """Same physical time on both sides (simulate stores the state *before* each step)."""
    x0 = np.array([1.0, 1.0, 1.0])
    ref = get_dynamics("lorenz63", dt=1e-4).simulate(5001, x0=x0)[-1]     # t = 0 .. 0.50 everywhere
    coarse = get_dynamics("lorenz63", dt=1e-2).simulate(51, x0=x0)[-1]
    fine = get_dynamics("lorenz63", dt=1e-3).simulate(501, x0=x0)[-1]
    e_coarse = np.abs(coarse - ref).max()
    e_fine = np.abs(fine - ref).max()
    assert e_coarse < 1e-3
    assert e_fine < e_coarse / 100                                        # RK4: 4th-order convergence


def test_gaspari_cohn_shape():
    pytest.importorskip("torch")
    pytest.importorskip("pytorch_lightning")
    from oceanml3d.models.ocean.assimilation.model import gaspari_cohn, periodic_distance

    z = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    g = gaspari_cohn(z)
    assert g[0] == pytest.approx(1.0) and g[-1] == 0 and g[-2] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(g) <= 1e-12)                                   # decreasing
    d = periodic_distance(8)
    assert d[0, 7] == 1 and d[0, 4] == 4                                 # periodic domain


def test_toy_dataset_matches_the_oceanml3d_layout(tmp_path):
    import xarray as xr
    from make_toy_data import make

    path = make(tmp_path, "lorenz96", steps=500, spinup=100, obs_density=0.3)
    ds = xr.open_dataset(path)
    assert ds.state.dims == ("time", "lat", "lon") and ds.sizes["lon"] == 40 and ds.sizes["lat"] == 1
    assert 0.2 < float(np.isfinite(ds.obs).mean()) < 0.4
    assert np.array_equal(np.isfinite(ds.obs.values), ds.obs_mask.values > 0)


def _toy_dm(tmp_path_factory, patch_time: int):
    pytest.importorskip("torch")
    pytest.importorskip("pytorch_lightning")
    from make_toy_data import make

    from oceanml3d.catalog import Catalog
    from oceanml3d.data.datamodule import OceanDataModule
    from oceanml3d.variables import VariableSet

    out = tmp_path_factory.mktemp("toy")
    make(out, "lorenz96", steps=600, spinup=100, obs_density=0.3, obs_noise=0.5)
    variables = VariableSet.from_config({
        "obs": {"source": "l96", "role": "input"},
        "obs_mask": {"source": "l96", "role": "input"},
        "state": {"source": "l96", "role": "target"}})
    catalog = Catalog({"l96": str(out / "lorenz96.nc")})
    days = lambda a, b: [str(np.datetime64("2000-01-01") + np.timedelta64(6 * a, "h")),  # noqa: E731
                         str(np.datetime64("2000-01-01") + np.timedelta64(6 * b, "h"))]
    dm = OceanDataModule(variables, catalog, domain={"lat": [-1, 1], "lon": [0, 40]},
                         splits={"train": days(0, 400), "val": days(401, 500), "test": days(501, 590)},
                         patch={"time": patch_time, "lat": 1, "lon": 40},
                         stride={"time": 5, "lat": 1, "lon": 40},
                         batch_size=2, num_workers=0)
    dm.setup("fit")
    return variables, dm


@pytest.fixture
def toy_dm(tmp_path_factory):
    return _toy_dm(tmp_path_factory, patch_time=10)


def test_enkf_converges_and_beats_the_raw_observations(tmp_path_factory):
    """A filter that knows the dynamics must reconstruct the state better than the sparse obs.

    The window is 80 steps, not the 10 of `toy_dm`, and that is the point being tested. Each patch
    restarts the filter from scratch: `_assimilate` seeds the ensemble from the first frame, which is
    70% NaN at this observation density, and fills the unobserved components with `N(0, 1)` draws
    while the true L96 state has a spread of ~3.6. The analysis therefore starts far outside the
    attractor and has to be pulled back onto it by the observations. Measured mean absolute error at
    observed points, same data and same filter, window length the only difference:

        T=10   obs 0.0966   EnKF 0.1637   per step: 0.07 0.25 0.16 0.18 0.25 0.16 0.17 0.17 0.13 0.14
        T=80   obs 0.0893   EnKF 0.0796   per step: ... 0.14 0.09 ... 0.06 0.05 ... 0.04 0.04 0.04

    So the filter does converge — to 0.04, half the observation error — it just needs more than ten
    steps to get there, and a window-average over the spin-up is dominated by it. Asserting this at
    T=10 asserts something an EnKF cannot do; see `tests/KNOWN_FAILURES.md`.
    """
    import torch

    from oceanml3d.models.ocean.assimilation.model import EnsembleKalmanFilter
    from oceanml3d.training.weights import patch_weight

    t = 80
    variables, dm = _toy_dm(tmp_path_factory, patch_time=t)
    w = patch_weight("constant", {"time": t, "lat": 1, "lon": 40}, {"time": 0, "lat": 0, "lon": 0})
    model = EnsembleKalmanFilter(variables, t, w, dynamics="lorenz96", dynamics_kw={"state_dim": 40},
                                 n_members=12, obs_noise=0.5, localization_radius=4,
                                 optimizer_kw={"lr": 1e-3, "t_max": 1}, norm_stats=dm.norm_stats())
    batch = next(iter(dm.test_dataloader()))
    out = model(batch)
    tgt = model.targets(batch)
    assert out.shape == tgt.shape and torch.isfinite(out).all()
    obs = batch[:, variables.index("obs")]
    seen = torch.isfinite(obs)
    err = lambda a, s: (a[s] - tgt[:, 0][s]).abs().mean()                            # noqa: E731
    assert err(out[:, 0], seen) < err(obs, seen)

    # and it is convergence, not luck: the second half must be clearly better than the spin-up.
    first, last = seen.clone(), seen.clone()
    first[:, t // 2:] = False
    last[:, : t // 2] = False
    assert err(out[:, 0], last) < 0.5 * err(out[:, 0], first)
    assert torch.isfinite(model.step(batch, "test"))


def test_enkf_can_return_an_ensemble(toy_dm):
    from oceanml3d.models.ocean.assimilation.model import EnsembleKalmanFilter
    from oceanml3d.training.weights import patch_weight

    variables, dm = toy_dm
    w = patch_weight("constant", {"time": 10, "lat": 1, "lon": 40}, {"time": 0, "lat": 0, "lon": 0})
    model = EnsembleKalmanFilter(variables, 10, w, dynamics="lorenz96", dynamics_kw={"state_dim": 40},
                                 n_members=8, return_ensemble=True, optimizer_kw={"lr": 1e-3, "t_max": 1},
                                 norm_stats=dm.norm_stats())
    batch = next(iter(dm.test_dataloader()))
    ens = model.predict_step(batch, 0)
    assert ens.shape[0] == 8 and ens.shape[1:] == (2, 1, 10, 1, 40)


def test_optimal_interpolation_fills_gaps(variables, catalog):
    """OI on the ocean task: the reconstruction must be finite everywhere, including unobserved cells."""
    pytest.importorskip("pytorch_lightning")
    torch = pytest.importorskip("torch")

    from oceanml3d.data.datamodule import OceanDataModule
    from oceanml3d.models.ocean.assimilation.model import OptimalInterpolation
    from oceanml3d.training.weights import patch_weight
    from oceanml3d.variables import VariableSet

    vs = VariableSet.from_config({"ssh_obs": {"source": "ssh", "var_name": "u_drifter", "role": "input"},
                                  "u_drifter": {"source": "ssh", "role": "target"}})
    dm = OceanDataModule(vs, catalog, domain={"lat": [-5, 5], "lon": [-5, 5]},
                         splits={"train": ["2019-01-01", "2019-01-20"], "val": ["2019-01-21", "2019-01-30"],
                                 "test": ["2019-01-31", "2019-02-09"]},
                         patch={"time": 5, "lat": 16, "lon": 16}, stride={"time": 1, "lat": 12, "lon": 12},
                         batch_size=2, num_workers=0)
    dm.setup("fit")
    w = patch_weight("constant", {"time": 5, "lat": 16, "lon": 16}, {"time": 0, "lat": 2, "lon": 2})
    model = OptimalInterpolation(vs, 5, w, obs_map={"u_drifter": "ssh_obs"}, length_scale=3.0,
                                 optimizer_kw={"lr": 1e-3, "t_max": 1}, norm_stats=dm.norm_stats())
    batch = next(iter(dm.test_dataloader()))
    out = model(batch)
    assert out.shape == (2, 1, 5, 16, 16) and torch.isfinite(out).all()
    with pytest.raises(ValueError, match="not input variables"):
        OptimalInterpolation(vs, 5, w, obs_map={"u_drifter": "nope"})
