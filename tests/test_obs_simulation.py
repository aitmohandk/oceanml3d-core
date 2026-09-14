"""Ported from NOSC first_implementation tests/test_synthetic_obs.py (+ pseudo-obs on synthetic data)."""
import numpy as np
import pytest

from oceanml3d.obs.missions import MISSIONS, SIX_SAT_NADIR, Mission, validate_missions
from oceanml3d.obs.orbits import ascending_node_lon_deg, ground_track
from oceanml3d.obs.sampling import n_samples_for_grid, nearest_grid_indices, rasterize_day


def test_orbital_periods_plausible():
    for m in MISSIONS.values():
        assert 90 * 60 <= m.orbital_period_s <= 130 * 60


def test_passes_vs_orbits_guard():
    with pytest.raises(ValueError):
        validate_missions([Mission("bad", 66.04, 9.9156, 254)])


def test_repeat_cycle_closes():
    m = MISSIONS["jason3"]
    l0 = ascending_node_lon_deg(m, 0)
    l1 = ascending_node_lon_deg(m, m.orbits_per_cycle)
    assert abs(((l1 - l0) + 180) % 360 - 180) < 1e-6


def test_daily_nadir_coverage_quarter_degree():
    lat = np.arange(-70, 70.01, 0.25)
    lon = np.arange(-180, 180, 0.25)
    cov = rasterize_day([MISSIONS["jason3"]], 0, lat, lon).mean()
    assert 0.015 <= cov <= 0.045


def test_track_continuity():
    lat = np.arange(-70, 70.01, 0.25)
    lon = np.arange(-180, 180, 0.25)
    la, lo = ground_track(MISSIONS["jason3"], 3, n_samples=n_samples_for_grid(lat, lon))
    r, c = nearest_grid_indices(la, lo, lat, lon)
    dr, dc = np.abs(np.diff(r)), np.abs(np.diff(c))
    dc = np.minimum(dc, len(lon) - dc)
    adjacent = (dr <= 1) & (dc <= 1)
    assert adjacent.mean() > 0.99


def test_pseudo_obs_and_mask(synthetic_dir):
    import xarray as xr

    from oceanml3d.obs.pseudo_obs import build_mask_dataset, make_pseudo_obs

    truth = synthetic_dir / "synthetic_surface.nc"
    mask = build_mask_dataset(truth, missions=SIX_SAT_NADIR[:2])
    assert mask.obs_mask.dims == ("time", "lat", "lon") and 0 < float(mask.obs_mask.mean()) < 0.5
    out = make_pseudo_obs(truth, "zos", mask, synthetic_dir / "pobs.nc", noise_std=0.02, seed=1)
    ds = xr.open_dataset(out)
    obs, m = ds.zos_obs.values, ds.obs_mask.values
    assert np.isnan(obs[m == 0]).all() and np.isfinite(obs[m == 1]).all()
    truth_v = xr.open_dataset(truth).zos.values
    err = obs[m == 1] - truth_v[m == 1]
    assert 0.015 < err.std() < 0.025
    out2 = make_pseudo_obs(truth, "zos", mask, synthetic_dir / "pobs2.nc", noise_std=0.02, seed=1)
    assert np.array_equal(xr.open_dataset(out2).zos_obs.values, obs, equal_nan=True)   # deterministic
