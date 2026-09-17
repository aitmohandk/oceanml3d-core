import numpy as np

from oceanml3d.data.augment import MissionDropout, ObsNoise, build_augmentations
from oceanml3d.data.patches import PatchArray, PatchSpec
from oceanml3d.variables import VariableSet


def _vs():
    return VariableSet.from_config({
        "ssh_obs": {"source": "p", "role": "input"}, "obs_mask": {"source": "p", "role": "input"},
        "mask_a": {"source": "p", "role": "aux"}, "mask_b": {"source": "p", "role": "aux"},
        "zos": {"source": "t", "role": "target"}})


def test_mission_dropout_removes_only_uncovered_cells():
    vs = _vs()
    aug = build_augmentations({"mission_dropout": {"obs": ["ssh_obs"], "mask": "obs_mask", "missions": ["mask_a", "mask_b"], "n_drop": [1, 1], "p": 1.0}}, vs)[0]
    assert isinstance(aug, MissionDropout)
    item = np.zeros((5, 1, 4, 4), np.float32)
    item[2, 0, 0, :] = 1          # mission a observes row 0
    item[3, 0, 1, :] = 1          # mission b observes row 1
    item[3, 0, 0, 0] = 1          # cell (0,0) observed by both
    item[1] = (item[2] + item[3] > 0)
    item[0] = np.where(item[1] > 0, 1.0, np.nan)
    out = aug(item.copy(), np.random.default_rng(0))
    assert np.isfinite(out[0]).sum() == 4 + 1 or np.isfinite(out[0]).sum() == 4   # one mission lost, shared cell kept
    assert np.isfinite(out[0, 0, 0, 0])
    assert (out[1] > 0).sum() == np.isfinite(out[0]).sum()
    assert vs["mask_a"].role.value == "aux" and not vs["mask_a"].is_input


def test_obs_noise_only_on_observed():
    vs = _vs()
    aug = ObsNoise({"ssh_obs": 0.1})
    aug.bind(vs)
    item = np.full((5, 1, 4, 4), np.nan, np.float32)
    item[0, 0, 0, :] = 0.0
    out = aug(item.copy(), np.random.default_rng(1))
    assert np.isfinite(out[0]).sum() == 4 and 0 < np.abs(out[0, 0, 0]).mean() < 0.4


def test_jitter_stays_in_domain(variables, catalog):
    import xarray as xr

    from oceanml3d.data.open import open_variable_set

    da = open_variable_set(variables, catalog, {"lat": slice(-5, 5), "lon": slice(-5, 5), "time": slice("2019-01-01", "2019-01-12")})
    spec = PatchSpec({"time": 5, "lat": 16, "lon": 16}, {"time": 2, "lat": 8, "lon": 8})
    pa = PatchArray(da, spec, jitter=True, seed=0)
    shapes = {pa[i].shape for i in range(len(pa))}
    assert shapes == {(len(variables), 5, 16, 16)}
    assert isinstance(da, xr.DataArray)


def test_jitter_is_centred_not_forward_only():
    """The offset used to be drawn from [0, stride]: forward only, so the leading edge of the domain
    was under-sampled and the trailing edge over-sampled, and an offset of exactly `stride`
    reproduced the next patch."""
    import numpy as np
    import xarray as xr

    from oceanml3d.data.patches import PatchArray, PatchSpec

    n = 60
    da = xr.DataArray(np.zeros((1, n, 8, 8), dtype="float32"), dims=("channel", "time", "lat", "lon"),
                      coords={"channel": ["a"], "time": np.arange(n),
                              "lat": np.arange(8.0), "lon": np.arange(8.0)})
    spec = PatchSpec({"time": 10, "lat": 8, "lon": 8}, {"time": 10, "lat": 8, "lon": 8})
    pa = PatchArray(da, spec, jitter=True, seed=0)

    middle = pa.slices(2)                       # a window with room on both sides
    offsets = {pa._jittered(middle)["time"].start - middle["time"].start for _ in range(400)}
    assert min(offsets) < 0, f"never shifted backwards: {sorted(offsets)}"
    assert max(offsets) > 0, f"never shifted forwards: {sorted(offsets)}"
    assert max(abs(o) for o in offsets) <= spec.stride["time"] // 2

    first, last = pa.slices(0), pa.slices(len(pa.index) - 1)
    for sl in (first, last):
        for _ in range(200):
            j = pa._jittered(sl)
            assert 0 <= j["time"].start and j["time"].stop <= n, "jitter left the domain"
