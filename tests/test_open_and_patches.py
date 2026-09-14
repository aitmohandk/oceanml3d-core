import numpy as np

from oceanml3d.data.open import compute_norm_stats, open_variable_set
from oceanml3d.data.patches import PatchArray, PatchIndex, PatchSpec
from oceanml3d.training.weights import patch_weight


def test_open_variable_set_shape(variables, catalog):
    da = open_variable_set(variables, catalog, {"lat": slice(-5, 5), "lon": slice(-5, 5)})
    assert da.dims == ("channel", "time", "lat", "lon")
    assert list(da.channel.values) == variables.names
    assert da.sizes["time"] == 40
    lat_field = da.sel(channel="lat").isel(time=0).values
    assert np.allclose(lat_field[:, 0], da.lat.values)          # static field broadcast
    assert np.isnan(da.sel(channel="u_drifter").values).mean() > 0.5  # sparse targets kept as NaN


def test_patch_index_covers_domain():
    spec = PatchSpec({"time": 5, "lat": 32, "lon": 32}, {"time": 1, "lat": 24, "lon": 24})
    idx = PatchIndex({"time": 10, "lat": 48, "lon": 70}, spec)
    last = idx.slices(len(idx) - 1)
    assert last["lat"].stop == 48 and last["lon"].stop == 70 and last["time"].stop == 10


def test_reconstruct_roundtrip(variables, catalog):
    da = open_variable_set(variables, catalog, {"lat": slice(-5, 5), "lon": slice(-5, 5)}).sel(time=slice("2019-01-01", "2019-01-12"))
    spec = PatchSpec({"time": 5, "lat": 16, "lon": 16}, {"time": 1, "lat": 12, "lon": 12})
    stats = compute_norm_stats(da, slice("2019-01-01", "2019-01-12"))
    pa = PatchArray(da, spec, norm_stats=stats)
    items = np.stack([pa[i] for i in range(len(pa))])
    assert items.shape[1:] == (len(variables), 5, 16, 16)
    w = patch_weight("triangular", spec.patch, {"time": 0, "lat": 2, "lon": 2})
    rec = pa.reconstruct(items[:, variables.target_indices], w, ["u", "v"])
    mean, std = stats
    ref = ((da.isel(channel=variables.target_indices) - mean[variables.target_indices, None, None, None])
           / std[variables.target_indices, None, None, None]).values
    both = np.isfinite(rec.values) & np.isfinite(ref)
    assert both.any()
    assert np.allclose(rec.values[both], ref[both], atol=1e-5)
