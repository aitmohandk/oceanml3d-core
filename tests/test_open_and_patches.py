import numpy as np
import pytest

from oceanml3d.data.open import compute_norm_stats, open_variable_set
from oceanml3d.data.patches import PatchAccumulator, PatchArray, PatchIndex, PatchSpec
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


def _holed(n_time=14, n_lat=20, n_lon=24, seed=0):
    """A (channel, time, lat, lon) array with two targets punched full of holes: whole empty days,
    and scattered finite cells elsewhere. This is the shape the drop-empty filter exists for."""
    import xarray as xr

    rng = np.random.default_rng(seed)
    inputs = rng.normal(size=(3, n_time, n_lat, n_lon))
    tgt = rng.normal(size=(2, n_time, n_lat, n_lon))
    tgt[rng.random(tgt.shape) < 0.97] = np.nan            # sparse, like real drifter maps
    # Days 0-8 wholly unobserved. The block has to be at least as long as the longest patch below,
    # or every window straddles an observed day and the filter has nothing to drop -- which is what
    # a first version of this fixture did, scattering five empty days instead of grouping them.
    tgt[:, :9] = np.nan
    tgt[0, 10] = np.nan                                   # one target empty on a day the other is not
    arr = np.concatenate([inputs, tgt], axis=0)
    return xr.DataArray(
        arr, dims=("channel", "time", "lat", "lon"),
        coords={"channel": ["a", "b", "c", "u", "v"],
                "time": np.arange(n_time), "lat": np.arange(n_lat), "lon": np.arange(n_lon)})


def test_valid_patches_matches_the_reference():
    """`_valid_patches` reduces once per spatial window instead of once per patch. It must select
    exactly the same patches as the per-patch oracle it replaced -- including when the spatial grid
    holds several windows, and when those windows overlap."""
    da = _holed()
    target_idx = [3, 4]
    cases = [({"time": 5, "lat": 20, "lon": 24}, {"time": 1, "lat": 1, "lon": 1}),   # one spatial window
             ({"time": 3, "lat": 8, "lon": 10}, {"time": 2, "lat": 6, "lon": 7}),    # several, overlapping
             ({"time": 1, "lat": 12, "lon": 12}, {"time": 12, "lat": 12, "lon": 12})]
    for patch, stride in cases:
        spec = PatchSpec(patch, stride)
        pa = PatchArray(da, spec)                                   # unfiltered: holds every window
        reference = [i for i in range(len(pa.index)) if pa._has_target(i, target_idx)]
        assert pa._valid_patches(target_idx) == reference, f"mismatch for patch={patch} stride={stride}"
        assert reference != list(range(len(pa.index))), "the fixture must actually drop something"


def test_construction_does_not_use_the_per_patch_oracle(monkeypatch):
    """The whole point of the rewrite is that cost stops scaling with the number of patches, so the
    per-patch path must not be reachable from the constructor."""
    def boom(*a, **k):
        raise AssertionError("_has_target costs one materialisation per patch; it must not run at construction")

    monkeypatch.setattr(PatchArray, "_has_target", boom)
    da = _holed()
    spec = PatchSpec({"time": 5, "lat": 20, "lon": 24}, {"time": 1, "lat": 1, "lon": 1})
    pa = PatchArray(da, spec, drop_all_nan_targets=[3, 4])
    assert 0 < len(pa) < len(pa.index)
    for i in range(len(pa)):
        assert np.isfinite(pa[i][[3, 4]]).any(), "a kept patch must hold a finite target"


def test_accumulator_matches_reconstruct(variables, catalog):
    """Streaming must be arithmetically identical to the all-at-once path, batch boundaries and all."""
    from oceanml3d.data.open import open_variable_set

    da = open_variable_set(variables, catalog, {"lat": slice(-5, 5), "lon": slice(-5, 5)}).sel(
        time=slice("2019-01-01", "2019-01-12"))
    spec = PatchSpec({"time": 5, "lat": 16, "lon": 16}, {"time": 1, "lat": 12, "lon": 12})
    pa = PatchArray(da, spec)
    items = np.stack([pa[i] for i in range(len(pa))])[:, variables.target_indices]
    w = patch_weight("triangular", spec.patch, {"time": 0, "lat": 2, "lon": 2})

    whole = pa.reconstruct(items, w, ["u", "v"])
    acc = pa.accumulator(items.shape[1], w)
    for start in range(0, len(items), 3):                      # fold in batches of three
        acc.add_batch(start, items[start:start + 3])
    streamed = acc.result(["u", "v"])

    assert streamed.dims == whole.dims and list(streamed.channel.values) == ["u", "v"]
    both = np.isfinite(whole.values) & np.isfinite(streamed.values)
    assert both.any()
    assert np.array_equal(np.isfinite(whole.values), np.isfinite(streamed.values))
    assert np.allclose(whole.values[both], streamed.values[both], rtol=0, atol=0)


def test_accumulator_refuses_a_partial_field(variables, catalog):
    """The DDP failure mode, made reachable without DDP: `trainer.predict` shards the loader, so a
    rank sees only part of the patches. Stitching that subset used to produce a plausible, wrong
    field -- element `i` of whatever arrived was written to patch `i` of the domain."""
    from oceanml3d.data.open import open_variable_set

    da = open_variable_set(variables, catalog, {"lat": slice(-5, 5), "lon": slice(-5, 5)}).sel(
        time=slice("2019-01-01", "2019-01-12"))
    spec = PatchSpec({"time": 5, "lat": 16, "lon": 16}, {"time": 1, "lat": 12, "lon": 12})
    pa = PatchArray(da, spec)
    items = np.stack([pa[i] for i in range(len(pa))])[:, variables.target_indices]

    acc = pa.accumulator(items.shape[1])
    acc.add_batch(0, items[: len(items) // 2])                 # only "rank 0"'s share
    assert acc.missing
    with pytest.raises(RuntimeError, match="never folded in"):
        acc.result(["u", "v"])


def test_accumulator_rejects_a_repeated_or_out_of_range_patch(variables, catalog):
    from oceanml3d.data.open import open_variable_set

    da = open_variable_set(variables, catalog, {"lat": slice(-5, 5), "lon": slice(-5, 5)}).sel(
        time=slice("2019-01-01", "2019-01-06"))
    spec = PatchSpec({"time": 3, "lat": 16, "lon": 16}, {"time": 1, "lat": 12, "lon": 12})
    pa = PatchArray(da, spec)
    item = pa[0][variables.target_indices]
    acc = pa.accumulator(item.shape[0])
    acc.add(0, item)
    with pytest.raises(ValueError, match="twice"):
        acc.add(0, item)
    with pytest.raises(IndexError):
        acc.add(len(pa), item)
    assert isinstance(acc, PatchAccumulator)


def test_one_file_handle_per_source(variables, catalog, monkeypatch):
    """Many variables share a file; the graph should hold one handle per file, not per variable.

    `osse3d_gs21` draws 64 targets from one GLORYS file and 42 inputs and masks from another, so
    opening per variable meant ~110 `open_dataset` calls over three files -- and 110 separately
    reindexed arrays for every `__getitem__` to walk.
    """
    import xarray as xr

    from oceanml3d.data import open as open_mod

    opened: list[str] = []
    original = xr.open_dataset

    def counting_open_dataset(path, *a, **kw):
        opened.append(str(path))
        return original(path, *a, **kw)

    monkeypatch.setattr(open_mod.xr, "open_dataset", counting_open_dataset)
    da = open_variable_set(variables, catalog, {"lat": slice(-5, 5), "lon": slice(-5, 5)})
    assert da.sizes["channel"] == len(variables)
    sources = {s.source for s in variables} | {s.mask for s in variables if s.mask}
    assert len(opened) == len(set(opened)), f"a file was opened more than once: {opened}"
    assert len(opened) <= len(sources)


def test_norm_stats_are_unchanged_by_the_single_pass(variables, catalog):
    """`compute_norm_stats` now schedules both reductions in one `compute()`. dask's `std` is
    moment-based and does not consume the mean, so this must be a scheduling change only."""
    da = open_variable_set(variables, catalog, {"lat": slice(-5, 5), "lon": slice(-5, 5)})
    window = slice("2019-01-01", "2019-01-12")
    mean, std = compute_norm_stats(da, window)

    sub = da.sel(time=window)
    ref_mean = sub.mean(dim=("time", "lat", "lon"), skipna=True).compute().values
    ref_std = sub.std(dim=("time", "lat", "lon"), skipna=True).compute().values
    ref_std = np.where(ref_std > 0, ref_std, 1.0)

    assert np.array_equal(mean, ref_mean.astype(np.float32))
    assert np.array_equal(std, ref_std.astype(np.float32))
    assert (std > 0).all()
