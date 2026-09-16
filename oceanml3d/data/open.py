"""Build one ``(channel, time, lat, lon)`` DataArray from many files.

Cleaned-up version of NOSC ``contrib/data_loading/data.py::open_multivar_datasets``:
- paths come from the catalog, not from the YAML;
- static fields (lat, bathymetry) are broadcast over time;
- dimension names are normalised to ``time/lat/lon``;
- everything stays lazy (dask) until patches are extracted.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr

from oceanml3d.catalog import Catalog
from oceanml3d.data.transforms import get_transform
from oceanml3d.variables import VariableSet, VariableSpec

_DIM_ALIASES = {"latitude": "lat", "longitude": "lon", "nav_lat": "lat", "nav_lon": "lon"}


def normalise_dims(ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    rename = {k: v for k, v in _DIM_ALIASES.items() if k in ds.dims or k in ds.coords}
    if rename:
        ds = ds.rename(rename)
    if "depth" in ds.dims and ds.sizes["depth"] == 1:
        ds = ds.squeeze("depth", drop=True)
    return ds


def _select_domain(da: xr.DataArray, domain: Mapping[str, slice]) -> xr.DataArray:
    sel = {k: v for k, v in domain.items() if k in da.dims}
    return da.sel(sel) if sel else da


def _open(path, chunks: Mapping[str, int] | None,
          cache: dict[str, xr.Dataset] | None = None) -> xr.Dataset:
    """Open one file, dimension-normalised, reusing an already-open handle when there is one.

    A task names variables, not files, and many variables share a file: ``osse3d_gs21`` draws 64
    targets from ``glorys_gs_multidepth`` and 42 inputs and masks from ``argo_virtual_thetao_gs21``,
    so opening per variable meant ~110 ``open_dataset`` calls over three files, and 110 separately
    reindexed arrays in the dask graph that every ``__getitem__`` then had to walk.
    """
    key = str(path)
    if cache is not None and key in cache:
        return cache[key]
    open_kw: dict[str, Any] = {"chunks": dict(chunks) if chunks else "auto"}
    ds = normalise_dims(xr.open_zarr(path, **open_kw) if key.endswith(".zarr")
                        else xr.open_dataset(path, **open_kw))
    if cache is not None:
        cache[key] = ds
    return ds


def open_variable(spec: VariableSpec, catalog: Catalog, domain: Mapping[str, slice],
                  chunks: Mapping[str, int] | None = None,
                  cache: dict[str, xr.Dataset] | None = None) -> xr.DataArray:
    path = catalog.resolve(spec.source)
    ds = _open(path, chunks, cache)
    var_name = spec.var_name
    if var_name not in ds and spec.depth_index is not None:
        base, suffix = (var_name[:-5], "_mask") if var_name.endswith("_mask") else (var_name, "")
        cand = f"{base}_d{spec.depth_index:02d}{suffix}"       # per-level 2D layout (e.g. virtual ARGO files)
        if cand in ds:
            var_name = cand
    if var_name not in ds:
        raise KeyError(f"'{var_name}' not found in {path}; variables: {list(ds.data_vars)}")
    da = ds[var_name]
    if "depth" in da.dims:
        idx = spec.depth_index or 0
        if idx >= da.sizes["depth"]:
            raise IndexError(f"{spec.name}: depth_index {idx} out of range (file has {da.sizes['depth']} levels)")
        da = da.isel(depth=idx, drop=True)
    elif spec.depth_index and var_name == spec.var_name:
        raise ValueError(f"{spec.name}: depth_index given but {path} has no depth axis nor '{var_name}_d{spec.depth_index:02d}'")
    da = _select_domain(da, domain)
    if spec.transform:
        da = get_transform(spec.transform)(da)
    if spec.fill_nan is not None:
        da = da.fillna(spec.fill_nan)
    if spec.mask:
        # Chunked like everything else. `open_dataarray` without `chunks` loads eagerly, which for a
        # daily 1/12 deg mask over ten years is a few hundred MB pulled into memory per masked
        # variable at setup -- and flatly contradicts this module's "everything stays lazy" contract.
        mask_path = catalog.resolve(spec.mask)
        mask_ds = _open(mask_path, chunks, cache)
        names = list(mask_ds.data_vars)
        if len(names) != 1:
            raise KeyError(f"{spec.name}: mask file {mask_path} must hold exactly one variable, found {names}")
        da = da.where(_select_domain(mask_ds[names[0]], domain) > 0)
    return da.rename(spec.name)


def open_variable_set(variables: VariableSet, catalog: Catalog, domain: Mapping[str, slice],
                      chunks: Mapping[str, int] | None = None) -> xr.DataArray:
    """Return a lazy DataArray with dims ``(channel, time, lat, lon)``."""
    arrays: dict[str, xr.DataArray] = {}
    reference: xr.DataArray | None = None
    cache: dict[str, xr.Dataset] = {}          # one handle per file for the whole set
    for spec in variables:
        da = open_variable(spec, catalog, domain, chunks, cache)
        if spec.is_static:
            if reference is None:
                raise ValueError("first variable cannot be static (needs a time axis to broadcast)")
            da = da.reindex(lat=reference.lat, lon=reference.lon, method="nearest")
            da = da.expand_dims(time=reference.time).broadcast_like(reference)
        else:
            if reference is None:
                reference = da
            else:
                da = da.reindex(lat=reference.lat, lon=reference.lon, method="nearest")
                da = da.reindex(time=reference.time)
        arrays[spec.name] = da.transpose("time", "lat", "lon")
    stacked = xr.concat(list(arrays.values()), dim="channel", coords="minimal", compat="override")
    stacked = stacked.assign_coords(channel=list(arrays))
    return stacked.astype(np.float32)


def compute_norm_stats(da: xr.DataArray, time_slice: slice) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and standard deviation over the train window.

    The two reductions are computed in a single ``compute()`` so dask loads each chunk once and
    feeds both from it; separate calls walked the train split twice. dask's ``std`` is moment-based
    and does not consume the mean, so this is a scheduling change only -- the values are unchanged.
    """
    sub = da.sel(time=time_slice)
    dims = ("time", "lat", "lon")
    stats = xr.Dataset({"mean": sub.mean(dim=dims, skipna=True),
                        "std": sub.std(dim=dims, skipna=True)}).compute()
    mean, std = stats["mean"].values, stats["std"].values
    std = np.where(std > 0, std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)
