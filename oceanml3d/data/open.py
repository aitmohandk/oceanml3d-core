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


def open_variable(spec: VariableSpec, catalog: Catalog, domain: Mapping[str, slice],
                  chunks: Mapping[str, int] | None = None) -> xr.DataArray:
    path = catalog.resolve(spec.source)
    open_kw: dict[str, Any] = {"chunks": dict(chunks) if chunks else "auto"}
    if str(path).endswith(".zarr"):
        ds = xr.open_zarr(path, **open_kw)
    else:
        ds = xr.open_dataset(path, **open_kw)
    ds = normalise_dims(ds)
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
        mask = normalise_dims(xr.open_dataarray(catalog.resolve(spec.mask)))
        da = da.where(_select_domain(mask, domain) > 0)
    return da.rename(spec.name)


def open_variable_set(variables: VariableSet, catalog: Catalog, domain: Mapping[str, slice],
                      chunks: Mapping[str, int] | None = None) -> xr.DataArray:
    """Return a lazy DataArray with dims ``(channel, time, lat, lon)``."""
    arrays: dict[str, xr.DataArray] = {}
    reference: xr.DataArray | None = None
    for spec in variables:
        da = open_variable(spec, catalog, domain, chunks)
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
    sub = da.sel(time=time_slice)
    mean = sub.mean(dim=("time", "lat", "lon"), skipna=True).compute().values
    std = sub.std(dim=("time", "lat", "lon"), skipna=True).compute().values
    std = np.where(std > 0, std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)
