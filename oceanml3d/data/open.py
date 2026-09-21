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


def _align_space(da: xr.DataArray, reference: xr.DataArray, name: str) -> xr.DataArray:
    """Nearest-neighbour reindex onto the reference grid, refusing a grid that is not the same one.

    ``method="nearest"`` without a tolerance never fails: point a variable at a 1/4 deg file while
    the reference is 1/12 deg, or at a grid offset by half a cell, and every target cell quietly
    takes the closest source cell. The run then converges on resampled data. Requiring every target
    cell to sit within half a grid step of its source keeps the intended use -- aligning grids that
    are nominally identical but differ in float representation -- and rejects the rest.
    """
    for dim in ("lat", "lon"):
        if dim not in da.dims:
            continue
        src = np.asarray(da[dim].values, dtype=float)
        dst = np.asarray(reference[dim].values, dtype=float)
        if src.size == 0:
            raise ValueError(f"{name}: empty '{dim}' axis after domain selection")
        step = float(np.median(np.abs(np.diff(dst)))) if dst.size > 1 else float("inf")
        nearest = np.abs(dst[:, None] - src[None, :]).argmin(axis=1)
        worst = float(np.abs(dst - src[nearest]).max())
        if step != float("inf") and worst > 0.5 * step:
            raise ValueError(
                f"{name}: '{dim}' is not on the reference grid -- the worst target cell is {worst:.4g} "
                f"away from its nearest source cell, more than half a grid step ({0.5 * step:.4g}). "
                f"Regrid the file (see scripts/prepare/regrid.py) rather than letting nearest-neighbour "
                f"resampling hide it."
            )
    return da.reindex(lat=reference.lat, lon=reference.lon, method="nearest")


def _missing_steps(da: xr.DataArray, reference: xr.DataArray) -> int:
    return int(np.count_nonzero(~np.isin(reference.time.values, da.time.values)))


def _describe_time(da: xr.DataArray) -> str:
    t = da.time.values
    if t.size == 0:
        return "no time step at all"
    return f"{t.size} steps, {str(t[0])[:19]} .. {str(t[-1])[:19]} ({t.dtype})"


def _align_time(da: xr.DataArray, reference: xr.DataArray, name: str) -> xr.DataArray:
    """Reindex onto the reference time axis. Missing dates become NaN.

    For an input, NaN becomes 0 in ``BaseOceanModel.inputs`` -- so a hole in a forcing file trains
    the model on zeros. How many steps were invented is reported by :func:`open_variable_set`, once
    per file, and a variable with *none* of the reference's steps is refused there outright.
    """
    return da.reindex(time=reference.time)


def _report_time_gaps(gaps: dict, reference: xr.DataArray, first: str) -> None:
    """One line per file rather than one per channel, and a hard stop when a file shares no date.

    ``osse3d_gs21`` draws 42 channels from the virtual ARGO file. A file built against another truth
    printed 42 identical lines saying 100% of the steps were absent -- and then training went ahead
    on an ARGO input that was zero everywhere, which no metric would ever have flagged: the model
    simply learns to ignore a channel that never carries information. A file whose time axis has no
    date in common with the reference is not a gap, it is the wrong file.
    """
    total = reference.sizes["time"]
    empty = []
    for path, (names, missing, axis) in gaps.items():
        worst = max(missing)
        head = ", ".join(names[:3]) + (f" and {len(names) - 3} more" if len(names) > 3 else "")
        print(f"[oceanml3d] {path}: {worst}/{total} time steps ({100 * worst / total:.1f}%) absent "
              f"for {len(names)} channel(s) ({head}); they will be NaN (0 for inputs)")
        if worst == total:
            empty.append(f"  {path}\n    file:      {axis}\n    reference: {_describe_time(reference)} "
                         f"(from '{first}')")
    if empty:
        raise ValueError(
            "these files share no date with the rest of the task, so every channel read from them "
            "would be empty for the whole run:\n" + "\n".join(empty) + "\n"
            "A derived file (a `prepare-obs` output: pseudo-obs, virtual ARGO) built against another "
            "truth, period or resolution is the usual cause -- delete it and it is rebuilt on the "
            "next run.")


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
        idx = spec.depth_index if spec.depth_index is not None else 0
        if idx >= da.sizes["depth"]:
            raise IndexError(f"{spec.name}: depth_index {idx} out of range (file has {da.sizes['depth']} levels)")
        da = da.isel(depth=idx, drop=True)
    elif spec.depth_index is not None and var_name == spec.var_name:
        # `is not None`, not truthiness: index 0 is falsy, so this guard never fired for the surface
        # level. `thetao_d00`, `uo_d00` and `vo_d00` in osse3d_gs21 are exactly that case -- pointed
        # at a file with no depth axis they silently returned the 2D field instead of raising.
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
    first = ""
    cache: dict[str, xr.Dataset] = {}          # one handle per file for the whole set
    gaps: dict[str, tuple[list[str], list[int], str]] = {}
    for spec in variables:
        da = open_variable(spec, catalog, domain, chunks, cache)
        if spec.is_static:
            if reference is None:
                raise ValueError("first variable cannot be static (needs a time axis to broadcast)")
            da = _align_space(da, reference, spec.name)
            da = da.expand_dims(time=reference.time).broadcast_like(reference)
        else:
            if reference is None:
                reference, first = da, spec.name
            else:
                da = _align_space(da, reference, spec.name)
                missing = _missing_steps(da, reference)
                if missing:
                    entry = gaps.setdefault(str(catalog.resolve(spec.source)), ([], [], _describe_time(da)))
                    entry[0].append(spec.name)
                    entry[1].append(missing)
                da = _align_time(da, reference, spec.name)
        arrays[spec.name] = da.transpose("time", "lat", "lon")
    if gaps:
        _report_time_gaps(gaps, reference, first)
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
    # A channel with no finite value in the train window has no mean either: NaN, which would then
    # normalise every value of that channel to NaN. It is reported rather than papered over -- it is
    # the same failure as a file with no date in common, caught one step later.
    blank = ~np.isfinite(mean)
    if blank.any():
        names = [str(c) for c in np.asarray(da.channel.values)[blank]] if "channel" in da.coords else []
        print(f"[oceanml3d] WARNING {int(blank.sum())} channel(s) have no finite value in the train "
              f"window {time_slice.start}..{time_slice.stop}: {', '.join(names[:5])}"
              f"{' ...' if len(names) > 5 else ''} -- normalised with mean 0, std 1")
    mean = np.where(blank, 0.0, mean)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)
