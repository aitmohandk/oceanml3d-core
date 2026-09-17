"""Daily observation masks and pseudo-observations for an OSSE.

Two steps, both written once to NetCDF (deterministic, identical in train/val/test):

1. :func:`build_mask_dataset` — simulated constellation ground tracks rasterised on the
   truth grid, one ``obs_mask`` (1/0) per day of the truth time axis; or a *real* dated
   L3 track mask (NetCDF) regridded onto the truth grid.
2. :func:`make_pseudo_obs` — truth sampled where ``obs_mask == 1`` plus Gaussian
   instrument noise, saved as ``<var>_obs`` next to ``obs_mask``.

Both outputs are then plain catalog entries used as ``role: input`` variables.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from oceanml3d.data.open import normalise_dims
from oceanml3d.io import write_netcdf_atomic
from oceanml3d.obs.missions import HISTORICAL, MISSIONS, SIX_SAT_NADIR, validate_missions
from oceanml3d.obs.sampling import rasterize_day


def _grid_and_time(reference: str | Path):
    ds = normalise_dims(xr.open_dataset(reference))
    lat = np.asarray(ds.lat.values, dtype=np.float64)
    lon = np.asarray(ds.lon.values, dtype=np.float64)
    time = pd.DatetimeIndex(ds.time.values).normalize()
    return lat, lon, time


def build_mask_dataset(reference: str | Path, missions: list[str] | None = None,
                       real_mask: str | Path | None = None, mask_var: str | None = None,
                       historical: bool = False, per_mission: bool = False) -> xr.Dataset:
    """``obs_mask(time, lat, lon)`` on the grid and time axis of ``reference``.

    ``historical=True`` uses each mission's activity window (non-stationary constellation);
    ``per_mission=True`` also stores ``mask_<mission>`` variables (needed by mission-dropout
    augmentation, see :class:`oceanml3d.data.augment.MissionDropout`).
    """
    lat, lon, time = _grid_and_time(reference)
    if real_mask is not None:
        mds = normalise_dims(xr.open_dataset(real_mask))
        cand = [v for v in ([mask_var] if mask_var else []) + ["l3_mask", "obs_mask"] if v in mds]
        raw = mds[cand[0] if cand else list(mds.data_vars)[0]]
        raw = raw.sel(time=time, method="nearest", tolerance=np.timedelta64(1, "D"))
        raw = raw.interp(lat=lat, lon=lon, method="nearest")
        mask = (np.isfinite(raw.values) & (raw.values != 0)).astype(np.float32)
        source = f"real:{real_mask}"
    else:
        names = missions or (HISTORICAL if historical else SIX_SAT_NADIR)
        ms = [MISSIONS[n] for n in names]
        validate_missions(ms)
        per = {}
        for m in ms:
            days = [rasterize_day([m], d, lat, lon) if (not historical or m.is_active(time[d])) else np.zeros((len(lat), len(lon)), bool)
                    for d in range(len(time))]
            per[m.name] = np.stack(days)
        mask = np.any(np.stack(list(per.values())), axis=0).astype(np.float32)
        source = ("historical:" if historical else "synthetic:") + ",".join(names)
    ds = xr.Dataset({"obs_mask": (("time", "lat", "lon"), mask)},
                    coords={"time": time, "lat": lat, "lon": lon},
                    attrs={"source": source, "mean_daily_coverage": float(mask.mean())})
    if real_mask is None and per_mission:
        for name, arr in per.items():
            ds[f"mask_{name}"] = (("time", "lat", "lon"), arr.astype(np.float32))
    return ds


def build_cloud_mask_dataset(reference: str | Path, coverage: float = 0.6, scale_cells: float = 8.0,
                             seed: int = 0) -> xr.Dataset:
    """Synthetic cloud-cover mask (``obs_mask`` = 1 where clear) for SST pseudo-observations:
    Gaussian-smoothed noise thresholded to reach the requested mean clear-sky ``coverage``.
    Replace by a real L3 cloud mask through ``real_mask`` when available."""
    from scipy.ndimage import gaussian_filter

    lat, lon, time = _grid_and_time(reference)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((len(time), len(lat), len(lon)))
    smooth = gaussian_filter(noise, sigma=(0.7, scale_cells, scale_cells))
    thr = np.quantile(smooth, 1 - coverage)
    mask = (smooth >= thr).astype(np.float32)
    return xr.Dataset({"obs_mask": (("time", "lat", "lon"), mask)}, coords={"time": time, "lat": lat, "lon": lon},
                      attrs={"source": f"synthetic_clouds:coverage={coverage},scale={scale_cells}",
                             "mean_daily_coverage": float(mask.mean())})


def make_pseudo_obs(truth: str | Path, truth_var: str, mask: xr.Dataset | str | Path, output: str | Path,
                    noise_std: float = 0.0, seed: int = 1234, depth_index: int | None = None,
                    skip_if_exists: bool = True) -> Path:
    output = Path(output)
    if skip_if_exists and output.exists():
        return output
    da = normalise_dims(xr.open_dataset(truth))[truth_var]
    if "depth" in da.dims:
        da = da.isel(depth=depth_index or 0, drop=True)
    mds = mask if isinstance(mask, xr.Dataset) else xr.open_dataset(mask)
    m = mds.obs_mask.sel(time=da.time, method="nearest", tolerance=np.timedelta64(1, "D"))
    if (m.sizes["lat"], m.sizes["lon"]) != (da.sizes["lat"], da.sizes["lon"]):
        m = m.interp(lat=da.lat, lon=da.lon, method="nearest")
    observed = m.values > 0
    values = da.values.astype(np.float32)
    if noise_std > 0:
        values = values + np.random.default_rng(seed).normal(0.0, noise_std, values.shape).astype(np.float32)
    obs = np.where(observed, values, np.nan).astype(np.float32)
    out = xr.Dataset({f"{truth_var}_obs": (("time", "lat", "lon"), obs),
                      "obs_mask": (("time", "lat", "lon"), observed.astype(np.float32))},
                     coords={c: da.coords[c] for c in ("time", "lat", "lon")},
                     attrs={"source_truth": str(truth), "source_var": truth_var, "noise_std": float(noise_std),
                            "noise_seed": int(seed), "mask_source": mds.attrs.get("source", "")})
    write_netcdf_atomic(out, output)
    return output


def prepare_pseudo_obs(truth: str | Path, truth_var: str, output: str | Path, mask_output: str | Path | None = None,
                       missions: list[str] | None = None, real_mask: str | Path | None = None,
                       noise_std: float = 0.02, seed: int = 1234, skip_if_exists: bool = True,
                       historical: bool = False, per_mission: bool = False, depth_index: int | None = None,
                       clouds: dict | None = None) -> Path:
    """One-call recipe used by ``oceanml3d prepare-obs`` and by experiment `prepare` hooks."""
    output = Path(output)
    if skip_if_exists and output.exists():
        return output
    mask_output = Path(mask_output) if mask_output else output.with_name(output.stem + "_mask.nc")
    if mask_output.exists() and skip_if_exists:
        mask = xr.open_dataset(mask_output)
    else:
        if clouds is not None:
            mask = build_cloud_mask_dataset(truth, **clouds) if real_mask is None else build_mask_dataset(truth, real_mask=real_mask)
        else:
            mask = build_mask_dataset(truth, missions, real_mask, historical=historical, per_mission=per_mission)
        write_netcdf_atomic(mask, mask_output)
    out = make_pseudo_obs(truth, truth_var, mask, output, noise_std, seed, depth_index, skip_if_exists=False)
    if per_mission and real_mask is None:                    # copy per-mission masks next to the obs
        with xr.open_dataset(out) as src:
            ds = src.load()                                  # closed before we replace the file
        for v in mask.data_vars:
            if v.startswith("mask_"):
                ds[v] = mask[v]
        write_netcdf_atomic(ds, out)
    return out
