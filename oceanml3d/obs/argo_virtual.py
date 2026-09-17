"""Virtual ARGO floats for a GLORYS-truth OSSE (port of NOSC contrib/argo/virtual.py).

Real floats' *geometry* (position, date, vertical coverage) is kept, their values are
replaced by the truth sampled at those points, then binned onto the grid, one variable
per depth index (``<var>_d<ii>``) so the result plugs into ``depth_indices`` variables.

Input ``profiles``: DataFrame with columns ``time, lat, lon`` and one column per depth
index ``d<ii>`` holding 1 where the real profile covered that level (NaN/0 otherwise).
Producing that table from raw ARGO (argopy download + QC + vertical interpolation) is
left to ``scripts/prepare/argo_profiles.py`` (see NOSC contrib/argo/{download,qc,vertical_interp}.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from oceanml3d.data.open import normalise_dims
from oceanml3d.io import write_netcdf_atomic


def virtualize(profiles: pd.DataFrame, truth: str | Path, truth_var: str, depth_indices: list[int],
               time_tolerance: str = "2D", noise_std: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """``noise_std`` adds Gaussian instrument + representativeness noise (point vs grid cell)."""
    da = normalise_dims(xr.open_dataset(truth))[truth_var]
    rng = np.random.default_rng(seed)
    out = profiles[["time", "lat", "lon"]].copy()
    out["time"] = pd.to_datetime(out["time"])
    for i in depth_indices:
        col = f"d{i:02d}"
        covered = profiles[col].fillna(0).values > 0 if col in profiles else np.ones(len(profiles), bool)
        vals = np.full(len(out), np.nan, np.float32)
        if covered.any():
            pts = out[covered]
            sampled = da.isel(depth=i).sel(time=xr.DataArray(pts.time.values, dims="p"),
                                            lat=xr.DataArray(pts.lat.values, dims="p"),
                                            lon=xr.DataArray(pts.lon.values, dims="p"), method="nearest")
            ok = np.abs(sampled.time.values - pts.time.values) <= pd.Timedelta(time_tolerance)
            vals[covered] = np.where(ok, sampled.values, np.nan)
            if noise_std > 0:
                vals[covered] += rng.normal(0.0, noise_std, covered.sum()).astype(np.float32)
        out[f"{truth_var}_d{i:02d}"] = vals
    return out


def grid_daily(virtual: pd.DataFrame, truth: str | Path, columns: list[str]) -> xr.Dataset:
    ref = normalise_dims(xr.open_dataset(truth))
    lat, lon = ref.lat.values, ref.lon.values
    time = pd.DatetimeIndex(ref.time.values).normalize()
    dlat, dlon = np.diff(lat).mean(), np.diff(lon).mean()
    lat_b = np.concatenate([lat - dlat / 2, [lat[-1] + dlat / 2]])
    lon_b = np.concatenate([lon - dlon / 2, [lon[-1] + dlon / 2]])
    day = virtual.time.dt.normalize()
    data = {c: np.full((len(time), len(lat), len(lon)), np.nan, np.float32) for c in columns}
    t_index = {t: k for k, t in enumerate(time)}
    for d, grp in virtual.groupby(day):
        if d not in t_index:
            continue
        for c in columns:
            g = grp.dropna(subset=[c])
            if g.empty:
                continue
            m = stats.binned_statistic_2d(g.lat, g.lon, g[c], "mean", bins=[lat_b, lon_b])[0]
            data[c][t_index[d]] = m
    ds = xr.Dataset({c: (("time", "lat", "lon"), v) for c, v in data.items()},
                    coords={"time": time, "lat": lat, "lon": lon}, attrs={"source": "virtual ARGO", "truth": str(truth)})
    for c in columns:                                   # explicit presence channel per level
        ds[f"{c}_mask"] = np.isfinite(ds[c]).astype(np.float32)
    return ds


def build_virtual_argo(profiles: pd.DataFrame, truth: str | Path, truth_var: str, depth_indices: list[int],
                       output: str | Path, noise_std: float = 0.0, seed: int = 0) -> Path:
    v = virtualize(profiles, truth, truth_var, depth_indices, noise_std=noise_std, seed=seed)
    cols = [f"{truth_var}_d{i:02d}" for i in depth_indices]
    ds = grid_daily(v, truth, cols)
    return write_netcdf_atomic(ds, output)
