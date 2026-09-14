"""Regrid any gridded product onto the reference grid of a task (replaces the ~20
``interpolation_*_4th.py`` scripts of NOSC with one recipe-driven tool).

Recipe (YAML):
    input: /path/or/glob/*.nc
    output: /path/out_4th.nc
    variables: {adt: zos, ugos: ugos, vgos: vgos}   # rename on the fly
    grid: {lat: [-80, 90, 0.25], lon: [-180, 180, 0.25]}   # or  reference: /path/ref.nc
    time: ["2010-01-01", "2022-01-01"]
    method: linear                                     # linear | nearest | conservative (xesmf)
    chunks: {time: 30}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from oceanml3d.data.open import normalise_dims


def target_grid(recipe: dict) -> tuple[np.ndarray, np.ndarray]:
    if "reference" in recipe:
        ref = normalise_dims(xr.open_dataset(recipe["reference"]))
        return ref.lat.values, ref.lon.values
    g = recipe["grid"]
    lat = np.arange(g["lat"][0], g["lat"][1] + 1e-9, g["lat"][2])
    lon = np.arange(g["lon"][0], g["lon"][1] + 1e-9, g["lon"][2])
    return lat, lon


def run(recipe: dict) -> Path:
    ds = normalise_dims(xr.open_mfdataset(recipe["input"], combine="by_coords", chunks=recipe.get("chunks", {"time": 30})))
    ds = ds[list(recipe["variables"])].rename(recipe["variables"])
    if "time" in recipe:
        ds = ds.sel(time=slice(*recipe["time"]))
    if "depth" in ds.dims and not recipe.get("keep_depth", False):
        ds = ds.isel(depth=0, drop=True)
    elif "depth" in ds.dims and recipe.get("depth_indices"):
        ds = ds.isel(depth=list(recipe["depth_indices"]))
    method = recipe.get("method", "linear")
    if method == "none":
        out = ds.astype(np.float32)
        path = Path(recipe["output"])
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_netcdf(path, encoding={v: {"zlib": True, "complevel": 4} for v in out.data_vars})
        return path
    lat, lon = target_grid(recipe)
    if method == "conservative":
        import xesmf as xe

        out = xe.Regridder(ds, xr.Dataset({"lat": lat, "lon": lon}), "conservative")(ds)
    else:
        out = ds.interp(lat=lat, lon=lon, method=method)
    out = out.astype(np.float32)
    path = Path(recipe["output"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(path, encoding={v: {"zlib": True, "complevel": 4} for v in out.data_vars})
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    a = p.parse_args()
    import os
    print(run(yaml.safe_load(os.path.expandvars(Path(a.config).read_text()))))
