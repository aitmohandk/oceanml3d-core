"""Write model outputs in the *product* format consumed by ``oceanml3d-eval``.

A product is:

* one NetCDF file per day ``<name>_YYYY-MM-DD.nc`` with CF-named variables on a
  regular ``lat/lon`` grid (``time`` of length 1);
* a ``product.yaml`` manifest describing where the files are and how to read them.

This replaces the hand-written ``metric/dictionary/*.json`` files of NOSC.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from oceanml3d.inference.product_contract import PRODUCT_FORMAT_VERSION, check_manifest
from oceanml3d.variables import VariableSet

DEFAULT_STANDARD_NAMES = {
    "u_drifter": ("u", "eastward_sea_water_velocity", "m s-1"),
    "v_drifter": ("v", "northward_sea_water_velocity", "m s-1"),
    "u": ("u", "eastward_sea_water_velocity", "m s-1"),
    "v": ("v", "northward_sea_water_velocity", "m s-1"),
    # CMEMS / GLORYS names. Without them the OSSE-3D product exported `uo_dNN` / `vo_dNN`, which the
    # contract shared with oceanml3d-eval does not recognise as currents (canonical: u, v).
    "uo": ("u", "eastward_sea_water_velocity", "m s-1"),
    "vo": ("v", "northward_sea_water_velocity", "m s-1"),
    "ssh": ("ssh", "sea_surface_height_above_geoid", "m"),
    "zos": ("ssh", "sea_surface_height_above_geoid", "m"),
    "sst": ("sst", "sea_surface_temperature", "degC"),
    "thetao": ("thetao", "sea_water_potential_temperature", "degC"),
    "so": ("so", "sea_water_salinity", "1e-3"),
}


def export_variable_names(variables: VariableSet) -> dict[str, tuple[str, str, str]]:
    out = {}
    for spec in variables.targets:
        base = spec.base_name
        short, std, units = DEFAULT_STANDARD_NAMES.get(base, (base, spec.standard_name or base, spec.units or ""))
        if spec.depth_index is not None:
            short = f"{short}_d{spec.depth_index:02d}"
        if spec.standard_name:
            std = spec.standard_name
        if spec.units:
            units = spec.units
        out[spec.name] = (short, std, units)
    return out


def write_product(field: xr.DataArray, variables: VariableSet, out_dir: str | Path, name: str,
                  time_slice: slice | None = None, attrs: Mapping[str, Any] | None = None,
                  depth_m: float | None = None) -> Path:
    """``field`` has dims ``(channel, time, lat, lon)``, optionally with a leading ``member``
    dimension for ensemble methods (EnKF, flow-matching / diffusion samplers); the manifest then
    carries ``ensemble_size`` and ``coords.member`` (product format v2)."""
    ensemble = "member" in field.dims
    out_dir = Path(out_dir)
    daily = out_dir / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    if time_slice is not None:
        field = field.sel(time=time_slice)
    names = export_variable_names(variables)

    ds = xr.Dataset()
    for src, (short, std, units) in names.items():
        da = field.sel(channel=src).drop_vars("channel")
        da.attrs.update({"standard_name": std, "units": units, "source_variable": src})
        spec = variables[src]
        if spec.depth_index is not None:
            da.attrs["depth_index"] = int(spec.depth_index)
            if spec.depth_m is not None:
                da.attrs["depth_m"] = float(spec.depth_m)
        if spec.group:
            da.attrs["group"] = spec.group
        ds[short] = da
    ds.attrs.update({"product": name, "conventions": "CF-1.8", **(attrs or {})})
    if ensemble:
        ds.attrs["ensemble_size"] = int(field.sizes["member"])
    if depth_m is not None:
        ds.attrs["depth_m"] = float(depth_m)

    for t in ds.time.values:
        day = pd.Timestamp(t).strftime("%Y-%m-%d")
        ds.sel(time=[t]).to_netcdf(daily / f"{name}_{day}.nc")

    manifest = {
        "format_version": PRODUCT_FORMAT_VERSION,
        "name": name,
        "path": str(daily.resolve()),
        "pattern": f"{name}_(\\d{{4}})-(\\d{{2}})-(\\d{{2}})\\.nc",
        "variables": {short: {"standard_name": std, "units": units,
                              **({"depth_index": variables[src].depth_index} if variables[src].depth_index is not None else {}),
                              **({"group": variables[src].group} if variables[src].group else {})}
                      for src, (short, std, units) in names.items()},
        "coords": {"lat": "lat", "lon": "lon", "time": "time", **({"member": "member"} if ensemble else {})},
        "ensemble_size": int(field.sizes["member"]) if ensemble else None,
        "time_coverage_hours": 24,
        "depth_m": depth_m,
        "first_date": pd.Timestamp(ds.time.values[0]).strftime("%Y-%m-%d"),
        "last_date": pd.Timestamp(ds.time.values[-1]).strftime("%Y-%m-%d"),
        "attrs": {k: (v if isinstance(v, (str, int, float)) else json.dumps(v)) for k, v in (attrs or {}).items()},
    }
    check_manifest(manifest, source=str(out_dir / "product.yaml"))
    manifest_path = out_dir / "product.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return manifest_path


def open_product(manifest_path: str | Path) -> xr.Dataset:
    """Convenience reader (mirror of ``oceanml3d_eval.product.open_product``)."""
    man = yaml.safe_load(Path(manifest_path).read_text())
    files = sorted(Path(man["path"]).glob("*.nc"))
    return xr.open_mfdataset(files, combine="by_coords") if files else xr.Dataset()


def stack_predictions(predictions: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(predictions, axis=0)
