"""Bin drifter trajectories into daily gridded mean velocities (targets of the surface
currents task). Generalises NOSC ``make_daily_uv_map_{aoml,cmems}*.py``.

Recipe (YAML):
    input: /path/drifter_*_{year}.nc        # AOML yearly files
    output: /path/drifters_uv_aoml_15m_4th.nc
    reference: /path/ref_grid.nc            # or grid: {...} as in regrid.py
    time: ["2010-01-01", "2023-01-01"]
    columns: {lon: lon, lat: lat, u: ums, v: vms, time: date}
    max_speed: 1000                          # cm/s or m/s filter (as in NOSC)
    scale: 1.0                               # multiply u/v (e.g. 0.01 cm/s -> m/s)
    depth_m: 15
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from regrid import target_grid
from scipy import stats


def bin_day(df: pd.DataFrame, cols: dict, lon_bins, lat_bins) -> tuple[np.ndarray, np.ndarray]:
    if df.empty:
        shape = (len(lat_bins) - 1, len(lon_bins) - 1)
        return np.full(shape, np.nan, np.float32), np.full(shape, np.nan, np.float32)
    u = stats.binned_statistic_2d(df[cols["lat"]], df[cols["lon"]], df[cols["u"]], "mean", bins=[lat_bins, lon_bins])[0]
    v = stats.binned_statistic_2d(df[cols["lat"]], df[cols["lon"]], df[cols["v"]], "mean", bins=[lat_bins, lon_bins])[0]
    return u.astype(np.float32), v.astype(np.float32)


def run(recipe: dict) -> Path:
    lat, lon = target_grid(recipe)
    dlat, dlon = np.diff(lat).mean(), np.diff(lon).mean()
    lat_bins = np.concatenate([lat - dlat / 2, [lat[-1] + dlat / 2]])
    lon_bins = np.concatenate([lon - dlon / 2, [lon[-1] + dlon / 2]])
    cols = recipe["columns"]
    days = pd.date_range(*recipe["time"], freq="D", inclusive="left")
    u_all, v_all = [], []
    for year, group in days.groupby(days.year).items():
        ds = xr.open_mfdataset(recipe["input"].format(year=year), combine="nested", concat_dim=cols["time"])
        df = ds[[cols["lon"], cols["lat"], cols["u"], cols["v"]]].to_dataframe().reset_index()
        df = df[(df[cols["u"]].abs() < recipe.get("max_speed", np.inf)) & (df[cols["v"]].abs() < recipe.get("max_speed", np.inf))]
        df[[cols["u"], cols["v"]]] *= recipe.get("scale", 1.0)
        df["_day"] = pd.to_datetime(df[cols["time"]]).dt.normalize()
        by_day = dict(tuple(df.groupby("_day")))
        for day in group:
            u, v = bin_day(by_day.get(day, pd.DataFrame(columns=df.columns)), cols, lon_bins, lat_bins)
            u_all.append(u)
            v_all.append(v)
        print(year, "done")
    out = xr.Dataset(
        {"u_drifter": (("time", "lat", "lon"), np.stack(u_all)), "v_drifter": (("time", "lat", "lon"), np.stack(v_all))},
        coords={"time": days, "lat": lat, "lon": lon},
        attrs={"depth_m": recipe.get("depth_m"), "source": recipe["input"]},
    )
    path = Path(recipe["output"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(path, encoding={v: {"zlib": True, "complevel": 4} for v in out.data_vars})
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    a = p.parse_args()
    print(run(yaml.safe_load(Path(a.config).read_text())))
