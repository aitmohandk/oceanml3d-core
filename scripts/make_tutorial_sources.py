"""Fabricate a miniature GLORYS mirror and Argo GDAC, in their real layouts, for the tutorial.

    python scripts/make_tutorial_sources.py $OCEANML3D_DATA

writes, under the given data root:

* ``raw/glorys/<year>/mercatorglorys12v1_gl12_mean_<YYYYMMDD>_R<YYYYMMDD>.nc`` -- one file per day,
  CMEMS names and dimensions (``time, depth, latitude, longitude``), ``thetao, uo, vo, zos`` on the
  26 upper GLORYS levels, over a box a little larger than the Gulf Stream task domain, at 0.25 deg.
  ``raw/glorys`` is where ``jobs/env/_lib.sh`` points ``GLORYS_SRC`` by default, so
  ``jobs/prepare.sh glorys <year>`` finds it without any setting.
* ``gdac/ar_index_global_prof.txt`` and ``gdac/dac/aoml/<wmo>/<wmo>_prof.nc`` -- the native GDAC
  tree, QC flags as ``char(N_PROF, N_LEVELS)`` the way the GDAC writes them. Point ``ARGO_GDAC`` at
  it and ``jobs/prepare.sh argo`` reads it exactly as it reads Datarmor's mirror.

What this is *not*: a physical ocean. The fields are a meandering front with noise -- enough for every
step of the chain to run on real tools and real configs in a few minutes on a laptop, and for the
checks each step prints to mean something. `docs/tutorial.md` walks through it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

YEARS = (2018, 2019)
DAYS_PER_YEAR = 45
DEPTH = np.array([0.494, 1.54, 2.65, 3.82, 5.08, 6.44, 7.93, 9.57, 11.4, 13.5, 15.8, 18.5, 21.6,
                  25.2, 29.4, 34.4, 40.3, 47.4, 55.8, 65.8, 77.9, 92.3, 109.7, 130.7, 155.9, 186.1])


def make_glorys(root: Path, rng: np.random.Generator) -> None:
    lat = np.arange(30.0, 46.01, 0.25)
    lon = np.arange(-68.0, -51.99, 0.25)
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    decay = np.exp(-DEPTH / 300)[:, None, None]
    for year in YEARS:
        out = root / "raw" / "glorys" / str(year)
        out.mkdir(parents=True, exist_ok=True)
        for day in pd.date_range(f"{year}-01-01", periods=DAYS_PER_YEAR):
            k = day.dayofyear
            front = np.tanh((yy - 38 - 2 * np.sin(xx / 3 + k / 20)) / 1.5)
            thetao = ((20 + 5 * front)[None] - 0.05 * DEPTH[:, None, None]
                      + 0.2 * rng.standard_normal((DEPTH.size, *yy.shape)))
            dims4 = ("time", "depth", "latitude", "longitude")
            ds = xr.Dataset(
                {"thetao": (dims4, thetao[None].astype("f4")),
                 "uo": (dims4, ((0.8 * (1 - front ** 2))[None] * decay)[None].astype("f4")),
                 "vo": (dims4, ((0.1 * np.sin(xx / 2 + k / 10))[None] * decay)[None].astype("f4")),
                 "zos": (("time", "latitude", "longitude"), (0.5 * front)[None].astype("f4"))},
                coords={"time": [day], "depth": DEPTH, "latitude": lat, "longitude": lon})
            ds.to_netcdf(out / f"mercatorglorys12v1_gl12_mean_{day:%Y%m%d}_R{day:%Y%m%d}.nc")


def make_gdac(root: Path, rng: np.random.Generator, n_floats: int = 12, n_prof: int = 8,
              n_lev: int = 60) -> None:
    import netCDF4

    gdac = root / "gdac"
    lines = ["# Title : Profile directory file of the Argo Global Data Assembly Center",
             "file,date,latitude,longitude,ocean,profiler_type,institution,date_update"]

    def chars(rows, width):
        return np.array([list(r.ljust(width).encode()) for r in rows], "u1").view("S1")

    for f in range(n_floats):
        wmo = f"49{f:05d}"
        dates = [pd.Timestamp(f"{YEARS[int(rng.integers(0, len(YEARS)))]}-01-02")
                 + pd.Timedelta(days=int(rng.integers(0, DAYS_PER_YEAR - 1))) for _ in range(n_prof)]
        lats, lons = rng.uniform(33, 43, n_prof), rng.uniform(-65, -55, n_prof)
        path = gdac / "dac" / "aoml" / wmo / f"{wmo}_prof.nc"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Real floats stop their ascent a few decibars below the surface: the top level is not sampled.
        pres = np.tile(np.linspace(4, 1000, n_lev), (n_prof, 1))
        temp = 20 - 0.01 * pres + 0.05 * rng.standard_normal(pres.shape)
        with netCDF4.Dataset(path, "w") as d:
            d.createDimension("N_PROF", n_prof)
            d.createDimension("N_LEVELS", n_lev)
            d.createDimension("STRING8", 8)
            for name, val in (("PRES", pres), ("TEMP", temp), ("PSAL", 35 + 0 * pres)):
                d.createVariable(name, "f4", ("N_PROF", "N_LEVELS"), fill_value=99999.0)[:] = val
                d.createVariable(f"{name}_QC", "S1", ("N_PROF", "N_LEVELS"))[:] = chars(["1" * n_lev] * n_prof, n_lev)
            d.createVariable("LATITUDE", "f8", ("N_PROF",))[:] = lats
            d.createVariable("LONGITUDE", "f8", ("N_PROF",))[:] = lons
            juld = d.createVariable("JULD", "f8", ("N_PROF",))
            juld.units = "days since 1950-01-01 00:00:00 UTC"
            juld[:] = [(t - pd.Timestamp("1950-01-01")).total_seconds() / 86400 + 0.4 for t in dates]
            for name in ("POSITION_QC", "JULD_QC"):
                d.createVariable(name, "S1", ("N_PROF",))[:] = np.array([b"1"] * n_prof)
            d.createVariable("DATA_MODE", "S1", ("N_PROF",))[:] = np.array([b"R"] * n_prof)
            d.createVariable("PLATFORM_NUMBER", "S1", ("N_PROF", "STRING8"))[:] = chars([wmo] * n_prof, 8)
            d.createVariable("CYCLE_NUMBER", "i4", ("N_PROF",))[:] = np.arange(1, n_prof + 1)
        for c, t in enumerate(dates):
            lines.append(f"aoml/{wmo}/profiles/R{wmo}_{c + 1:03d}.nc,{t:%Y%m%d}093000,"
                         f"{lats[c]:.3f},{lons[c]:.3f},A,846,AO,20200101000000")
    (gdac / "ar_index_global_prof.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("root", help="the data root, i.e. $OCEANML3D_DATA")
    root = Path(p.parse_args().root)
    rng = np.random.default_rng(0)
    make_glorys(root, rng)
    make_gdac(root, rng)
    print(f"GLORYS mirror: {root / 'raw' / 'glorys'}  ({len(YEARS)} years x {DAYS_PER_YEAR} days)")
    print(f"Argo GDAC:     {root / 'gdac'}  (export ARGO_GDAC={root / 'gdac'})")


if __name__ == "__main__":
    main()
