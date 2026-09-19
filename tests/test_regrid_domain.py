"""`regrid.py` domain subsetting, size guard, and the Datarmor GLORYS glob.

Regression for the Datarmor run that tried to write 1949 GB: the recipe's `domain:` was never read,
and `*2020*` matched production dates as well as validity dates.
"""
from __future__ import annotations

import fnmatch
import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

from scripts.prepare.regrid import _select_domain, check_size, run

REPO = Path(__file__).resolve().parents[1]


def _globe(lat_desc: bool = False, lon360: bool = False, ntime: int = 2, t0: str = "2010-01-01"):
    lat = np.arange(-80.0, 90.1, 2.0)
    lon = np.arange(0.0, 360.0, 2.0) if lon360 else np.arange(-180.0, 180.0, 2.0)
    if lat_desc:
        lat = lat[::-1]
    time = xr.date_range(t0, periods=ntime, freq="D")
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {
            "thetao": (("time", "depth", "latitude", "longitude"),
                       rng.normal(size=(ntime, 3, lat.size, lon.size)).astype("f4")),
            "zos": (("time", "latitude", "longitude"),
                    rng.normal(size=(ntime, lat.size, lon.size)).astype("f4")),
        },
        coords={"time": time, "depth": [0.5, 1.5, 2.6], "latitude": lat, "longitude": lon},
    )


def _norm(ds):
    return ds.rename({"latitude": "lat", "longitude": "lon"})


@pytest.mark.parametrize("lat_desc", [False, True])
@pytest.mark.parametrize("lon360", [False, True])
def test_domain_selects_the_box_whatever_the_conventions(lat_desc, lon360):
    ds = _select_domain(_norm(_globe(lat_desc, lon360)), {"lat": [32, 44], "lon": [-66, -54]})
    assert ds.sizes["lat"] == 7 and ds.sizes["lon"] == 7
    assert float(ds.lat.min()) == 32 and float(ds.lat.max()) == 44
    lon = ds.lon.values % 360
    assert lon.min() == 294 and lon.max() == 306


def test_domain_across_the_seam_stays_contiguous():
    ds = _select_domain(_norm(_globe(lon360=True)), {"lat": [0, 10], "lon": [-10, 10]})
    assert ds.lon.values.tolist() == list(np.arange(-10.0, 10.1, 2.0))


def test_domain_outside_the_data_is_an_error():
    with pytest.raises(ValueError, match="selects nothing"):
        _select_domain(_norm(_globe()), {"lat": [-89, -85]})


def test_size_guard():
    check_size(5.0, {})
    with pytest.raises(RuntimeError, match="max_gb"):
        check_size(1949.15, {})
    check_size(1949.15, {"max_gb": 3000})
    with pytest.raises(RuntimeError):
        check_size(21.0, {"max_gb": 20})


def test_run_applies_domain_per_file(tmp_path):
    for i, t0 in enumerate(["2010-01-01", "2010-01-03"]):
        _globe(lat_desc=True, t0=t0).to_netcdf(tmp_path / f"f{i}.nc")
    out = run({
        "input": str(tmp_path / "f*.nc"),
        "output": str(tmp_path / "out.nc"),
        "variables": {"thetao": "thetao", "zos": "zos"},
        "domain": {"lat": [32, 44], "lon": [-66, -54]},
        "keep_depth": True,
        "method": "none",
    })
    with xr.open_dataset(out) as ds:
        assert dict(ds.sizes) == {"time": 4, "depth": 3, "lat": 7, "lon": 7}


def test_datarmor_glorys_glob_anchors_the_validity_year():
    recipe = (REPO / "scripts/prepare/recipes/glorys_gs_multidepth.datarmor.yaml").read_text()
    env = {"GLORYS_SRC": "/m", "YEAR": "2020", "NEXT": "2021", "OCEANML3D_DATA": "/d"}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        r = yaml.safe_load(os.path.expandvars(recipe))
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert "domain" in r and r["time"] == ["2020-01-01", "2020-12-31"]
    pat = r["input"].replace("/**/", "/*/")
    name = "/m/2020/01/mercatorglorys12v1_gl12_mean_{}_R{}.nc"
    assert fnmatch.fnmatch(name.format("20200115", "20200122"), pat)
    assert fnmatch.fnmatch(name.format("20201231", "20210106"), pat)
    # the files the old `*2020*` pulled in through their production date
    assert not fnmatch.fnmatch(name.format("20020130", "20020206"), pat)
    assert not fnmatch.fnmatch(name.format("20220209", "20220216"), pat)
    assert not fnmatch.fnmatch(name.format("20191231", "20200108"), pat)
