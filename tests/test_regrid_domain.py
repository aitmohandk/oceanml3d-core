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


# --- resolution (NOSC's --target-res) -------------------------------------------------------------

from scripts.prepare.regrid import domain_grid, resolution_of  # noqa: E402

GS = {"lat": [32, 44], "lon": [-66, -54]}


def _native(tmp_path, lat_desc=False, lon360=False, t0s=("2010-01-01", "2010-01-03")):
    """A 1/12-deg-like source whose cell centres do NOT fall on the domain bounds."""
    lat = np.arange(20.0 + 1 / 24, 56.0, 1 / 12)
    lon = (np.arange(280.0 + 1 / 24, 320.0, 1 / 12) if lon360 else np.arange(-80.0 + 1 / 24, -40.0, 1 / 12))
    if lat_desc:
        lat = lat[::-1]
    for i, t0 in enumerate(t0s):
        time = xr.date_range(t0, periods=2, freq="D")
        field = np.broadcast_to(lat[:, None] + 0 * lon[None, :], (2, lat.size, lon.size)).astype("f4")
        xr.Dataset({"zos": (("time", "latitude", "longitude"), field)},
                   coords={"time": time, "latitude": lat, "longitude": lon}).to_netcdf(tmp_path / f"n{i}.nc")


def _recipe(tmp_path, res, out="out.nc", **kw):
    return {"input": str(tmp_path / "n*.nc"), "output": str(tmp_path / out),
            "variables": {"zos": "zos"}, "domain": GS, "resolution": res, "method": "none", **kw}


def test_resolution_placeholder_means_native():
    for raw in (None, "", "${OCEANML3D_TARGET_RES}", "native"):
        assert resolution_of({"resolution": raw, "domain": GS}) is None
    assert resolution_of({"resolution": "0.25", "domain": GS}) == 0.25
    with pytest.raises(ValueError, match="domain"):
        resolution_of({"resolution": 0.25})
    with pytest.raises(ValueError):
        resolution_of({"resolution": -1, "domain": GS})


def test_domain_grid_is_nosc_target_grid():
    lat, lon = domain_grid(GS, 0.25)
    assert lat[0] == 32 and lat[-1] == 44 and lat.size == 49
    assert lon[0] == -66 and lon[-1] == -54 and lon.size == 49


@pytest.mark.parametrize("lat_desc,lon360", [(False, False), (True, True)])
def test_run_at_quarter_degree_lands_on_the_shared_grid_without_nan(tmp_path, lat_desc, lon360):
    _native(tmp_path, lat_desc, lon360)
    out = run(_recipe(tmp_path, 0.25))
    with xr.open_dataset(out) as ds:
        lat, lon = domain_grid(GS, 0.25)
        np.testing.assert_allclose(ds.lat, lat)
        np.testing.assert_allclose(ds.lon, lon)
        assert not bool(ds.zos.isnull().any()), "edge nodes need source cells on both sides"
        np.testing.assert_allclose(ds.zos.isel(time=0), np.broadcast_to(lat[:, None], (49, 49)), atol=1e-4)
        assert ds.attrs["oceanml3d_resolution"] == "0.25"


def test_an_output_at_another_resolution_is_refused_not_skipped(tmp_path):
    _native(tmp_path)
    run(_recipe(tmp_path, 0.25))
    with pytest.raises(RuntimeError, match="resolution 0.25"):
        run(_recipe(tmp_path, "${OCEANML3D_TARGET_RES}"))
    run(_recipe(tmp_path, "${OCEANML3D_TARGET_RES}"), force=True)
    with xr.open_dataset(tmp_path / "out.nc") as ds:
        assert ds.attrs["oceanml3d_resolution"] == "native"


def test_merge_refuses_inputs_at_different_resolutions(tmp_path):
    _native(tmp_path, t0s=("2010-01-01",))
    run(_recipe(tmp_path, 0.25, out="by_year_a.nc"))
    (tmp_path / "n0.nc").rename(tmp_path / "n0.nc.bak")
    _native(tmp_path, t0s=("2010-01-05",))
    run(_recipe(tmp_path, 0.5, out="by_year_b.nc"))
    merge = {"input": str(tmp_path / "by_year_*.nc"), "output": str(tmp_path / "all.nc"),
             "variables": {"zos": "zos"}, "method": "none"}
    with pytest.raises(RuntimeError, match="different resolutions"):
        run(merge)


# --- Zarr end to end: per-year stores, merged with threads -----------------------------------------

def test_per_year_zarr_then_threaded_merge(tmp_path):
    pytest.importorskip("zarr")
    src = tmp_path / "src"
    src.mkdir()
    for i, t0 in enumerate(["2010-01-01", "2010-01-03", "2011-01-01"]):
        _globe(lat_desc=True, t0=t0).to_netcdf(src / f"f_{t0[:4]}_{i}.nc")
    for year in ("2010", "2011"):
        run({"input": str(src / f"f_{year}_*.nc"),
             "output": str(tmp_path / "by_year" / f"g_{year}.zarr"),
             "variables": {"thetao": "thetao", "zos": "zos"}, "domain": GS,
             "resolution": "${OCEANML3D_TARGET_RES}", "keep_depth": True, "method": "none",
             "zarr_chunks": {"time": 32}})
    assert not list((tmp_path / "by_year").glob(".tmp.*")), "temporary stores must be renamed away"
    out = run({"input": str(tmp_path / "by_year" / "g_*.zarr"),
               "output": str(tmp_path / "glorys" / "all.zarr"),
               "variables": {"thetao": "thetao", "zos": "zos"}, "keep_depth": True, "method": "none",
               "parallel": True, "dask_scheduler": "threads"})
    with xr.open_zarr(out) as ds:
        assert dict(ds.sizes) == {"time": 6, "depth": 3, "lat": 7, "lon": 7}
        assert ds.attrs["oceanml3d_resolution"] == "native"


def test_an_interrupted_write_leaves_nothing_to_skip(tmp_path, monkeypatch):
    pytest.importorskip("zarr")
    import scripts.prepare.regrid as regrid

    _native(tmp_path)

    def boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(regrid, "_write_zarr", boom)
    with pytest.raises(KeyboardInterrupt):
        run(_recipe(tmp_path, 0.25, out="y.zarr"))
    assert not (tmp_path / "y.zarr").exists()
    monkeypatch.undo()
    assert run(_recipe(tmp_path, 0.25, out="y.zarr")).exists()


def test_the_glorys_chain_paths_meet():
    """Per-year output -> merge input -> merge output -> catalog key and ARGO truth."""
    rec = REPO / "scripts/prepare/recipes"
    per_year = yaml.safe_load((rec / "glorys_gs_multidepth.datarmor.yaml").read_text())["output"]
    concat = yaml.safe_load((rec / "glorys_gs_concat.yaml").read_text())
    argo = yaml.safe_load((rec / "argo_profiles_gs.yaml").read_text())
    catalog = yaml.safe_load((REPO / "config/paths/local.yaml").read_text())["datasets"]
    assert per_year.endswith(".zarr") and concat["input"].endswith(".zarr")
    assert fnmatch.fnmatch(per_year.replace("${YEAR}", "2010"), concat["input"])
    merged = concat["output"].replace("${YEAR}-${NEXT}", "2010-2020")
    assert merged == "${OCEANML3D_DATA}/" + catalog["glorys_gs_multidepth"]["path"]
    assert argo["truth"] == merged
    datarmor_argo = yaml.safe_load((rec / "argo_profiles_gs.datarmor.yaml").read_text())
    assert datarmor_argo["source"] == "gdac" and datarmor_argo["truth"] == merged
    for r in (argo, datarmor_argo):
        assert r["output"] == "${OCEANML3D_DATA}/" + catalog["argo_profiles_gs"]["path"]


def test_zarr_stores_are_format_2_and_the_inode_count_is_exact(tmp_path, capsys):
    pytest.importorskip("zarr")
    _native(tmp_path)
    out = run(_recipe(tmp_path, 0.25, out="y.zarr", zarr_chunks={"time": 1}))
    assert (out / ".zgroup").exists() and not (out / "zarr.json").exists()
    announced = int(capsys.readouterr().out.split(" inodes")[0].rsplit("-> ", 1)[1])
    actual = 1 + sum(len(d) + len(f) for _, d, f in os.walk(out))
    assert announced == actual


# --- one data directory per resolution, derived in jobs/env/_lib.sh --------------------------------

def _load_site(tmp_path, **env):
    import subprocess

    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    script = f'source "{REPO}/jobs/env/_lib.sh"; load_site local "{REPO}" >/dev/null; echo "$OCEANML3D_DATA"'
    clean = {"PATH": os.environ["PATH"], "HOME": str(home), **env}
    return subprocess.run(["bash", "-c", script], env=clean, capture_output=True, text=True,
                          check=True).stdout.strip()


def test_target_resolution_derives_the_data_directory(tmp_path):
    home = tmp_path / "home"
    assert _load_site(tmp_path) == f"{home}/data/oceanml3d"
    assert _load_site(tmp_path, OCEANML3D_TARGET_RES="0.25") == f"{home}/data/oceanml3d/res0.25"
    assert _load_site(tmp_path, OCEANML3D_TARGET_RES="native") == f"{home}/data/oceanml3d"
    # already suffixed (the old advice was to pass it by hand): left alone
    assert _load_site(tmp_path, OCEANML3D_TARGET_RES="0.25",
                      OCEANML3D_DATA="/d/oceanml3d/res0.25") == "/d/oceanml3d/res0.25"


def test_user_settings_file_applies_and_the_command_line_still_wins(tmp_path):
    cfg = tmp_path / "home/.config/oceanml3d"
    cfg.mkdir(parents=True)
    (cfg / "env.sh").write_text('export OCEANML3D_TARGET_RES="${OCEANML3D_TARGET_RES:-0.25}"\n')
    home = tmp_path / "home"
    assert _load_site(tmp_path) == f"{home}/data/oceanml3d/res0.25"
    assert _load_site(tmp_path, OCEANML3D_TARGET_RES="0.5") == f"{home}/data/oceanml3d/res0.5"


def test_an_empty_merge_says_where_the_files_are(tmp_path):
    pytest.importorskip("zarr")
    (tmp_path / "res0.25" / "by_year" / "g_2010.zarr").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="res0.25.*OCEANML3D_TARGET_RES"):
        run({"input": str(tmp_path / "by_year" / "g_*.zarr"), "output": str(tmp_path / "all.zarr"),
             "variables": {"zos": "zos"}, "method": "none"})
