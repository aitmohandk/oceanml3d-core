"""Ported from NOSC first_implementation tests/test_argo_and_pseudo_obs.py (numpy part)."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from oceanml3d.obs import argo


def _cloud():
    # profile A unsorted in pressure (10, 30, 20, 30): the 20 dbar level is valid but a naive
    # inversion test on the unsorted order rejects it; profile B has a spike at 30 dbar.
    plat = np.array(["A", "A", "A", "A", "B", "B", "B"])
    cyc = np.array([1, 1, 1, 1, 1, 1, 1])
    pres = np.array([10.0, 30.0, 20.0, 30.0, 10.0, 30.0, 50.0])
    temp = np.array([20.0, 18.0, 19.0, 17.0, 21.0, 30.0, 15.0])
    qc = np.ones(7, np.int8)
    return xr.Dataset({"PLATFORM_NUMBER": ("N_POINTS", plat), "CYCLE_NUMBER": ("N_POINTS", cyc), "PRES": ("N_POINTS", pres),
                       "TEMP": ("N_POINTS", temp), "LATITUDE": ("N_POINTS", np.full(7, 40.0)), "LONGITUDE": ("N_POINTS", np.full(7, -60.0)),
                       "TIME": ("N_POINTS", np.repeat(np.datetime64("2019-01-05"), 7)),
                       "PRES_QC": ("N_POINTS", qc), "TEMP_QC": ("N_POINTS", qc), "POSITION_QC": ("N_POINTS", qc), "JULD_QC": ("N_POINTS", qc)})


def test_sorting_fixes_false_inversion_rejections():
    ds = _cloud()
    unsorted = argo.reject_pressure_inversions(ds)
    sorted_ = argo.reject_pressure_inversions(argo.sort_pointcloud(ds))
    a_uns = unsorted.PRES.values[unsorted.PLATFORM_NUMBER.values == "A"]
    a_srt = sorted_.PRES.values[sorted_.PLATFORM_NUMBER.values == "A"]
    assert 20.0 not in a_uns                                        # false rejection without sorting
    assert list(a_srt) == [10.0, 20.0, 30.0]                        # sorted: only the duplicate 30 dbar dropped


def test_spike_and_coverage_table():
    ds = argo.apply_standard_qc(_cloud(), spike_thresholds={"TEMP": 5.0})
    assert 30.0 not in ds.TEMP.values
    table = argo.coverage_table(ds, depth_values=[5.0, 25.0, 100.0], depth_indices=[0, 3, 9])
    assert len(table) == 2 and set(table.columns) >= {"d00", "d03", "d09", "TEMP_d03"}
    a = table[table.profile_id == "A_1"].iloc[0]
    assert a.d00 == 0 and a.d03 == 1 and a.d09 == 0                   # covers 10-30 dbar only
    assert abs(a.TEMP_d03 - 18.5) < 1e-6


def test_gdac_pointcloud_flatten():
    ds = xr.Dataset({"PRES": (("N_PROF", "N_LEVELS"), [[5.0, 15.0], [8.0, np.nan]]),
                     "TEMP": (("N_PROF", "N_LEVELS"), [[20.0, 19.0], [21.0, np.nan]]),
                     "PRES_QC": (("N_PROF", "N_LEVELS"), [[b"1", b"1"], [b"1", b" "]]),
                     "TEMP_QC": (("N_PROF", "N_LEVELS"), [[b"1", b"1"], [b"1", b" "]]),
                     "LATITUDE": ("N_PROF", [40.0, 41.0]), "LONGITUDE": ("N_PROF", [-60.0, -61.0]),
                     "JULD": ("N_PROF", pd.to_datetime(["2019-01-01", "2019-01-02"]).values),
                     "POSITION_QC": ("N_PROF", [b"1", b"1"]), "JULD_QC": ("N_PROF", [b"1", b"1"]),
                     "PLATFORM_NUMBER": ("N_PROF", [b"6901", b"6902"]), "CYCLE_NUMBER": ("N_PROF", [3, 4])})
    pc = argo.pointcloud_from_gdac_file(ds)
    assert pc.sizes["N_POINTS"] == 4 and list(pc.PRES_QC.values) == [1, 1, 1, 9]
    assert argo.filter_by_qc(pc, ["PRES_QC"]).sizes["N_POINTS"] == 3


def _prof(path, lat, lon, date, wmo):
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"PRES": (("N_PROF", "N_LEVELS"), [[5.0, 15.0]]),
                "TEMP": (("N_PROF", "N_LEVELS"), [[20.0, 19.0]]),
                "PRES_QC": (("N_PROF", "N_LEVELS"), [[b"1", b"1"]]),
                "TEMP_QC": (("N_PROF", "N_LEVELS"), [[b"1", b"1"]]),
                "LATITUDE": ("N_PROF", [lat]), "LONGITUDE": ("N_PROF", [lon]),
                "JULD": ("N_PROF", pd.to_datetime([date]).values),
                "POSITION_QC": ("N_PROF", [b"1"]), "JULD_QC": ("N_PROF", [b"1"]),
                "PLATFORM_NUMBER": ("N_PROF", [wmo.encode()]), "CYCLE_NUMBER": ("N_PROF", [1])}).to_netcdf(path)


def test_local_gdac_opens_only_the_floats_the_index_selects(tmp_path, monkeypatch):
    """A GDAC laid out like /home/ref-argo/gdac: index at the root, paths relative to dac/."""
    _prof(tmp_path / "dac/aoml/1900001/1900001_prof.nc", 40.0, -60.0, "2015-06-01", "1900001")   # in the box
    _prof(tmp_path / "dac/coriolis/6900002/6900002_prof.nc", 10.0, -20.0, "2015-06-01", "6900002")  # outside
    (tmp_path / "ar_index_global_prof.txt").write_text(
        "# Title : Profile directory file of the Argo Global Data Assembly Center\n"
        "file,date,latitude,longitude,ocean,profiler_type,institution,date_update\n"
        "aoml/1900001/profiles/R1900001_001.nc,20150601000000,40.0,-60.0,A,846,AO,20160101000000\n"
        "coriolis/6900002/profiles/R6900002_001.nc,20150601000000,10.0,-20.0,A,846,IF,20160101000000\n"
        "aoml/1900001/profiles/R1900001_002.nc,20090101000000,40.0,-60.0,A,846,AO,20160101000000\n")
    files = argo.prof_files_from_index(tmp_path, -66, -54, 32, 44, "2010-01-01", "2020-01-11")
    assert files == [tmp_path / "dac/aoml/1900001/1900001_prof.nc"]

    opened = []
    real_open = xr.open_dataset
    monkeypatch.setattr(argo.xr, "open_dataset", lambda f, *a, **k: opened.append(f) or real_open(f, *a, **k))
    pc = argo.fetch_argo_profiles_local(tmp_path, -66, -54, 32, 44, "2010-01-01", "2020-01-11")
    assert opened == files and pc.sizes["N_POINTS"] == 2


def test_local_gdac_without_index_falls_back_to_scanning(tmp_path):
    _prof(tmp_path / "aoml/1900001/1900001_prof.nc", 40.0, -60.0, "2015-06-01", "1900001")
    assert argo.prof_files_from_index(tmp_path, -66, -54, 32, 44, "2010-01-01", "2020-01-11") is None
    pc = argo.fetch_argo_profiles_local(tmp_path, -66, -54, 32, 44, "2010-01-01", "2020-01-11")
    assert pc.sizes["N_POINTS"] == 2


# --- real GDAC layout: char QC arrays, DATA_MODE, many cycles per float ------------------------------

def _real_prof(path, lats, lons, dates, wmo="1900001", n_lev=4, modes=None):
    """Written with netCDF4 as the GDAC writes it: QC flags as char(N_PROF, N_LEVELS), which xarray
    decodes into one string per profile -- the layout the synthetic fixtures above do not have."""
    netCDF4 = pytest.importorskip("netCDF4")
    n = len(lats)
    path.parent.mkdir(parents=True, exist_ok=True)
    chars = lambda rows, width: np.array([list(r.ljust(width).encode()) for r in rows], "u1").view("S1")  # noqa: E731
    with netCDF4.Dataset(path, "w") as d:
        d.createDimension("N_PROF", n)
        d.createDimension("N_LEVELS", n_lev)
        d.createDimension("STRING8", 8)
        for name, vals in (("LATITUDE", lats), ("LONGITUDE", lons)):
            d.createVariable(name, "f8", ("N_PROF",))[:] = vals
        juld = d.createVariable("JULD", "f8", ("N_PROF",))
        juld.units = "days since 1950-01-01 00:00:00"
        juld[:] = (pd.to_datetime(dates) - pd.Timestamp("1950-01-01")).days
        d.createVariable("CYCLE_NUMBER", "i4", ("N_PROF",))[:] = np.arange(1, n + 1)
        d.createVariable("PLATFORM_NUMBER", "S1", ("N_PROF", "STRING8"))[:] = chars([wmo] * n, 8)
        d.createVariable("DATA_MODE", "S1", ("N_PROF",))[:] = np.array([m.encode() for m in (modes or "R" * n)], "S1")
        d.createVariable("POSITION_QC", "S1", ("N_PROF",))[:] = np.array([b"1"] * n, "S1")
        d.createVariable("JULD_QC", "S1", ("N_PROF",))[:] = np.array([b"1"] * n, "S1")
        pres = np.tile(np.arange(n_lev, dtype=float) * 10 + 5, (n, 1))
        for name, vals in (("PRES", pres), ("PRES_ADJUSTED", pres), ("TEMP", 20 - pres / 10),
                           ("TEMP_ADJUSTED", 30 - pres / 10)):
            d.createVariable(name, "f4", ("N_PROF", "N_LEVELS"))[:] = vals
        qc = ["1" * (n_lev - 1) + "4"] * n                      # last level bad
        for name in ("PRES_QC", "TEMP_QC", "PRES_ADJUSTED_QC", "TEMP_ADJUSTED_QC"):
            d.createVariable(name, "S1", ("N_PROF", "N_LEVELS"))[:] = chars(qc, n_lev)


def test_real_gdac_char_qc_flags_are_decoded_per_level(tmp_path):
    f = tmp_path / "1900001_prof.nc"
    _real_prof(f, [40.0, 41.0], [-60.0, -61.0], ["2015-06-01", "2015-06-11"])
    with xr.open_dataset(f) as ds:
        pc = argo.pointcloud_from_gdac_file(ds)
    assert pc.sizes["N_POINTS"] == 8
    assert list(pc.PRES_QC.values) == [1, 1, 1, 4] * 2
    assert set(pc.PLATFORM_NUMBER.values) == {"1900001"}
    assert argo.filter_by_qc(pc, ["PRES_QC", "TEMP_QC"]).sizes["N_POINTS"] == 6


def test_qc_flags_joined_into_one_string_per_profile_are_split_back():
    """Depending on how the dimension is used, xarray's char decoding joins char(N_PROF, N_LEVELS)
    into (N_PROF,) strings -- b'1114' for one profile. Both layouts must give per-level flags."""
    joined = xr.DataArray(np.array([b"1114", b"12  "], "S4"), dims="N_PROF")
    split = argo._decode_qc(argo._qc_levels(joined, 2, 4))
    assert split.tolist() == [[1, 1, 1, 4], [1, 2, 9, 9]]
    short = xr.DataArray(np.array([b"11", b"1"], "S2"), dims="N_PROF")        # trailing blanks trimmed
    assert argo._decode_qc(argo._qc_levels(short, 2, 3)).tolist() == [[1, 1, 9], [1, 9, 9]]
    per_level = xr.DataArray(np.array([[b"1", b"4"]], "S1"), dims=("N_PROF", "N_LEVELS"))
    assert argo._decode_qc(argo._qc_levels(per_level, 1, 2)).tolist() == [[1, 4]]


def test_adjusted_values_replace_raw_ones_in_delayed_mode(tmp_path):
    f = tmp_path / "1900001_prof.nc"
    _real_prof(f, [40.0, 41.0], [-60.0, -61.0], ["2015-06-01", "2015-06-11"], modes="RD")
    with xr.open_dataset(f) as ds:
        pc = argo.pointcloud_from_gdac_file(ds)
    temp = pc.TEMP.values.reshape(2, 4)
    np.testing.assert_allclose(temp[0], 20 - (np.arange(4) * 10 + 5) / 10)      # R: raw
    np.testing.assert_allclose(temp[1], 30 - (np.arange(4) * 10 + 5) / 10)      # D: adjusted


def test_only_the_profiles_in_the_box_are_flattened(tmp_path):
    """A <wmo>_prof.nc holds the float's whole life; 2 cycles of 300 cross the box."""
    n = 300
    lats = np.full(n, 10.0)
    lats[[5, 200]] = 40.0
    _real_prof(tmp_path / "dac/aoml/1900001/1900001_prof.nc", lats, np.full(n, -60.0),
               pd.date_range("2012-01-01", periods=n, freq="10D"), n_lev=500)
    pc = argo.fetch_argo_profiles_local(tmp_path, -66, -54, 32, 44, "2010-01-01", "2020-01-11")
    assert pc.sizes["N_POINTS"] == 2 * 500


def test_without_index_the_scan_does_not_descend_into_profiles(tmp_path):
    _real_prof(tmp_path / "dac/aoml/1900001/1900001_prof.nc", [40.0], [-60.0], ["2015-06-01"])
    _real_prof(tmp_path / "dac/aoml/1900001/profiles/R1900001_001_prof.nc", [40.0], [-60.0], ["2015-06-01"])
    assert argo._scan_prof_files(tmp_path) == [tmp_path / "dac/aoml/1900001/1900001_prof.nc"]


def test_decode_qc_is_vectorised_and_total():
    got = argo._decode_qc(np.array([b"1", b"4", b" ", b"", b"9", b"x"], "S1"))
    assert got.tolist() == [1, 4, 9, 9, 9, 9]
    assert argo._decode_qc(np.array(["1", " 2 ", ""])).tolist() == [1, 2, 9]
    assert argo._decode_qc(np.array([b"3", "5"], dtype=object)).tolist() == [3, 5]


def test_argo_recipe_end_to_end_on_a_fake_gdac(tmp_path):
    import importlib.util

    _real_prof(tmp_path / "gdac/dac/aoml/1900001/1900001_prof.nc", [40.0, 41.0], [-60.0, -61.0],
               ["2015-06-01", "2015-06-11"], n_lev=6)
    (tmp_path / "gdac/ar_index_global_prof.txt").write_text(
        "# header\nfile,date,latitude,longitude,ocean,profiler_type,institution,date_update\n"
        "aoml/1900001/profiles/R1900001_001.nc,20150601000000,40.0,-60.0,A,846,AO,20160101000000\n")
    xr.Dataset(coords={"depth": [5.0, 15.0, 25.0, 35.0, 45.0, 55.0]}).to_netcdf(tmp_path / "truth.nc")
    spec = importlib.util.spec_from_file_location("argo_profiles", "scripts/prepare/argo_profiles.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.run({"source": "gdac", "gdac_dir": str(tmp_path / "gdac"), "bbox": [-66, -54, 32, 44],
                   "time": ["2010-01-01", "2020-01-11"], "truth": str(tmp_path / "truth.nc"),
                   "depth_indices": [0, 2, 4], "value_var": "TEMP", "output": str(tmp_path / "argo.csv")})
    table = pd.read_csv(out)
    assert len(table) == 2 and {"d00", "d02", "d04"} <= set(table.columns)
    assert table["d00"].eq(1).all()
