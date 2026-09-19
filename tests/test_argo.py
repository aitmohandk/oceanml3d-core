"""Ported from NOSC first_implementation tests/test_argo_and_pseudo_obs.py (numpy part)."""
import numpy as np
import pandas as pd
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
