"""ARGO profiles: download (argopy or local GDAC), QC, vertical interpolation, and the
per-profile *coverage table* consumed by :mod:`oceanml3d.obs.argo_virtual`.

Port of NOSC ``contrib/argo/{download,qc,vertical_interp}.py``. Heavy dependencies
(argopy) are imported lazily; the QC / interpolation part is pure numpy/pandas and tested.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

GOOD_FLAGS = (1, 2)
VALUE_VARS = ("TEMP", "PSAL")


# ----------------------------------------------------------------------------- download
def fetch_argo_profiles(lon_min, lon_max, lat_min, lat_max, start_date, end_date,
                        min_depth=0, max_depth=2000, mode="standard") -> xr.Dataset:
    import argopy

    argopy.set_options(mode=mode)
    fetcher = argopy.DataFetcher(mode=mode).region([lon_min, lon_max, lat_min, lat_max, min_depth, max_depth,
                                                    str(start_date), str(end_date)])
    return fetcher.to_xarray()


def fetch_argo_profiles_chunked(lon_min, lon_max, lat_min, lat_max, start_date, end_date, freq="MS",
                                on_chunk_error="warn", **kw) -> xr.Dataset:
    """Monthly chunks with resume-on-error: a decade-long single request fails in practice."""
    edges = pd.date_range(start_date, end_date, freq=freq)
    if len(edges) == 0 or edges[0] > pd.Timestamp(start_date):
        edges = edges.insert(0, pd.Timestamp(start_date))
    if edges[-1] < pd.Timestamp(end_date):
        edges = edges.append(pd.DatetimeIndex([pd.Timestamp(end_date)]))
    chunks = []
    for t0, t1 in zip(edges[:-1], edges[1:], strict=True):
        try:
            chunks.append(fetch_argo_profiles(lon_min, lon_max, lat_min, lat_max, t0.date(), t1.date(), **kw))
            print(f"[argo] {t0.date()}..{t1.date()}: {chunks[-1].sizes.get('N_POINTS', 0)} points")
        except Exception as exc:  # noqa: BLE001
            if on_chunk_error == "raise":
                raise
            print(f"[argo] WARNING chunk {t0.date()}..{t1.date()} failed ({exc}); skipped")
    if not chunks:
        raise RuntimeError("no ARGO chunk could be fetched")
    return xr.concat(chunks, dim="N_POINTS")


def _decode_qc(arr) -> np.ndarray:
    out = np.full(np.shape(arr), 9, dtype=np.int8).ravel()
    for i, v in enumerate(np.asarray(arr).ravel()):
        v = v.decode("ascii", "ignore") if isinstance(v, bytes) else str(v)
        v = v.strip()
        if v.isdigit():
            out[i] = int(v)
    return out.reshape(np.shape(arr))


def pointcloud_from_gdac_file(ds: xr.Dataset, value_vars=VALUE_VARS) -> xr.Dataset:
    """Flatten one GDAC profile file (N_PROF, N_LEVELS) into the argopy N_POINTS layout."""
    n_prof = ds.sizes.get("N_PROF", 1)
    n_lev = ds.sizes.get("N_LEVELS", ds.sizes.get("N_LEVEL", 1))

    def per_level(name):
        return np.asarray(ds[name].values).reshape(n_prof, -1)[:, :n_lev].reshape(-1) if name in ds else None

    def per_prof(name):
        return np.repeat(np.asarray(ds[name].values).reshape(n_prof), n_lev) if name in ds else None

    data = {"PRES": per_level("PRES"), "LATITUDE": per_prof("LATITUDE"), "LONGITUDE": per_prof("LONGITUDE"),
            "TIME": per_prof("JULD"), "PRES_QC": _decode_qc(per_level("PRES_QC")),
            "POSITION_QC": _decode_qc(per_prof("POSITION_QC")), "JULD_QC": _decode_qc(per_prof("JULD_QC")),
            "PLATFORM_NUMBER": per_prof("PLATFORM_NUMBER") if "PLATFORM_NUMBER" in ds else np.repeat("unknown", n_prof * n_lev),
            "CYCLE_NUMBER": per_prof("CYCLE_NUMBER") if "CYCLE_NUMBER" in ds else np.repeat(np.arange(n_prof), n_lev)}
    for v in value_vars:
        if v in ds:
            data[v] = per_level(v)
            data[f"{v}_QC"] = _decode_qc(per_level(f"{v}_QC"))
    return xr.Dataset({k: ("N_POINTS", v) for k, v in data.items() if v is not None})


def fetch_argo_profiles_local(gdac_dir, lon_min, lon_max, lat_min, lat_max, start_date, end_date,
                              file_glob="**/*_prof.nc", **_) -> xr.Dataset:
    files = sorted(Path(gdac_dir).glob(file_glob))
    if not files:
        raise RuntimeError(f"no GDAC profile files under {gdac_dir} ({file_glob})")
    t0, t1 = pd.Timestamp(start_date), pd.Timestamp(end_date)
    clouds = []
    for f in files:
        try:
            with xr.open_dataset(f) as ds:
                pc = pointcloud_from_gdac_file(ds)
        except Exception as exc:  # noqa: BLE001
            print(f"[argo] WARNING {f}: {exc}")
            continue
        t = pd.to_datetime(pc["TIME"].values)
        keep = ((pc.LATITUDE.values >= lat_min) & (pc.LATITUDE.values <= lat_max) & (pc.LONGITUDE.values >= lon_min)
                & (pc.LONGITUDE.values <= lon_max) & (t >= t0) & (t <= t1))
        if keep.any():
            clouds.append(pc.isel(N_POINTS=keep))
    return xr.concat(clouds, dim="N_POINTS") if clouds else xr.Dataset()


# ----------------------------------------------------------------------------- QC
def _qc_int(da: xr.DataArray) -> np.ndarray:
    v = da.values
    if v.dtype.kind in "SUO":
        return _decode_qc(v).astype(int)
    return v.astype(int)


def _profile_key(ds: xr.Dataset) -> np.ndarray:
    return ds["PLATFORM_NUMBER"].astype(str).values.astype(object) + "_" + ds["CYCLE_NUMBER"].astype(str).values.astype(object)


def sort_pointcloud(ds: xr.Dataset) -> xr.Dataset:
    """(profile, cycle, pressure) ordering required by the inversion test (argopy does not guarantee it)."""
    order = np.lexsort((ds["PRES"].values, ds["CYCLE_NUMBER"].values.astype(str), ds["PLATFORM_NUMBER"].values.astype(str)))
    return ds.isel(N_POINTS=order)


def filter_by_qc(ds: xr.Dataset, qc_vars, good=GOOD_FLAGS) -> xr.Dataset:
    keep = np.ones(ds.sizes["N_POINTS"], bool)
    for q in qc_vars:
        if q in ds:
            keep &= np.isin(_qc_int(ds[q]), good)
    return ds.isel(N_POINTS=keep)


def reject_pressure_inversions(ds: xr.Dataset) -> xr.Dataset:
    key, pres = _profile_key(ds), ds["PRES"].values
    keep = np.ones(ds.sizes["N_POINTS"], bool)
    keep[1:] &= ~((key[1:] == key[:-1]) & (pres[1:] <= pres[:-1]))
    return ds.isel(N_POINTS=keep)


def reject_spikes(ds: xr.Dataset, value_var: str, threshold: float) -> xr.Dataset:
    key, v = _profile_key(ds), ds[value_var].values
    spike = np.zeros(ds.sizes["N_POINTS"], bool)
    i = np.arange(1, len(v) - 1)
    same = (key[i - 1] == key[i]) & (key[i + 1] == key[i])
    spike[i] = same & (np.abs(v[i] - (v[i - 1] + v[i + 1]) / 2) > threshold)
    return ds.isel(N_POINTS=~spike)


def apply_standard_qc(ds: xr.Dataset, value_vars=VALUE_VARS, spike_thresholds: dict | None = None) -> xr.Dataset:
    ds = sort_pointcloud(ds)
    ds = filter_by_qc(ds, ["POSITION_QC", "JULD_QC", "PRES_QC"] + [f"{v}_QC" for v in value_vars])
    ds = reject_pressure_inversions(ds)
    for v, thr in (spike_thresholds or {}).items():
        ds = reject_spikes(ds, v, thr)
    return ds


# ----------------------------------------------------------------------------- vertical interpolation / coverage
def interp_profile(pres, values, targets) -> np.ndarray:
    pres, values = np.asarray(pres, float), np.asarray(values, float)
    ok = np.isfinite(pres) & np.isfinite(values)
    pres, values = pres[ok], values[ok]
    if pres.size < 2:
        return np.full(len(targets), np.nan)
    order = np.argsort(pres)
    pres, idx = np.unique(pres[order], return_index=True)
    return np.interp(targets, pres, values[order][idx], left=np.nan, right=np.nan)


def coverage_table(ds: xr.Dataset, depth_values: list[float], depth_indices: list[int], value_var: str = "TEMP",
                   time_var: str = "TIME") -> pd.DataFrame:
    """One row per profile: ``time, lat, lon, profile_id, d<ii>`` (1 if the profile covers level ii, else 0)
    plus ``<value_var>_d<ii>`` (interpolated real value, kept for OSE validation)."""
    cols = ["PRES", value_var, "LATITUDE", "LONGITUDE", time_var, "PLATFORM_NUMBER", "CYCLE_NUMBER"]
    df = ds[[c for c in cols if c in ds]].to_dataframe().reset_index(drop=True)
    rows = []
    for (plat, cyc), g in df.groupby(["PLATFORM_NUMBER", "CYCLE_NUMBER"]):
        vals = interp_profile(g["PRES"], g[value_var], depth_values)
        row = {"profile_id": f"{plat}_{cyc}", "time": g[time_var].iloc[0], "lat": float(g["LATITUDE"].iloc[0]),
               "lon": float(g["LONGITUDE"].iloc[0])}
        for i, v in zip(depth_indices, vals, strict=True):
            row[f"d{i:02d}"] = float(np.isfinite(v))
            row[f"{value_var}_d{i:02d}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def depth_values_from_truth(truth: str | Path, depth_indices: list[int]) -> list[float]:
    ds = xr.open_dataset(truth)
    return [float(ds["depth"].values[i]) for i in depth_indices]
