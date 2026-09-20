"""ARGO profiles: download (argopy or local GDAC), QC, vertical interpolation, and the
per-profile *coverage table* consumed by :mod:`oceanml3d.obs.argo_virtual`.

Port of NOSC ``contrib/argo/{download,qc,vertical_interp}.py``. Heavy dependencies
(argopy) are imported lazily; the QC / interpolation part is pure numpy/pandas and tested.
"""
from __future__ import annotations

import time
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
    """QC flags ('0'..'9', blank = missing) to int8, vectorised; anything but a digit becomes 9."""
    a = np.asarray(arr)
    if a.dtype.kind in "iuf":
        return np.where(np.isfinite(a.astype(float)), a, 9).astype(np.int8)
    if a.dtype.kind == "U":
        a = np.char.encode(a, "ascii", "ignore")
    elif a.dtype.kind == "O":
        a = np.array([v if isinstance(v, bytes) else str(v).encode("ascii", "ignore") for v in a.ravel()],
                     dtype="S").reshape(a.shape)
    if a.dtype.itemsize != 1:
        a = np.char.strip(a).astype("S1")          # '' stays '' -> byte 0 -> 9 below
    code = a.view(np.uint8).astype(np.int16) - ord("0")
    return np.where((code >= 0) & (code <= 9), code, 9).astype(np.int8)


def _qc_levels(da: xr.DataArray, n_prof: int, n_lev: int) -> np.ndarray:
    """A per-level QC variable as ``(n_prof, n_lev)`` single characters, whatever xarray made of it.

    In a real GDAC file ``PRES_QC`` is ``char(N_PROF, N_LEVELS)``, and xarray's character decoding
    joins the last dimension: it comes back as ``(N_PROF,)`` strings of length ``N_LEVELS``
    (``b'1114 '``), not as ``(N_PROF, N_LEVELS)`` flags. Reshaping that like a numeric variable gave
    ``n_prof`` values instead of ``n_prof x n_lev`` and every file failed to flatten.
    """
    a = np.asarray(da.values)
    if a.shape == (n_prof, n_lev):
        return a
    if a.dtype.kind == "S" and a.shape == (n_prof,):
        raw = np.frombuffer(a.astype(f"S{max(a.dtype.itemsize, 1)}").tobytes(), dtype=np.uint8)
        raw = raw.reshape(n_prof, -1)
        out = np.full((n_prof, n_lev), ord(" "), np.uint8)
        w = min(n_lev, raw.shape[1])
        out[:, :w] = raw[:, :w]
        out[out == 0] = ord(" ")
        return out.view("S1")
    return a.reshape(n_prof, -1)[:, :n_lev]


def _as_text(a: np.ndarray) -> np.ndarray:
    return np.char.strip(np.char.decode(a, "ascii", "ignore")) if a.dtype.kind == "S" else a.astype(str)


def pointcloud_from_gdac_file(ds: xr.Dataset, value_vars=VALUE_VARS, keep_prof: np.ndarray | None = None,
                              use_adjusted: bool = True) -> xr.Dataset:
    """Flatten one GDAC profile file (N_PROF, N_LEVELS) into the argopy N_POINTS layout.

    ``keep_prof`` (boolean over N_PROF) drops profiles *before* flattening: a ``<wmo>_prof.nc`` holds
    every cycle of the float's life, of which a handful cross the box. ``use_adjusted``: where
    ``DATA_MODE`` is ``A`` or ``D``, the ``*_ADJUSTED`` values and flags replace the raw ones -- what
    argopy's "standard" mode returns, and the only calibrated values for delayed-mode profiles.
    """
    if keep_prof is not None:
        ds = ds.isel(N_PROF=np.flatnonzero(keep_prof))
    n_prof = ds.sizes.get("N_PROF", 1)
    n_lev = ds.sizes.get("N_LEVELS", ds.sizes.get("N_LEVEL", 1))
    if n_prof == 0:
        return xr.Dataset()
    mode = _as_text(np.asarray(ds["DATA_MODE"].values)).reshape(n_prof) if "DATA_MODE" in ds else np.full(n_prof, "R")
    adjusted = np.isin(mode, ["A", "D"]) if use_adjusted else np.zeros(n_prof, bool)

    def level_values(name):
        raw = np.asarray(ds[name].values, float).reshape(n_prof, -1)[:, :n_lev]
        adj = f"{name}_ADJUSTED"
        if adjusted.any() and adj in ds:
            raw = np.where(adjusted[:, None], np.asarray(ds[adj].values, float).reshape(n_prof, -1)[:, :n_lev], raw)
        return raw.reshape(-1)

    def level_qc(name):
        raw = _decode_qc(_qc_levels(ds[f"{name}_QC"], n_prof, n_lev)) if f"{name}_QC" in ds \
            else np.full((n_prof, n_lev), 9, np.int8)
        adj = f"{name}_ADJUSTED_QC"
        if adjusted.any() and adj in ds:
            raw = np.where(adjusted[:, None], _decode_qc(_qc_levels(ds[adj], n_prof, n_lev)), raw)
        return raw.reshape(-1)

    def per_prof(name, default=None):
        if name not in ds:
            return None if default is None else np.repeat(default, n_lev)
        return np.repeat(np.asarray(ds[name].values).reshape(n_prof), n_lev)

    platform = (_as_text(np.asarray(ds["PLATFORM_NUMBER"].values)).reshape(n_prof) if "PLATFORM_NUMBER" in ds
                else np.full(n_prof, "unknown"))
    data = {"PRES": level_values("PRES"), "PRES_QC": level_qc("PRES"),
            "LATITUDE": per_prof("LATITUDE"), "LONGITUDE": per_prof("LONGITUDE"), "TIME": per_prof("JULD"),
            "POSITION_QC": np.repeat(_decode_qc(np.asarray(ds["POSITION_QC"].values).reshape(n_prof)), n_lev)
            if "POSITION_QC" in ds else None,
            "JULD_QC": np.repeat(_decode_qc(np.asarray(ds["JULD_QC"].values).reshape(n_prof)), n_lev)
            if "JULD_QC" in ds else None,
            "PLATFORM_NUMBER": np.repeat(platform, n_lev),
            "CYCLE_NUMBER": per_prof("CYCLE_NUMBER", np.arange(n_prof)) if "CYCLE_NUMBER" in ds
            else np.repeat(np.arange(n_prof), n_lev)}
    for v in value_vars:
        if v in ds:
            data[v] = level_values(v)
            data[f"{v}_QC"] = level_qc(v)
    return xr.Dataset({k: ("N_POINTS", v) for k, v in data.items() if v is not None})


INDEX_NAMES = ("ar_index_global_prof.txt", "ar_index_global_prof.txt.gz")


def find_index(gdac_dir) -> Path | None:
    root = Path(gdac_dir)
    for base in (root, root.parent):
        for n in INDEX_NAMES:
            if (base / n).exists():
                return base / n
    return None


def prof_files_from_index(gdac_dir, lon_min, lon_max, lat_min, lat_max, start_date, end_date,
                          index: str | Path | None = None) -> list[Path] | None:
    """The ``<dac>/<wmo>/<wmo>_prof.nc`` files of the floats with a profile in the box and period.

    Every GDAC ships ``ar_index_global_prof.txt``: one line per profile with its date and position.
    Filtering it first means opening the few hundred floats that crossed the box instead of all
    ~20 000 in the archive -- minutes instead of hours on a shared filesystem. Returns ``None`` when
    the mirror has no index, so the caller can fall back to scanning.
    """
    root = Path(gdac_dir)
    index = Path(index) if index else find_index(root)
    if index is None or not index.exists():
        return None
    t0 = time.monotonic()
    df = pd.read_csv(index, comment="#", usecols=["file", "date", "latitude", "longitude"],
                     dtype={"file": str, "date": str})
    t = pd.to_datetime(df["date"].str.slice(0, 14), format="%Y%m%d%H%M%S", errors="coerce")
    keep = (df.latitude.between(lat_min, lat_max) & df.longitude.between(lon_min, lon_max)
            & (t >= pd.Timestamp(start_date)) & (t <= pd.Timestamp(end_date)))
    # aoml/1900722/profiles/D1900722_061.nc -> aoml/1900722/1900722_prof.nc. Index paths are relative
    # to the `dac/` directory, which sits next to the index in a standard GDAC tree.
    base = index.parent / "dac" if (index.parent / "dac").is_dir() else root
    floats = sorted({"/".join(f.split("/")[:2]) for f in df.loc[keep, "file"]})
    print(f"[argo] index {index} ({len(df)} profiles, read in {time.monotonic() - t0:.0f} s): "
          f"{int(keep.sum())} in box/period, from {len(floats)} floats")
    return [base / f / f"{f.split('/')[1]}_prof.nc" for f in floats]


def _scan_prof_files(gdac_dir) -> list[Path]:
    """Fallback without an index: ``<dac>/<wmo>/<wmo>_prof.nc`` at a fixed depth.

    Not ``**/*_prof.nc``: a recursive glob descends into every ``profiles/`` directory, which on a full
    GDAC holds millions of per-cycle files -- hours of directory listing on Lustre before the first
    open. The per-float files sit exactly two levels under ``dac/``.
    """
    root = Path(gdac_dir)
    base = root / "dac" if (root / "dac").is_dir() else root
    return sorted(base.glob("*/*/*_prof.nc"))


def fetch_argo_profiles_local(gdac_dir, lon_min, lon_max, lat_min, lat_max, start_date, end_date,
                              index: str | Path | None = None, value_vars=VALUE_VARS,
                              progress_every: int = 50, **_) -> xr.Dataset:
    """Read a local GDAC mirror (``/home/ref-argo/gdac`` on Datarmor) instead of downloading."""
    files = prof_files_from_index(gdac_dir, lon_min, lon_max, lat_min, lat_max, start_date, end_date, index)
    if files is None:
        print(f"[argo] WARNING: no ar_index_global_prof.txt in {gdac_dir} or its parent (set `index:` in "
              f"the recipe if it lives elsewhere); scanning <dac>/<wmo>/<wmo>_prof.nc instead -- every float "
              f"of the archive is opened, expect a long run", flush=True)
        files = _scan_prof_files(gdac_dir)
        print(f"[argo] {len(files)} float files found", flush=True)
    if not files:
        raise RuntimeError(f"no GDAC profile file for the box/period under {gdac_dir}")
    t0, t1 = pd.Timestamp(start_date), pd.Timestamp(end_date)
    clouds, failed, n_prof, start = [], 0, 0, time.monotonic()
    for k, f in enumerate(files, 1):
        try:
            with xr.open_dataset(f) as ds:
                lat = np.asarray(ds["LATITUDE"].values, float).reshape(-1)
                lon = np.asarray(ds["LONGITUDE"].values, float).reshape(-1)
                t = pd.to_datetime(np.asarray(ds["JULD"].values).reshape(-1))
                keep = ((lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
                        & np.asarray(t >= t0) & np.asarray(t <= t1))
                if keep.any():
                    n_prof += int(keep.sum())
                    clouds.append(pointcloud_from_gdac_file(ds, value_vars, keep_prof=keep))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            if failed <= 10:
                print(f"[argo] WARNING {f}: {type(exc).__name__}: {exc}", flush=True)
        if k % progress_every == 0 or k == len(files):
            print(f"[argo] {k}/{len(files)} files, {n_prof} profiles kept, {failed} failed, "
                  f"{time.monotonic() - start:.0f} s", flush=True)
    if failed and not clouds:
        raise RuntimeError(f"all {failed} GDAC files failed to read (first errors above)")
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
