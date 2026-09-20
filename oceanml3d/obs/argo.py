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
INDEX_CHUNK = 500_000


def find_index(gdac_dir, verbose: bool = True) -> Path | None:
    """The global profile index, beside the GDAC root or one level above it.

    Verbose on purpose: the fallback (scanning every float directory) costs hours, so the log has to
    say which of the two happened, and when no index is found, where it looked.
    """
    root = Path(gdac_dir)
    tried = []
    for base in (root, root.parent):
        for n in INDEX_NAMES:
            tried.append(base / n)
            if tried[-1].exists():
                if verbose:
                    print(f"[argo] index: {tried[-1]} ({tried[-1].stat().st_size / 1e6:.0f} MB)", flush=True)
                return tried[-1]
    if verbose:
        print("[argo] no index found; looked for " + ", ".join(str(t) for t in tried), flush=True)
    return None


def prof_files_from_index(gdac_dir, lon_min, lon_max, lat_min, lat_max, start_date, end_date,
                          index: str | Path | None = None, chunksize: int = INDEX_CHUNK) -> list[Path] | None:
    """The ``<dac>/<wmo>/<wmo>_prof.nc`` files of the floats with a profile in the box and period.

    Every GDAC ships ``ar_index_global_prof.txt``: one line per profile with its date and position.
    Filtering it first means opening the few hundred floats that crossed the box instead of all
    ~20 000 in the archive -- minutes instead of hours on a shared filesystem. Returns ``None`` when
    the mirror has no index, so the caller can fall back to scanning.

    Read in chunks, and reporting each one. Whole-file, this is three million rows of Python strings
    (a couple of gigabytes of objects) and one silent call: a job killed on walltime could not say
    whether the index read was the slow part or had never even started.
    """
    root = Path(gdac_dir)
    index = Path(index) if index else find_index(root)
    if index is None or not index.exists():
        return None
    t0, t1 = pd.Timestamp(start_date), pd.Timestamp(end_date)
    started = time.monotonic()
    print(f"[argo] reading the index in chunks of {chunksize} rows (a global index holds 2-3 million "
          f"profiles: seconds on a local disk, minutes on a network mount)", flush=True)
    rows = kept = 0
    floats: set[str] = set()
    first = last = None
    for chunk in pd.read_csv(index, comment="#", usecols=["file", "date", "latitude", "longitude"],
                             dtype={"file": str, "date": str}, chunksize=chunksize):
        t = pd.to_datetime(chunk["date"].str.slice(0, 14), format="%Y%m%d%H%M%S", errors="coerce")
        keep = (chunk.latitude.between(lat_min, lat_max) & chunk.longitude.between(lon_min, lon_max)
                & (t >= t0) & (t <= t1))
        rows, kept = rows + len(chunk), kept + int(keep.sum())
        floats.update("/".join(f.split("/")[:2]) for f in chunk.loc[keep, "file"])
        lo, hi = t.min(), t.max()
        first = lo if first is None or (pd.notna(lo) and lo < first) else first
        last = hi if last is None or (pd.notna(hi) and hi > last) else last
        print(f"[argo]   {rows:>9} index rows, {kept} in box/period, {len(floats)} floats, "
              f"{time.monotonic() - started:.0f} s", flush=True)
    # aoml/1900722/profiles/D1900722_061.nc -> aoml/1900722/1900722_prof.nc. Index paths are relative
    # to the `dac/` directory, which sits next to the index in a standard GDAC tree.
    base = index.parent / "dac" if (index.parent / "dac").is_dir() else root
    print(f"[argo] index covers {first} .. {last}; {kept} profiles in the box and period from "
          f"{len(floats)} floats, read in {time.monotonic() - started:.0f} s", flush=True)
    if last is not None and last < t1:
        print(f"[argo] NOTE the index stops at {last}, before the requested {t1.date()}: the mirror "
              f"may be behind, or this is all there is", flush=True)
    return [base / f / f"{f.split('/')[1]}_prof.nc" for f in sorted(floats)]


def _scan_prof_files(gdac_dir) -> list[Path]:
    """Fallback without an index: ``<dac>/<wmo>/<wmo>_prof.nc`` at a fixed depth.

    Not ``**/*_prof.nc``: a recursive glob descends into every ``profiles/`` directory, which on a full
    GDAC holds millions of per-cycle files -- hours of directory listing on Lustre before the first
    open. The per-float files sit exactly two levels under ``dac/``.
    """
    root = Path(gdac_dir)
    base = root / "dac" if (root / "dac").is_dir() else root
    return sorted(base.glob("*/*/*_prof.nc"))


def gdac_prof_files(gdac_dir, lon_min, lon_max, lat_min, lat_max, start_date, end_date,
                    index: str | Path | None = None) -> list[Path]:
    """The float files to open, from the index when there is one, by scanning when there is not.

    Reports the mirror before touching it: an unbound bind mount looks exactly like an empty GDAC,
    and the difference decides whether the run is broken or merely slow.
    """
    root = Path(gdac_dir)
    print(f"[argo] GDAC {root}: {'present' if root.is_dir() else 'ABSENT -- bound into the container?'}",
          flush=True)
    if root.is_dir():
        names = sorted(p.name for p in list(root.iterdir())[:12])
        print(f"[argo] top level: {', '.join(names) or '(empty)'}", flush=True)
    files = prof_files_from_index(root, lon_min, lon_max, lat_min, lat_max, start_date, end_date, index)
    if files is None:
        print(f"[argo] WARNING: no ar_index_global_prof.txt in {root} or its parent (set `index:` in "
              f"the recipe if it lives elsewhere); scanning <dac>/<wmo>/<wmo>_prof.nc instead -- every float "
              f"of the archive is opened, expect a long run", flush=True)
        t = time.monotonic()
        files = _scan_prof_files(root)
        print(f"[argo] {len(files)} float files found by scanning in {time.monotonic() - t:.0f} s", flush=True)
    if not files:
        raise RuntimeError(f"no GDAC profile file for the box/period under {root}")
    head = files[: min(20, len(files))]
    if not any(f.exists() for f in head):
        raise RuntimeError(
            f"none of the first {len(head)} files the index points at exist, e.g. {files[0]}. Index "
            f"paths are relative to the GDAC's `dac/` directory: set `gdac_dir` to the directory that "
            f"contains `dac/` (or `index:` to the index beside it).")
    return files


class FileProgress:
    """Per-file timing for a loop over thousands of files on a shared filesystem.

    Written because the failure it exists for was silence: an ARGO job killed on walltime left one
    header line in the log, with no way to tell whether the index read, the first open or the two
    thousandth file was the slow part. So: the first few files by name, then a rate and an estimate,
    and any single file that takes more than ``slow_s`` called out on its own.
    """

    def __init__(self, total: int, every: int = 10, slow_s: float = 20.0, announce_first: int = 3,
                 time_budget_s: float | None = None):
        self.total, self.every, self.slow_s = total, every, slow_s
        self.announce_first, self.budget = announce_first, time_budget_s
        self.start = self.t_file = time.monotonic()
        self.kept = self.failed = self.done = 0

    def before(self, k: int, f) -> None:
        self.t_file = time.monotonic()
        if k <= self.announce_first:
            print(f"[argo] opening {k}/{self.total}: {f}", flush=True)

    def after(self, k: int, f, kept: int = 0, failed: bool = False) -> None:
        dt = time.monotonic() - self.t_file
        self.done, self.kept, self.failed = k, self.kept + kept, self.failed + int(failed)
        if dt > self.slow_s:
            print(f"[argo] slow file, {dt:.0f} s: {f}", flush=True)
        if k % self.every == 0 or k == self.total or k <= self.announce_first:
            elapsed = time.monotonic() - self.start
            rate = elapsed / max(k, 1)
            print(f"[argo] {k}/{self.total} files, {self.kept} profiles kept, {self.failed} failed, "
                  f"{elapsed:.0f} s, {rate:.2f} s/file, ~{rate * (self.total - k) / 60:.0f} min left",
                  flush=True)

    def over_budget(self) -> bool:
        return self.budget is not None and time.monotonic() - self.start > self.budget


class WalltimeBudgetExceeded(RuntimeError):
    """``time_budget_s`` ran out. With a cache, what is done is on disk and a re-run continues."""


def _profiles_in_box(ds: xr.Dataset, lon_min, lon_max, lat_min, lat_max, t0, t1) -> np.ndarray:
    lat = np.asarray(ds["LATITUDE"].values, float).reshape(-1)
    lon = np.asarray(ds["LONGITUDE"].values, float).reshape(-1)
    t = pd.to_datetime(np.asarray(ds["JULD"].values).reshape(-1))
    return ((lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
            & np.asarray(t >= t0) & np.asarray(t <= t1))


def fetch_argo_profiles_local(gdac_dir, lon_min, lon_max, lat_min, lat_max, start_date, end_date,
                              index: str | Path | None = None, value_vars=VALUE_VARS,
                              progress_every: int = 10, **_) -> xr.Dataset:
    """Read a local GDAC mirror (``/home/ref-argo/gdac`` on Datarmor) instead of downloading.

    Holds every profile of the box in memory at once. :func:`coverage_table_local` is what the
    preparation uses: same reading, one row per profile kept instead, and resumable.
    """
    files = gdac_prof_files(gdac_dir, lon_min, lon_max, lat_min, lat_max, start_date, end_date, index)
    t0, t1 = pd.Timestamp(start_date), pd.Timestamp(end_date)
    clouds, prog = [], FileProgress(len(files), progress_every)
    for k, f in enumerate(files, 1):
        prog.before(k, f)
        try:
            with xr.open_dataset(f) as ds:
                keep = _profiles_in_box(ds, lon_min, lon_max, lat_min, lat_max, t0, t1)
                if keep.any():
                    clouds.append(pointcloud_from_gdac_file(ds, value_vars, keep_prof=keep))
            prog.after(k, f, int(keep.sum()))
        except Exception as exc:  # noqa: BLE001
            prog.after(k, f, failed=True)
            if prog.failed <= 10:
                print(f"[argo] WARNING {f}: {type(exc).__name__}: {exc}", flush=True)
    if prog.failed and not clouds:
        raise RuntimeError(f"all {prog.failed} GDAC files failed to read (first errors above)")
    return xr.concat(clouds, dim="N_POINTS") if clouds else xr.Dataset()


def coverage_table_local(gdac_dir, bbox, start_date, end_date, depth_values: list[float],
                         depth_indices: list[int], value_var: str = "TEMP",
                         spike_thresholds: dict | None = None, index: str | Path | None = None,
                         value_vars=VALUE_VARS, cache_dir: str | Path | None = None,
                         progress_every: int = 10, time_budget_s: float | None = None) -> pd.DataFrame:
    """Float by float: read, QC, one row per profile — cached on disk, so a killed job resumes.

    Streaming rather than "read the whole box, then QC it, then tabulate": every step of
    :func:`apply_standard_qc` works inside one profile (the sort is by platform, cycle and pressure;
    the filters are pointwise; inversions and spikes look at neighbours of the same profile), and
    :func:`coverage_table` groups by profile, so per float gives the same table as all at once. What
    it buys is that the work is finished float by float. ``cache_dir`` holds one small CSV per float
    plus a marker, so resubmitting the same job continues instead of starting the four hours again.
    """
    lon_min, lon_max, lat_min, lat_max = bbox
    files = gdac_prof_files(gdac_dir, lon_min, lon_max, lat_min, lat_max, start_date, end_date, index)
    t0, t1 = pd.Timestamp(start_date), pd.Timestamp(end_date)
    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        print(f"[argo] cache {cache}: {len(list(cache.glob('*.done')))} floats already done", flush=True)
    else:
        print("[argo] no `cache_dir:` in the recipe: a job killed on walltime starts over", flush=True)

    prog = FileProgress(len(files), progress_every, time_budget_s=time_budget_s)
    tables, reused, stopped = [], 0, 0
    for k, f in enumerate(files, 1):
        wmo = f.parent.name
        done, csv = (cache / f"{wmo}.done", cache / f"{wmo}.csv") if cache else (None, None)
        if done is not None and done.exists():
            reused += 1
            if csv.exists():
                # datetime64[ns], as the in-memory path produces: pandas reads a CSV date back at
                # second resolution, and a table half of whose rows came from the cache would carry
                # two different time dtypes depending on where the previous run stopped.
                rows = pd.read_csv(csv, parse_dates=["time"])
                tables.append(rows.astype({"time": "datetime64[ns]"}))
            continue
        prog.before(k, f)
        try:
            rows = pd.DataFrame()
            kept = 0
            with xr.open_dataset(f) as ds:
                keep = _profiles_in_box(ds, lon_min, lon_max, lat_min, lat_max, t0, t1)
                kept = int(keep.sum())
                pc = pointcloud_from_gdac_file(ds, value_vars, keep_prof=keep) if kept else None
            if pc is not None and pc.sizes.get("N_POINTS", 0):
                pc = apply_standard_qc(pc, value_vars, spike_thresholds)
                if pc.sizes.get("N_POINTS", 0):
                    rows = coverage_table(pc, depth_values, depth_indices, value_var)
            if len(rows):
                tables.append(rows)
                if csv is not None:
                    rows.to_csv(csv, index=False)
            if done is not None:
                done.touch()
            prog.after(k, f, kept)
        except Exception as exc:  # noqa: BLE001
            prog.after(k, f, failed=True)
            if prog.failed <= 10:
                print(f"[argo] WARNING {f}: {type(exc).__name__}: {exc}", flush=True)
        if prog.over_budget() and k < len(files):
            stopped = k
            break

    if stopped:
        raise WalltimeBudgetExceeded(
            f"stopped after {stopped}/{len(files)} floats, {time_budget_s:.0f} s budget reached. "
            + (f"{len(list(cache.glob('*.done')))} floats are cached in {cache}: resubmit the same job "
               f"and it continues from there." if cache
               else "Set `cache_dir:` in the recipe so a re-run resumes instead of restarting."))
    if prog.failed and not tables:
        raise RuntimeError(f"all {prog.failed} GDAC files failed to read (first errors above)")
    print(f"[argo] {len(files)} floats ({reused} from the cache), {prog.failed} failed", flush=True)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


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
    """The metres behind ``depth_indices``, read from the prepared truth.

    Explicit about Zarr: the truth is a Zarr store since the preparation stopped writing NetCDF
    intermediates, and this is read *before* the profiles are, so a wrong path fails in a second
    instead of after four hours of reading.
    """
    path = Path(truth)
    if not path.exists():
        raise FileNotFoundError(f"the truth {path} does not exist: `truth:` in the recipe must point at "
                                f"the merged GLORYS store, which `jobs/prepare.sh concat` writes")
    ds = xr.open_zarr(path) if path.suffix == ".zarr" else xr.open_dataset(path)
    depth = [float(ds["depth"].values[i]) for i in depth_indices]
    print(f"[argo] depth levels from {path.name}: {depth[0]:.1f} .. {depth[-1]:.1f} m "
          f"({len(depth)} levels)", flush=True)
    return depth


# ----------------------------------------------------------------------------- acceptance summary
def summarise_coverage(table: pd.DataFrame, first: str | None = None, last: str | None = None,
                       min_per_year: int = 100) -> list[str]:
    """What the table actually contains: span, floats, profiles per year, coverage per level.

    A count alone says nothing. 8 640 profiles over a decade and 8 640 concentrated in 2018 are the
    same number, and the second would train a model on a year of in-situ data and validate it on
    none. The per-level line is the other half: a float that stops at 100 m contributes nothing at
    186 m, and `d25` near zero means the deep targets have no observation behind them.
    """
    out = []
    if table.empty:
        return ["[argo] the table is empty"]
    t = pd.to_datetime(table["time"])
    floats = table["profile_id"].astype(str).str.split("_").str[0].nunique()
    out.append(f"[argo] table: {len(table)} profiles, {floats} floats, "
               f"{t.min().date()} .. {t.max().date()}")
    per_year = t.dt.year.value_counts().sort_index()
    out.append("[argo] per year: " + "  ".join(f"{y} {n}" for y, n in per_year.items()))

    levels = sorted(c for c in table.columns if len(c) == 3 and c[0] == "d" and c[1:].isdigit())
    if levels:
        cov = {c: float(table[c].mean()) for c in levels}
        out.append("[argo] reaching each level: "
                   + "  ".join(f"{c} {100 * v:.0f}%" for c, v in cov.items()))
        shallow = [c for c, v in cov.items() if v < 0.5]
        if shallow:
            out.append(f"[argo] WARNING fewer than half the profiles reach {', '.join(shallow)}: the "
                       f"virtual ARGO input will be nearly empty at those depths")

    # Holes are what a total hides. A year inside the requested window with almost nothing in it is
    # either a real gap in the array or a filter that is cutting too much -- both worth knowing
    # before the model is trained on it.
    if first and last:
        want = range(pd.Timestamp(first).year, pd.Timestamp(last).year + 1)
        thin = [y for y in want if per_year.get(y, 0) < min_per_year]
        # The last year of a window ending in January is legitimately short; judge it pro rata.
        edge = pd.Timestamp(last)
        if edge.year in thin and per_year.get(edge.year, 0) >= min_per_year * edge.dayofyear / 365:
            thin.remove(edge.year)
        if thin:
            out.append(f"[argo] WARNING fewer than {min_per_year} profiles in "
                       + ", ".join(str(y) for y in thin)
                       + " -- the requested window is not evenly covered")
    return out
