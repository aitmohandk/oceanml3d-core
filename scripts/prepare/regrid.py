"""Regrid any gridded product onto the reference grid of a task (replaces the ~20
``interpolation_*_4th.py`` scripts of NOSC with one recipe-driven tool).

Recipe (YAML):
    input: /path/or/glob/*.nc
    output: /path/out_4th.nc
    variables: {adt: zos, ugos: ugos, vgos: vgos}   # rename on the fly
    grid: {lat: [-80, 90, 0.25], lon: [-180, 180, 0.25]}   # or  reference: /path/ref.nc
    time: ["2010-01-01", "2022-01-01"]
    domain: {lat: [32, 44], lon: [-66, -54]}           # subdomain, cut per file before any read
    resolution: 0.25                                   # regular grid at this step (deg) over `domain`
    max_gb: 100                                        # refuse larger uncompressed outputs
    method: linear                                     # linear | nearest | conservative (xesmf)
    keep_depth: true                                   # keep the depth axis (3D tasks)
    depth_indices: [0, 2, 4]                           # positions on the SOURCE depth axis
    chunks: {time: 30}                                 # only if you know you need it (see below)
    parallel: false                                    # open files from dask threads -- see below
    dask_scheduler: synchronous                        # synchronous | threads | processes
    zarr_chunks: {time: 32}                            # only when `output` ends in .zarr

An ``output`` ending in ``.zarr`` writes a Zarr store instead of a NetCDF file. That is the better
format for **intermediate** data: chunks are independent objects with no C-library global state, so
the thread-safety problem that forces ``parallel: false`` on the netCDF path does not arise, and a
recipe reading these back can set ``parallel: true`` and ``dask_scheduler: threads``. The catalog and
``data/open.py`` already accept ``.zarr`` sources, so nothing downstream changes.

Keep the two ends as they are: the GLORYS mirror is read-only and not ours to convert, and the
exported product stays NetCDF because that is the contract with ``oceanml3d-eval``.

``resolution`` is NOSC's ``--target-res``: a regular grid at that spacing, anchored on the ``domain``
bounds (``arange(lo, hi + res/2, res)``), bilinear by default. It depends only on the domain and the
step -- not on the source -- so every product prepared with the same pair lands on exactly the same
grid, which ``data/open.py`` then requires. Recipes write ``resolution: ${OCEANML3D_TARGET_RES}``:
unset, the placeholder stays literal and the native grid is kept. The value is recorded in the
output's attributes, and an existing output at another resolution is refused rather than skipped.

CAREFUL WITH ``depth_indices``. It selects *positions on the source file's depth axis*, and the
positions it leaves are what the data config's own ``depth_index`` then addresses. Change this list
and every ``depth_index`` in ``config/data/*.yaml`` silently means a different depth. If in doubt,
leave it out: store every level the download produced and let the task select.
"""
from __future__ import annotations

import argparse
import glob
import inspect
import os
import shutil
import sys
from pathlib import Path

import dask
import numpy as np
import xarray as xr
import yaml

from oceanml3d.data.open import normalise_dims

# In a batch job stdout is a pipe, not a tty, so Python buffers it whole and nothing is flushed until
# exit -- if the job is killed on walltime the log looks empty and you cannot tell how far it got.
# Line-buffer so the progress below lands in the .o/.out file as it happens. (Same as `python -u`,
# kept here so it holds however the script is launched.)
# Lustre, which is what every one of these centres runs, does not implement the POSIX locking HDF5
# reaches for. Left alone it produces either an error about file locking or a hang on the first open.
# Harmless for us: these files are read-only inputs.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass


def target_grid(recipe: dict) -> tuple[np.ndarray, np.ndarray]:
    if "reference" in recipe:
        ref = normalise_dims(xr.open_dataset(recipe["reference"]))
        return ref.lat.values, ref.lon.values
    g = recipe["grid"]
    lat = np.arange(g["lat"][0], g["lat"][1] + 1e-9, g["lat"][2])
    lon = np.arange(g["lon"][0], g["lon"][1] + 1e-9, g["lon"][2])
    return lat, lon


def _lon_to(convention_max: float, x: float) -> float:
    """Express longitude ``x`` in the convention of a coordinate whose maximum is ``convention_max``."""
    if convention_max > 180:
        return x % 360
    return ((x + 180) % 360) - 180 if x > 180 else x


def _select_domain(ds: xr.Dataset, domain: dict) -> xr.Dataset:
    """Cut ``ds`` to ``domain`` = ``{lat: [lo, hi], lon: [lo, hi]}``, inclusive bounds.

    By position, from a mask, rather than ``sel(lat=slice(...))``: a label slice silently returns an
    *empty* array when the stored axis is descending, and a longitude range given in -180..180
    against a 0..360 file (or the reverse) matches nothing at all. Both conventions occur across the
    products this script reads. A box that crosses the seam of the file's convention (e.g. -10..10 on
    a 0..360 axis) is returned in -180..180, sorted, so it stays contiguous.
    """
    for dim in ("lat", "lon"):
        if dim not in domain:
            continue
        if dim not in ds.dims:
            raise KeyError(f"recipe domain has {dim!r} but the data has no {dim!r} dimension "
                           f"(dims: {dict(ds.sizes)})")
        lo, hi = (float(v) for v in domain[dim])
        coord = ds[dim].values
        if dim == "lon" and hi - lo >= 360:
            continue
        if dim == "lon":
            top = float(np.nanmax(coord))
            lo, hi = _lon_to(top, lo), _lon_to(top, hi)
            mask = (coord >= lo) & (coord <= hi) if lo <= hi else (coord >= lo) | (coord <= hi)
        else:
            mask = (coord >= min(lo, hi)) & (coord <= max(lo, hi))
        idx = np.nonzero(mask)[0]
        if idx.size == 0:
            raise ValueError(f"domain {dim}={domain[dim]} selects nothing: the data spans "
                             f"{float(np.nanmin(coord))}..{float(np.nanmax(coord))}")
        ds = ds.isel({dim: idx})
        if dim == "lon" and lo > hi:
            ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby("lon")
    return ds


def resolution_of(recipe: dict) -> float | None:
    """The recipe's target step in degrees, or None for the native grid.

    ``${OCEANML3D_TARGET_RES}`` left unexpanded (variable unset), empty, ``native`` and ``null`` all
    mean native. Anything else must be a positive number.
    """
    raw = recipe.get("resolution")
    if raw is None or (isinstance(raw, str) and (raw.strip() in ("", "native", "none", "null")
                                                 or raw.strip().startswith("$"))):
        return None
    res = float(raw)
    if res <= 0:
        raise ValueError(f"resolution must be a positive step in degrees, got {raw!r}")
    if not recipe.get("domain") or not {"lat", "lon"} <= set(recipe["domain"]):
        raise ValueError("`resolution` needs `domain: {lat: [lo, hi], lon: [lo, hi]}`: the target grid "
                         "is anchored on those bounds so every product lands on the same one")
    if "grid" in recipe or "reference" in recipe:
        raise ValueError("give either `resolution` (with `domain`) or `grid`/`reference`, not both")
    return res


def domain_grid(domain: dict, res: float) -> tuple[np.ndarray, np.ndarray]:
    """Regular grid at ``res`` anchored on the lower domain bounds -- NOSC's ``target_grid``."""
    (la0, la1), (lo0, lo1) = sorted(map(float, domain["lat"])), map(float, domain["lon"])
    lat = np.arange(la0, la1 + 0.5 * res, res)
    lon = np.arange(lo0, lo1 + 0.5 * res, res)
    return np.round(lat, 10), np.round(lon, 10)


def _to_resolution(ds: xr.Dataset, domain: dict, res: float, method: str) -> xr.Dataset:
    """Interpolate a domain-cut dataset onto ``domain_grid(domain, res)``.

    Longitudes are first put in the convention the domain is written in (a 0..360 file against a
    -66..-54 box would otherwise interpolate to all-NaN), and both axes sorted ascending, which
    ``interp`` needs.
    """
    if min(map(float, domain["lon"])) < 0:
        ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180)
    ds = ds.sortby("lat").sortby("lon")
    lat, lon = domain_grid(domain, res)
    return ds.interp(lat=lat, lon=lon, method=method)


def _preprocess(ds: xr.Dataset, recipe: dict) -> xr.Dataset:
    """Per-file work, pushed into ``open_mfdataset(preprocess=...)`` on purpose.

    Selecting variables and depth levels here means xarray only ever decompresses what is kept: five
    levels out of fifty instead of all fifty, and any regridding downstream then runs on five. Doing
    it after the merge -- which is what this script used to do -- reads and interpolates everything
    first and throws most of it away. On the GLORYS Gulf Stream box that difference is roughly an
    order of magnitude, and it is what pushed the job past walltime (found and fixed upstream in
    NOSC, `8f97ab8`).
    """
    ds = normalise_dims(ds)
    wanted = [v for v in recipe["variables"] if v in ds.data_vars]
    if wanted:
        ds = ds[wanted]
    if recipe.get("domain"):
        # Before anything is read: a GLORYS file is the whole globe at 1/12 deg (2041 x 4320 x 50),
        # and the Gulf Stream box is 1/400th of it. This key used to be accepted and ignored, which
        # on a year of files meant a 1949 GB write and an OOM kill.
        res = resolution_of(recipe)
        domain = recipe["domain"]
        if res is not None:
            # One target step of margin, so the edge nodes of the target grid have source cells on
            # both sides: the native cell centres rarely fall exactly on the domain bounds.
            domain = {k: [float(min(v)) - res, float(max(v)) + res] if k in ("lat", "lon") else v
                      for k, v in domain.items()}
        ds = _select_domain(ds, domain)
    if "depth" in ds.dims:
        if not recipe.get("keep_depth", False):
            ds = ds.isel(depth=0, drop=True)
        elif recipe.get("depth_indices"):
            ds = ds.isel(depth=list(recipe["depth_indices"]))
    res = resolution_of(recipe)
    if res is not None:
        # Per file, after the cut and the depth selection, as NOSC does: each global file is reduced
        # to the box and the kept levels before it is interpolated and merged.
        method = recipe.get("method", "linear")
        ds = _to_resolution(ds, recipe["domain"], res, "linear" if method == "none" else method)
    return ds


def run(recipe: dict, *, force: bool = False) -> Path:
    path = Path(recipe["output"])
    # Idempotent. These jobs are long enough to be killed on walltime, and a re-run that starts from
    # scratch every time never finishes. Delete the file, or pass --force, to rebuild.
    res = resolution_of(recipe)
    label = "native" if res is None else f"{res:g}"
    if path.exists() and not force:
        # Same file name at every resolution, so a skip must check which one is on disk: otherwise
        # switching OCEANML3D_TARGET_RES silently reuses the previous grid. A recipe that names no
        # resolution (the merge step) must match what its inputs carry instead.
        found = _stored_resolution(path)
        expected = label if "resolution" in recipe else _first_input_resolution(recipe)
        if found is not None and expected is not None and found != expected:
            raise RuntimeError(f"{path} exists at resolution {found}, expected {expected}. "
                               f"Delete it (and the files derived from it) or pass --force.")
        print(f"[regrid] {path} already exists (resolution {found or 'unrecorded'}), skipped "
              f"(--force to rebuild)")
        return path

    if recipe.get("keep_depth") is False and recipe.get("depth_indices"):
        raise ValueError("recipe sets depth_indices but keep_depth is false: the depth axis is "
                         "dropped before the selection could apply. Set keep_depth: true.")

    seen: set[str] = set()

    def _pre(d: xr.Dataset) -> xr.Dataset:
        seen.add(str(d.attrs.get("oceanml3d_resolution", "unrecorded")))
        return _preprocess(d, recipe)

    open_kw: dict = {"combine": "by_coords", "preprocess": _pre}
    if "chunks" in recipe:
        # Only when the recipe asks. Forcing a time chunking that does not match how the files are
        # stored re-fragments every read, and the compressed write then streams through that
        # fragmented graph -- slower than the native chunking by a wide margin (NOSC, `dd520be`).
        open_kw["chunks"] = recipe["chunks"]
    pattern = str(recipe["input"])
    if any(c in pattern for c in "*?["):
        # `glob.glob(..., recursive=True)`, not `Path().glob()`: the latter raises
        # `NotImplementedError: Non-relative patterns are unsupported` on an absolute pattern, which
        # every real recipe has. `recursive=True` is what makes `**` descend, as the per-year GLORYS
        # recipes need against a mirror laid out by year and month.
        inputs = sorted(glob.glob(pattern, recursive=True))
        if not inputs:
            raise FileNotFoundError(
                f"no file matches {pattern}. Check the pattern, and -- inside a container -- that the "
                f"directory is bound: an unbound path is empty rather than missing."
            )
        print(f"[regrid] {len(inputs)} input file(s): {Path(inputs[0]).name} … {Path(inputs[-1]).name}")
        # `parallel=True` opens and preprocesses files from dask threads. That is faster where it
        # works -- and where it does not it does not raise, it segfaults, because netCDF4/HDF5 is not
        # thread-safe and the C library dies under the interpreter:
        #     jobs/env/_lib.sh: line 40: 23487 Segmentation fault  singularity exec ...
        # Whether a given build is safe depends on how HDF5 was compiled, so it cannot be decided
        # here. Off by default; turn it on per recipe once you have seen it work on that machine.
        if recipe.get("parallel", False):
            open_kw["parallel"] = True
    else:
        inputs = pattern
    if str(inputs if isinstance(inputs, str) else inputs[0]).rstrip("/").endswith(".zarr"):
        # Zarr stores are directories; without the engine xarray tries them as netCDF and fails.
        open_kw["engine"] = "zarr"
    # Pass the resolved list rather than the pattern: xarray does its own globbing, but not `**`.
    # The synchronous scheduler for the same reason as `parallel` above: the read goes through the
    # netCDF4 C library, and a threaded scheduler is the other way to reach it from several threads
    # at once. This costs little here -- the work is I/O against one shared filesystem, not CPU --
    # and `dask_scheduler: threads` in the recipe restores the old behaviour.
    with dask.config.set(scheduler=recipe.get("dask_scheduler", "synchronous")):
        ds = xr.open_mfdataset(inputs, **open_kw)

    if len(seen - {"unrecorded"}) > 1:
        # combine="by_coords" would take the union of the grids and pad each file with NaN.
        raise RuntimeError(f"input files were prepared at different resolutions {sorted(seen)}: "
                           f"re-prepare them at one (see OCEANML3D_TARGET_RES) before merging")
    ds = ds.rename({k: v for k, v in recipe["variables"].items() if k in ds.data_vars})
    if "time" in recipe:
        ds = ds.sel(time=slice(*recipe["time"]))

    method = recipe.get("method", "linear")
    if res is not None:
        out = ds                    # already on the target grid, per file in _preprocess
        print(f"[regrid] resolution {res:g} deg on the domain grid: "
              f"{ds.sizes['lat']} x {ds.sizes['lon']} (lat x lon), "
              f"{'linear' if method == 'none' else method}")
    elif method == "none":
        out = ds
    else:
        lat, lon = target_grid(recipe)
        if method == "conservative":
            import xesmf as xe

            out = xe.Regridder(ds, xr.Dataset({"lat": lat, "lon": lon}), "conservative")(ds)
        else:
            out = ds.interp(lat=lat, lon=lon, method=method)

    out = out.astype(np.float32)
    if res is not None or "resolution" in recipe:
        out.attrs["oceanml3d_resolution"] = label
    elif method == "none":
        out.attrs["oceanml3d_resolution"] = str(ds.attrs.get("oceanml3d_resolution", "native"))
    else:
        out.attrs["oceanml3d_resolution"] = "grid"
    gb = out.nbytes / 1e9
    print(f"[regrid] writing {path} ({dict(out.sizes)}, {gb:.2f} GB uncompressed)")
    check_size(gb, recipe)
    path.parent.mkdir(parents=True, exist_ok=True)
    scheduler = recipe.get("dask_scheduler", "synchronous")

    # Written under a hidden temporary name and renamed at the end. The skip above tests only for
    # existence, so a job killed mid-write must not leave something at `path`: a half-written Zarr
    # store is a directory that exists, and the next run would have skipped it as complete.
    tmp = path.with_name(f".tmp.{path.name}")
    _remove(tmp)
    if path.suffix == ".zarr":
        _write_zarr(out, tmp, recipe, scheduler)
    else:
        # Materialise before writing. A subdomained, depth-reduced box is a few GB at most, and one
        # in-memory write beats streaming a compressed write through a dask graph by a long way.
        with dask.config.set(scheduler=scheduler):
            out = out.load()
        out.to_netcdf(tmp, encoding={v: {"zlib": True, "complevel": 4} for v in out.data_vars})
    out.close()
    ds.close()
    _remove(path)
    os.replace(tmp, path)
    print(f"[regrid] done: {path}")
    return path


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _stored_resolution(path: Path) -> str | None:
    try:
        opener = xr.open_zarr if path.suffix == ".zarr" else xr.open_dataset
        with opener(path) as ds:
            value = ds.attrs.get("oceanml3d_resolution")
    except Exception:  # noqa: BLE001 -- an unreadable file is reported by whoever reads it next
        return None
    return None if value is None else str(value)


def _first_input_resolution(recipe: dict) -> str | None:
    matches = sorted(glob.glob(str(recipe["input"]), recursive=True))
    return _stored_resolution(Path(matches[0])) if matches else None


DEFAULT_MAX_GB = 100.0


def check_size(gb: float, recipe: dict) -> None:
    """Refuse an output larger than ``max_gb`` (default 100) before anything is read.

    Everything this script writes is a regional box: a few GB per year, tens for a decade. An output
    in the hundreds of GB means the recipe did not do what it says -- a domain not applied, a glob
    that matched other years -- and the job would otherwise spend its walltime reading the globe
    before the node kills it. Raise ``max_gb`` in the recipe if the size is really intended.
    """
    limit = float(recipe.get("max_gb", DEFAULT_MAX_GB))
    if gb > limit:
        raise RuntimeError(
            f"output would be {gb:.1f} GB uncompressed, above max_gb={limit:g}. Check `domain`, "
            f"`time` and the `input` glob (the file list above); set max_gb in the recipe if "
            f"this size is intended."
        )


def _write_zarr(out: xr.Dataset, path: Path, recipe: dict, scheduler: str) -> None:
    """Write a Zarr store, with the chunking stated rather than inherited.

    Zarr is the right format for the *intermediate* files: a chunk is an independent object, there
    is no C-library global state, and concurrent reads are what it was designed for -- so the
    thread-safety problem that forces `parallel: false` on the netCDF path simply does not arise.
    Reading these back, `parallel: true` and `dask_scheduler: threads` are safe.

    The cost is inodes: every chunk is a file. Two choices keep that small.

    **Zarr format 2, pinned.** zarr-python 3 writes format 3 by default, which nests every chunk
    key in directories (``c/<t>/0/0/0``): measured on one year of the Gulf Stream box, 210 inodes
    against 80 in format 2, and 1 920 against 536 for the eleven-year store. Format 2 keeps flat
    keys (``<t>.0.0.0``) and is what every reader of this project handles.

    **Chunks of tens of megabytes.** ``{time: 32}`` with whole depth and horizontal slabs: ~12
    chunks per variable and year. Daily chunks would multiply the count by 32.

    The whole GLORYS chain then costs ~1 400 inodes (eleven yearly stores plus the merged one)
    against quotas in the hundreds of thousands -- Jean Zay's ``$WORK`` allows 500 000 for the
    project. The count printed below is the real one: chunks, metadata files and directories.
    """
    chunks = recipe.get("zarr_chunks") or {"time": 32}
    chunks = {d: min(int(n), out.sizes[d]) for d, n in chunks.items() if d in out.sizes}
    out = out.chunk({**{d: out.sizes[d] for d in out.dims}, **chunks})

    n_chunks, biggest = 0, 0.0
    for name, var in out.variables.items():
        if name in out.dims:        # dimension coordinates are indexes, written as a single chunk
            n_chunks += 1
            continue
        per_var, nbytes = 1, var.dtype.itemsize
        for dim, size in zip(var.dims, var.shape, strict=True):
            step = chunks.get(dim, size)
            per_var *= -(-size // step)
            nbytes *= min(step, size)
        n_chunks += per_var
        biggest = max(biggest, nbytes)
    # format 2: per array a directory, .zarray and .zattrs; per store the root, .zgroup, .zattrs
    # and .zmetadata. Checked against os.walk in tests/test_regrid_domain.py.
    inodes = n_chunks + 3 * len(out.variables) + 4
    print(f"[regrid] zarr chunks {chunks} -> {inodes} inodes ({n_chunks} chunks), "
          f"largest chunk {biggest / 1e6:.0f} MB uncompressed")
    if inodes > 50_000:
        print(f"[regrid] WARNING: {inodes} inodes is a lot. On Jean Zay $WORK allows 500 000 for the "
              f"whole project. Increase zarr_chunks.")

    # Encodings inherited from netCDF inputs (chunksizes, zlib, contiguous, ...) are not valid Zarr
    # encodings and some make to_zarr raise; let xarray pick Zarr's own.
    for var in out.variables.values():
        var.encoding = {}
    kw = {"zarr_format": 2} if "zarr_format" in inspect.signature(out.to_zarr).parameters else {}
    with dask.config.set(scheduler=scheduler):
        out.to_zarr(path, mode="w", consolidated=True, **kw)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--force", action="store_true", help="rebuild even if the output exists")
    a = p.parse_args()
    print(run(yaml.safe_load(os.path.expandvars(Path(a.config).read_text())), force=a.force))
