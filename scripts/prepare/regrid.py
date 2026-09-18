"""Regrid any gridded product onto the reference grid of a task (replaces the ~20
``interpolation_*_4th.py`` scripts of NOSC with one recipe-driven tool).

Recipe (YAML):
    input: /path/or/glob/*.nc
    output: /path/out_4th.nc
    variables: {adt: zos, ugos: ugos, vgos: vgos}   # rename on the fly
    grid: {lat: [-80, 90, 0.25], lon: [-180, 180, 0.25]}   # or  reference: /path/ref.nc
    time: ["2010-01-01", "2022-01-01"]
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

CAREFUL WITH ``depth_indices``. It selects *positions on the source file's depth axis*, and the
positions it leaves are what the data config's own ``depth_index`` then addresses. Change this list
and every ``depth_index`` in ``config/data/*.yaml`` silently means a different depth. If in doubt,
leave it out: store every level the download produced and let the task select.
"""
from __future__ import annotations

import argparse
import glob
import os
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
    if "depth" in ds.dims:
        if not recipe.get("keep_depth", False):
            ds = ds.isel(depth=0, drop=True)
        elif recipe.get("depth_indices"):
            ds = ds.isel(depth=list(recipe["depth_indices"]))
    return ds


def run(recipe: dict, *, force: bool = False) -> Path:
    path = Path(recipe["output"])
    # Idempotent. These jobs are long enough to be killed on walltime, and a re-run that starts from
    # scratch every time never finishes. Delete the file, or pass --force, to rebuild.
    if path.exists() and not force:
        print(f"[regrid] {path} already exists, skipped (--force to rebuild)")
        return path

    if recipe.get("keep_depth") is False and recipe.get("depth_indices"):
        raise ValueError("recipe sets depth_indices but keep_depth is false: the depth axis is "
                         "dropped before the selection could apply. Set keep_depth: true.")

    open_kw: dict = {"combine": "by_coords", "preprocess": lambda d: _preprocess(d, recipe)}
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
    # Pass the resolved list rather than the pattern: xarray does its own globbing, but not `**`.
    # The synchronous scheduler for the same reason as `parallel` above: the read goes through the
    # netCDF4 C library, and a threaded scheduler is the other way to reach it from several threads
    # at once. This costs little here -- the work is I/O against one shared filesystem, not CPU --
    # and `dask_scheduler: threads` in the recipe restores the old behaviour.
    with dask.config.set(scheduler=recipe.get("dask_scheduler", "synchronous")):
        ds = xr.open_mfdataset(inputs, **open_kw)

    ds = ds.rename({k: v for k, v in recipe["variables"].items() if k in ds.data_vars})
    if "time" in recipe:
        ds = ds.sel(time=slice(*recipe["time"]))

    method = recipe.get("method", "linear")
    if method == "none":
        out = ds
    else:
        lat, lon = target_grid(recipe)
        if method == "conservative":
            import xesmf as xe

            out = xe.Regridder(ds, xr.Dataset({"lat": lat, "lon": lon}), "conservative")(ds)
        else:
            out = ds.interp(lat=lat, lon=lon, method=method)

    out = out.astype(np.float32)
    print(f"[regrid] writing {path} "
          f"({dict(out.sizes)}, {out.nbytes / 1e9:.2f} GB uncompressed)")
    path.parent.mkdir(parents=True, exist_ok=True)
    scheduler = recipe.get("dask_scheduler", "synchronous")

    if path.suffix == ".zarr":
        _write_zarr(out, path, recipe, scheduler)
    else:
        # Materialise before writing. A subdomained, depth-reduced box is a few GB at most, and one
        # in-memory write beats streaming a compressed write through a dask graph by a long way.
        with dask.config.set(scheduler=scheduler):
            out = out.load()
        out.to_netcdf(path, encoding={v: {"zlib": True, "complevel": 4} for v in out.data_vars})
    out.close()
    ds.close()
    print(f"[regrid] done: {path}")
    return path


def _write_zarr(out: xr.Dataset, path: Path, recipe: dict, scheduler: str) -> None:
    """Write a Zarr store, with the chunking stated rather than inherited.

    Zarr is the right format for the *intermediate* files: a chunk is an independent object, there
    is no C-library global state, and concurrent reads are what it was designed for -- so the
    thread-safety problem that forces `parallel: false` on the netCDF path simply does not arise.
    Reading these back, `parallel: true` and `dask_scheduler: threads` are safe.

    The cost is inodes, and on Jean Zay that is the quota that bites: 500 000 on `$WORK`, shared
    across the project. Chunk size is therefore not a detail. For the GLORYS Gulf Stream box --
    4018 days, 26 levels, 144x144, three 3D variables plus SSH -- `{time: 32}` with whole depth and
    horizontal slabs gives ~69 MB chunks and about 400 objects. The same data in daily chunks would
    be ~16 000. Aim for tens to hundreds of megabytes, aligned on how the data is read: for this
    project, spatial patches traversed in time.
    """
    chunks = recipe.get("zarr_chunks") or {"time": 32}
    chunks = {d: min(int(n), out.sizes[d]) for d, n in chunks.items() if d in out.sizes}
    out = out.chunk({**{d: out.sizes[d] for d in out.dims}, **chunks})

    n_objects, biggest = 0, 0.0
    for var in out.data_vars.values():
        per_var, nbytes = 1, var.dtype.itemsize
        for dim, size in zip(var.dims, var.shape, strict=True):
            step = chunks.get(dim, size)
            per_var *= -(-size // step)
            nbytes *= min(step, size)
        n_objects += per_var
        biggest = max(biggest, nbytes)
    print(f"[regrid] zarr chunks {chunks} -> ~{n_objects} chunk objects, "
          f"largest {biggest / 1e6:.0f} MB uncompressed")
    if n_objects > 50_000:
        print(f"[regrid] WARNING: {n_objects} objects is a lot of inodes. On Jean Zay $WORK allows "
              f"500 000 for the whole project. Increase zarr_chunks.")

    with dask.config.set(scheduler=scheduler):
        out.to_zarr(path, mode="w", consolidated=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--force", action="store_true", help="rebuild even if the output exists")
    a = p.parse_args()
    print(run(yaml.safe_load(os.path.expandvars(Path(a.config).read_text())), force=a.force))
