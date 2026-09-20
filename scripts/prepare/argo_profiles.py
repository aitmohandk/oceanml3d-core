"""Build the ARGO profile coverage table for virtual floats (`oceanml3d prepare-obs`).

Recipe (YAML):
    source: argopy                     # argopy (downloads) | gdac (local mirror, no network)
    gdac_dir: /home/ref-argo/gdac      # if source: gdac -- Datarmor mirrors the Argo GDAC
    index: /path/ar_index_global_prof.txt   # only if it is not beside gdac_dir or its parent
    bbox: [-66, -54, 32, 44]           # lon_min lon_max lat_min lat_max
    time: ["2010-01-01", "2020-01-01"]
    truth: ${OCEANML3D_DATA}/glorys/glorys_gs_multidepth_2010-2020.zarr   # depth values for the indices
    depth_indices: [0, 2, 4, ...]
    value_var: TEMP
    spike_thresholds: {TEMP: 2.0}
    cache_dir: ${OCEANML3D_DATA}/argo/.cache_gs   # one small file per float: a killed job resumes
    time_budget_s: 12600               # stop cleanly before the scheduler's walltime
    progress_every: 10                 # files between progress lines
    output: ${OCEANML3D_DATA}/argo/argo_profiles_gs.csv
"""
from __future__ import annotations

import os
import sys
import time

_STARTED = time.monotonic()

# Lustre and the centres' NFS mounts do not implement the POSIX locking HDF5 reaches for. Left alone
# it gives either an error about file locking or -- what happened here -- a hang on the first open,
# with no output at all. Harmless: every file this script opens is a read-only input. `regrid.py`
# has carried the same line since the GLORYS preparation hit it; this script did not, and it is the
# one that opens thousands of files on a read-only mirror.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# A batch job's stdout is a pipe, fully buffered: killed on walltime, the log showed nothing past the
# first line and there was no way to tell whether the index was found or how far the reading got.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

# Before the imports, not after: `import oceanml3d` pulls in xarray and pandas, which on a cold
# container over Lustre is a minute or more. Without this line a log that stops here is
# indistinguishable from a job that never started Python at all.
print(f"[argo] python {sys.version.split()[0]} started, importing", flush=True)

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402

from oceanml3d.obs import argo  # noqa: E402

print(f"[argo] imports done in {time.monotonic() - _STARTED:.0f} s", flush=True)

# Which `oceanml3d` is running, not which one you edited. The container image carries its own copy of
# the package (container/oceanml3d.def, /opt/oceanml3d), so a clone whose scripts are new and whose
# image is old runs one against the other -- `AttributeError: module 'oceanml3d.obs.argo' has no
# attribute 'coverage_table_local'` on a function committed hours earlier. `container_exec` puts the
# clone first on PYTHONPATH; this line is how you see that it worked.
_pkg = Path(argo.__file__).resolve().parents[2]
print(f"[argo] oceanml3d from {_pkg}", flush=True)
if _pkg != Path.cwd().resolve():
    print(f"[argo] WARNING that is not this working directory ({Path.cwd()}): the scripts and the "
          f"package come from different places. Rebuild the image, or let jobs/prepare.sh set "
          f"PYTHONPATH (jobs/env/_lib.sh, container_exec).", flush=True)


def run(recipe: dict) -> Path:
    out = Path(recipe["output"])
    if out.exists() and not recipe.get("force", False):
        print(f"[argo] {out} already exists, skipped (delete it to rebuild)")
        return out
    idx = list(recipe["depth_indices"])
    source = recipe.get("source", "argopy")
    print(f"[argo] source={source} bbox={recipe['bbox']} time={recipe['time']} -> {out}", flush=True)
    t0 = time.monotonic()

    if source == "gdac":
        # The depth levels first: a second's worth of reading that fails a wrong `truth:` immediately
        # rather than after hours of profiles.
        depth = argo.depth_values_from_truth(recipe["truth"], idx)
        table = argo.coverage_table_local(
            recipe["gdac_dir"], recipe["bbox"], *recipe["time"], depth, idx,
            value_var=recipe.get("value_var", "TEMP"), spike_thresholds=recipe.get("spike_thresholds"),
            index=recipe.get("index"), cache_dir=recipe.get("cache_dir"),
            progress_every=int(recipe.get("progress_every", 10)),
            time_budget_s=recipe.get("time_budget_s"))
    else:
        lon0, lon1, lat0, lat1 = recipe["bbox"]
        ds = argo.fetch_argo_profiles_chunked(lon0, lon1, lat0, lat1, *recipe["time"])
        print(f"[argo] read: {ds.sizes.get('N_POINTS', 0)} points in {time.monotonic() - t0:.0f} s")
        ds = argo.apply_standard_qc(ds, spike_thresholds=recipe.get("spike_thresholds"))
        print(f"[argo] after QC: {ds.sizes.get('N_POINTS', 0)} points")
        table = argo.coverage_table(ds, argo.depth_values_from_truth(recipe["truth"], idx), idx,
                                    recipe.get("value_var", "TEMP"))

    if table.empty:
        raise SystemExit("[argo] no profile survived: nothing to write. The lines above say how many "
                         "floats were opened and how many profiles each step kept.")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False) if out.suffix == ".csv" else table.to_parquet(out)
    print(f"{len(table)} profiles -> {out}  ({time.monotonic() - t0:.0f} s)", flush=True)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    try:
        run(yaml.safe_load(os.path.expandvars(Path(p.parse_args().config).read_text())))
    except argo.WalltimeBudgetExceeded as exc:
        # 75 = EX_TEMPFAIL: the step is unfinished but nothing is wrong. Resubmitting continues.
        print(f"[argo] {exc}", flush=True)
        raise SystemExit(75) from None
