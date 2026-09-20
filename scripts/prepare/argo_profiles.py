"""Build the ARGO profile coverage table for virtual floats (`oceanml3d prepare-obs`).

Recipe (YAML):
    source: argopy                     # argopy (downloads) | gdac (local mirror, no network)
    gdac_dir: /home/ref-argo/gdac      # if source: gdac -- Datarmor mirrors the Argo GDAC
    bbox: [-66, -54, 32, 44]           # lon_min lon_max lat_min lat_max
    time: ["2010-01-01", "2020-01-01"]
    truth: ${OCEANML3D_DATA}/glorys/glorys_gs_multidepth_2010-2020.zarr   # depth values for the indices
    depth_indices: [0, 2, 4, ...]
    value_var: TEMP
    spike_thresholds: {TEMP: 2.0}
    output: ${OCEANML3D_DATA}/argo/argo_profiles_gs.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import yaml

from oceanml3d.obs import argo

# A batch job's stdout is a pipe, fully buffered: killed on walltime, the log showed nothing past the
# first line and there was no way to tell whether the index was found or how far the reading got.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass


def run(recipe: dict) -> Path:
    out = Path(recipe["output"])
    if out.exists() and not recipe.get("force", False):
        print(f"[argo] {out} already exists, skipped (delete it to rebuild)")
        return out
    lon0, lon1, lat0, lat1 = recipe["bbox"]
    t0 = time.monotonic()
    if recipe.get("source", "argopy") == "gdac":
        ds = argo.fetch_argo_profiles_local(recipe["gdac_dir"], lon0, lon1, lat0, lat1, *recipe["time"],
                                            index=recipe.get("index"))
    else:
        ds = argo.fetch_argo_profiles_chunked(lon0, lon1, lat0, lat1, *recipe["time"])
    print(f"[argo] read: {ds.sizes.get('N_POINTS', 0)} points in {time.monotonic() - t0:.0f} s")
    ds = argo.apply_standard_qc(ds, spike_thresholds=recipe.get("spike_thresholds"))
    print(f"[argo] after QC: {ds.sizes.get('N_POINTS', 0)} points")
    idx = list(recipe["depth_indices"])
    table = argo.coverage_table(ds, argo.depth_values_from_truth(recipe["truth"], idx), idx, recipe.get("value_var", "TEMP"))
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False) if out.suffix == ".csv" else table.to_parquet(out)
    print(f"{len(table)} profiles -> {out}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    run(yaml.safe_load(os.path.expandvars(Path(p.parse_args().config).read_text())))
