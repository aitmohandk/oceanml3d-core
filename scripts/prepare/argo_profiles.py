"""Build the ARGO profile coverage table for virtual floats (`oceanml3d prepare-obs`).

Recipe (YAML):
    source: argopy                     # argopy | gdac
    gdac_dir: /home/ref-argo/gdac      # if source: gdac
    bbox: [-66, -54, 32, 44]           # lon_min lon_max lat_min lat_max
    time: ["2010-01-01", "2020-01-01"]
    truth: ${OCEANML3D_DATA}/glorys_gs_multidepth_2010-2020.nc   # gives depth values for the indices
    depth_indices: [0, 2, 4, ...]
    value_var: TEMP
    spike_thresholds: {TEMP: 2.0}
    output: ${OCEANML3D_DATA}/argo_profiles_gs.csv
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from oceanml3d.obs import argo


def run(recipe: dict) -> Path:
    lon0, lon1, lat0, lat1 = recipe["bbox"]
    if recipe.get("source", "argopy") == "gdac":
        ds = argo.fetch_argo_profiles_local(recipe["gdac_dir"], lon0, lon1, lat0, lat1, *recipe["time"])
    else:
        ds = argo.fetch_argo_profiles_chunked(lon0, lon1, lat0, lat1, *recipe["time"])
    ds = argo.apply_standard_qc(ds, spike_thresholds=recipe.get("spike_thresholds"))
    idx = list(recipe["depth_indices"])
    table = argo.coverage_table(ds, argo.depth_values_from_truth(recipe["truth"], idx), idx, recipe.get("value_var", "TEMP"))
    out = Path(recipe["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False) if out.suffix == ".csv" else table.to_parquet(out)
    print(f"{len(table)} profiles -> {out}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    run(yaml.safe_load(os.path.expandvars(Path(p.parse_args().config).read_text())))
