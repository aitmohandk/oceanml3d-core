"""Download a CMEMS dataset subset (replaces the ``import_data_*.py`` scripts of NOSC).

Recipe (YAML):
    dataset_id: cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D
    variables: [adt, ugos, vgos]
    bbox: [-180, 180, -80, 90]
    time: ["2010-01-01", "2024-01-01"]
    depth: [15, 15]           # optional
    output: /path/dir
Credentials: `copernicusmarine login` once.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def run(recipe: dict) -> None:
    import copernicusmarine

    kw = dict(dataset_id=recipe["dataset_id"], variables=recipe["variables"],
              minimum_longitude=recipe["bbox"][0], maximum_longitude=recipe["bbox"][1],
              minimum_latitude=recipe["bbox"][2], maximum_latitude=recipe["bbox"][3],
              start_datetime=recipe["time"][0], end_datetime=recipe["time"][1],
              output_directory=recipe["output"])
    if "depth" in recipe:
        kw.update(minimum_depth=recipe["depth"][0], maximum_depth=recipe["depth"][1])
    copernicusmarine.subset(**kw)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    import os
    doc = yaml.safe_load(os.path.expandvars(Path(p.parse_args().config).read_text()))
    run(doc.get("download", doc))
