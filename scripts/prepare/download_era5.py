"""Download ERA5 single-level fields per year via cdsapi (from NOSC import_data_era5*.py)."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def run(recipe: dict) -> None:
    import cdsapi

    c = cdsapi.Client()
    out = Path(recipe["output"])
    out.mkdir(parents=True, exist_ok=True)
    for year in range(recipe["years"][0], recipe["years"][1] + 1):
        c.retrieve("reanalysis-era5-single-levels", {
            "product_type": "reanalysis", "format": "netcdf",
            "variable": recipe.get("variables", ["10m_u_component_of_wind", "10m_v_component_of_wind"]),
            "year": str(year), "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)], "time": recipe.get("hours", ["00:00", "06:00", "12:00", "18:00"]),
            "area": recipe.get("area", [90, -180, -90, 180]),
        }, str(out / f"era5_{year}.nc"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    run(yaml.safe_load(Path(p.parse_args().config).read_text()))
