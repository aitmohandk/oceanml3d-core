import numpy as np
import pandas as pd
import xarray as xr
import yaml

from oceanml3d.inference.export import open_product, write_product


def test_write_product(variables, tmp_path):
    time = pd.date_range("2019-01-01", periods=3)
    field = xr.DataArray(np.random.rand(2, 3, 4, 5).astype("f4"), dims=("channel", "time", "lat", "lon"),
                         coords={"channel": ["u_drifter", "v_drifter"], "time": time, "lat": np.arange(4), "lon": np.arange(5)})
    man = write_product(field, variables, tmp_path, "xp", depth_m=15, attrs={"model": "nosc_unet"})
    doc = yaml.safe_load(man.read_text())
    assert set(doc["variables"]) == {"u", "v"} and doc["depth_m"] == 15
    ds = open_product(man)
    assert ds.u.attrs["standard_name"] == "eastward_sea_water_velocity"
    assert ds.sizes["time"] == 3
