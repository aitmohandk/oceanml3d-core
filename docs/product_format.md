# Product format (contract with `oceanml3d-eval`) — v1

The machine-readable contract is `oceanml3d/inference/product_contract.py`
(`validate_manifest`), duplicated **byte for byte** in `oceanml3d-eval/oceanml3d_eval/product_contract.py`
so the evaluation repo stays installable without the training stack. Both copies are pinned:

    product_contract.py sha256 = dd44f0e198d0122b88207509801ed5c60a7e2fa36172151d89d78a622fe88846

`tests/test_product_contract.py` enforces the hash in each repo. To change the format: edit the file,
bump `PRODUCT_FORMAT_VERSION`, copy it to the other repo, update both hashes and both changelogs.

`oceanml3d command=predict` (or `train` with `export.enabled=true`) writes:

```
outputs/<xp>/<run>/product/
├── product.yaml
└── daily/<xp>_YYYY-MM-DD.nc     # one file per day, time dim of length 1
```

`product.yaml`:

```yaml
name: nosc_15m_duacs
path: /abs/path/to/daily
pattern: nosc_15m_duacs_(\d{4})-(\d{2})-(\d{2})\.nc
variables:
  u: {standard_name: eastward_sea_water_velocity, units: m s-1}
  v: {standard_name: northward_sea_water_velocity, units: m s-1}
coords: {lat: lat, lon: lon, time: time}
time_coverage_hours: 24
depth_m: 15
first_date: 2019-01-01
last_date: 2019-12-31
attrs: {experiment: ..., model: nosc_unet, git_hash: ...}
```

Rules:
* variables carry CF `standard_name` + `units` attributes; short names are `u, v, ssh, sst, thetao, so`;
* regular `lat`/`lon` in degrees, `lon` in [-180, 180);
* the manifest is sufficient to open the product: `oceanml3d_eval.product.open_product("product.yaml")`.

The same manifest describes *external* products (DUACS geostrophy, GlobCurrent, NeurOST…)
in `oceanml3d-eval/products/`, so models and baselines are evaluated identically.
