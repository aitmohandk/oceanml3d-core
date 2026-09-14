"""Create a tiny synthetic dataset so that `oceanml3d train experiment=smoke` runs anywhere."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def make(out: Path, days: int = 59, n: int = 80, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    lat = np.linspace(-10, 10, n)
    lon = np.linspace(-10, 10, n)
    time = pd.date_range("2019-01-01", periods=days, freq="D")
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    t = np.arange(days)[:, None, None]
    zos = 0.3 * np.sin(lo / 3 + t / 10) * np.cos(la / 3)
    ugos = -np.gradient(zos, lat, axis=1)
    vgos = np.gradient(zos, lon, axis=2)
    u10 = 5 * np.sin(t / 7 + lo / 10) + rng.normal(0, 0.5, zos.shape)
    v10 = 3 * np.cos(t / 9 + la / 10) + rng.normal(0, 0.5, zos.shape)
    u_true = ugos + 0.02 * u10
    v_true = vgos + 0.02 * v10
    mask = rng.random(zos.shape) < 0.15
    u_dr = np.where(mask, u_true + rng.normal(0, 0.02, zos.shape), np.nan)
    v_dr = np.where(mask, v_true + rng.normal(0, 0.02, zos.shape), np.nan)
    coords = {"time": time, "lat": lat, "lon": lon}
    dims = ("time", "lat", "lon")
    ds = xr.Dataset({k: (dims, v.astype(np.float32)) for k, v in
                     dict(zos=zos, ugos=ugos, vgos=vgos, u10=u10, v10=v10, u_drifter=u_dr, v_drifter=v_dr).items()},
                    coords=coords)
    out.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out / "synthetic_surface.nc")
    static = xr.Dataset({"deptho": (("lat", "lon"), (3000 + 500 * np.cos(la / 2)).astype(np.float32))},
                        coords={"lat": lat, "lon": lon})
    static.to_netcdf(out / "synthetic_static.nc")
    print(f"wrote synthetic data to {out}")


def make_osse(out: Path, days: int = 59, n: int = 80, n_depth: int = 3, seed: int = 0) -> None:
    """Multi-depth 'truth' (zos, thetao/uo/vo x n_depth), a virtual-ARGO file and the static file.
    Pseudo-obs are then produced by `oceanml3d command=prepare-obs experiment=osse3d_smoke`."""
    rng = np.random.default_rng(seed)
    lat = np.linspace(-10, 10, n)
    lon = np.linspace(-10, 10, n)
    depth = np.array([0.5, 5.0, 15.0, 30.0])[: max(n_depth, 1)] if n_depth <= 4 else np.linspace(0.5, 200, n_depth)
    time = pd.date_range("2019-01-01", periods=days, freq="D")
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    t = np.arange(days)[:, None, None]
    zos = 0.3 * np.sin(lo / 3 + t / 10) * np.cos(la / 3)
    ug, vg = -np.gradient(zos, lat, axis=1), np.gradient(zos, lon, axis=2)
    z = depth[:, None, None, None]
    decay = np.exp(-z / 50.0)
    thetao = 20 - 0.05 * z + 3 * zos[None] * decay
    uo, vo = ug[None] * decay, vg[None] * decay
    dims4 = ("time", "depth", "lat", "lon")
    coords = {"time": time, "depth": depth, "lat": lat, "lon": lon}
    md = xr.Dataset({k: (dims4, np.transpose(v, (1, 0, 2, 3)).astype(np.float32)) for k, v in
                     dict(thetao=thetao, uo=uo, vo=vo).items()}, coords=coords)
    out.mkdir(parents=True, exist_ok=True)
    md.to_netcdf(out / "synthetic_osse_multidepth.nc")
    xr.Dataset({"zos": (("time", "lat", "lon"), zos.astype(np.float32))},
               coords={"time": time, "lat": lat, "lon": lon}).to_netcdf(out / "synthetic_osse_surface.nc")
    if not (out / "synthetic_static.nc").exists():
        xr.Dataset({"deptho": (("lat", "lon"), (3000 + 500 * np.cos(la / 2)).astype(np.float32))},
                   coords={"lat": lat, "lon": lon}).to_netcdf(out / "synthetic_static.nc")
    from oceanml3d.obs.argo_virtual import build_virtual_argo

    n_prof = days * 6
    profiles = pd.DataFrame({"time": rng.choice(time, n_prof), "lat": rng.uniform(-9, 9, n_prof), "lon": rng.uniform(-9, 9, n_prof)})
    for i in range(len(depth)):
        profiles[f"d{i:02d}"] = (rng.random(n_prof) < 0.9).astype(float)
    profiles.to_csv(out / "synthetic_argo_profiles.csv", index=False)
    build_virtual_argo(profiles, out / "synthetic_osse_multidepth.nc", "thetao", list(range(len(depth))), out / "synthetic_argo_virtual.nc")
    print(f"wrote synthetic OSSE data to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/synthetic"))
    p.add_argument("--days", type=int, default=59)
    p.add_argument("--osse", action="store_true", help="also create the multi-depth OSSE truth + virtual ARGO")
    a = p.parse_args()
    make(a.out, a.days)
    if a.osse:
        make_osse(a.out, a.days)
