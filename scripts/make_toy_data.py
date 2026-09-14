"""Generate a toy-system dataset in the oceanml3d layout (Lorenz-63/96 as a 1 x K 'grid').

The framework is grid-based; a toy state of dimension K is stored as ``(time, lat=1, lon=K)``, so
Lorenz runs go through exactly the same datamodule, patches, models, export and metrics as an ocean
task. That is the cheapest possible check that the interfaces are not ocean-specific — and it is
what method development in 4dvarnet-fm-opencode needs.

    python scripts/make_toy_data.py --system lorenz96 --steps 20000 --obs-density 0.25
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from oceanml3d.dynamics import get_dynamics


def make(out: Path, system: str = "lorenz96", steps: int = 20000, spinup: int = 1000,
         obs_density: float = 0.25, obs_noise: float = 1.0, seed: int = 0, **dyn_kw) -> Path:
    dyn = get_dynamics(system, **dyn_kw)
    traj = dyn.simulate(steps, spinup=spinup, seed=seed)                      # (steps, K)
    rng = np.random.default_rng(seed + 1)
    mask = rng.random(traj.shape) < obs_density
    obs = np.where(mask, traj + rng.normal(0, obs_noise, traj.shape), np.nan)
    time = pd.date_range("2000-01-01", periods=steps, freq="6h")              # a regular axis, units are arbitrary
    dims = ("time", "lat", "lon")
    coords = {"time": time, "lat": [0.0], "lon": np.arange(traj.shape[1], dtype=float)}
    ds = xr.Dataset({"state": (dims, traj[:, None, :].astype(np.float32)),
                     "obs": (dims, obs[:, None, :].astype(np.float32)),
                     "obs_mask": (dims, mask[:, None, :].astype(np.float32))},
                    coords=coords, attrs={"system": system, "dt": dyn.dt, "obs_density": obs_density,
                                          "obs_noise": obs_noise, "state_dim": traj.shape[1]})
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{system}.nc"
    ds.to_netcdf(path)
    print(f"wrote {path}: {steps} steps, state_dim={traj.shape[1]}, {100 * mask.mean():.0f}% observed")
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/toy"))
    p.add_argument("--system", default="lorenz96", choices=["lorenz63", "lorenz96"])
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--spinup", type=int, default=1000)
    p.add_argument("--obs-density", type=float, default=0.25)
    p.add_argument("--obs-noise", type=float, default=1.0)
    a = p.parse_args()
    make(a.out, a.system, a.steps, a.spinup, a.obs_density, a.obs_noise)
