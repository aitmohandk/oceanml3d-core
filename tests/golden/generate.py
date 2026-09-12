"""Record the behaviour of the code BEFORE a refactoring, so the merge can be proven behaviour-preserving.

    python tests/golden/generate.py            # writes tests/golden/*.npz from the current HEAD

Cases stay cheap, deterministic, CPU-only and free of trained weights.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = pathlib.Path(__file__).resolve().parent


def commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def qg_2layer():
    from models.qg_dynamics import QGDynamics
    d = QGDynamics(nx=32, dt=7200.0)
    s0 = d._flatten(d._initial_q(1, seed=7, device=d.device)).squeeze(0)
    traj = d.rollout_trajectory(s0, steps=5)
    wind = d.generate_wind_state(6, seed=3)
    return {"state0": s0.detach().numpy(), "trajectory": traj.detach().numpy(),
            "wind_state": wind.detach().numpy(),
            "wind_curl": d.wind_curl_field(wind).detach().numpy(),
            "ke": d.kinetic_energy(traj).detach().numpy(),
            "enstrophy": d.enstrophy(traj).detach().numpy()}


def qg_1layer():
    from models.qg1l_dynamics import QG1LDynamics
    d = QG1LDynamics(nx=32, dt=7200.0)
    s0 = d._flatten(d._initial_q(1, seed=7, device=d.device)).squeeze(0)
    traj = d.rollout_trajectory(s0, steps=5)
    wind = d.generate_wind_state(6, seed=3)
    return {"state0": s0.detach().numpy(), "trajectory": traj.detach().numpy(),
            "wind_state": wind.detach().numpy(),
            "wind_curl": d.wind_curl_field(wind).detach().numpy(),
            "ke": d.kinetic_energy(traj).detach().numpy(),
            "enstrophy": d.enstrophy(traj).detach().numpy()}


def lorenz63():
    from models.lorenz63_dynamics import Lorenz63Dynamics
    d = Lorenz63Dynamics(dt=0.01, c1=0.0, clip_range=None)
    s = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)
    W = torch.zeros(1, dtype=torch.float64)
    traj = [s.numpy().copy()]
    for _ in range(200):
        s = d.step(s, W, sigma=10.0, rho=28.0, beta=8.0 / 3.0)
        traj.append(s.numpy().copy())
    return {"trajectory": np.concatenate(traj)}


CASES = {"qg_2layer": qg_2layer, "qg_1layer": qg_1layer, "lorenz63": lorenz63}

if __name__ == "__main__":
    torch.manual_seed(0)
    wanted = sys.argv[1:] or list(CASES)
    for name in wanted:
        try:
            data = CASES[name]()
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {name}: {exc}")
            continue
        np.savez(OUT / f"{name}.npz", commit=commit(), **data)
        print(f"wrote {name}.npz ({', '.join(f'{k}{v.shape}' for k, v in data.items() if hasattr(v, 'shape'))})")
