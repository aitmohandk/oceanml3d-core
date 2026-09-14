"""Produce reference outputs by running the ORIGINAL implementations.

Run this on a machine where the legacy repos are checked out and their dependencies installed:

    python tests/legacy/generate_golden.py --fm ~/src/4dvarnet-fm-opencode --case all

Each case is a small deterministic computation whose result pins the science of one ported item.
Keep them cheap (seconds) and free of learned weights, so they stay reproducible for years.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

GOLDEN = Path(__file__).resolve().parent / "golden"


def _commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def case_lorenz96_2scale(fm: Path) -> dict:
    """Two-scale L96 trajectory from fm ``models/lorenz96_dynamics.py``."""
    sys.path.insert(0, str(fm))
    import torch  # noqa: F401
    from models.lorenz96_dynamics import Lorenz96Dynamics

    dyn = Lorenz96Dynamics(dt=0.001, NO=8, J=4, h=1.0, hx=1.0, eps=0.1, c1=1.0, clip_range=50.0)
    x0 = np.concatenate([np.full(8, 8.0) + np.eye(8)[0] * 0.01, np.zeros(32)])
    state = torch.tensor(x0, dtype=torch.float64)
    forcing = torch.zeros((), dtype=torch.float64)
    traj = [state.numpy().copy()]
    for _ in range(200):
        state = dyn.step(state, forcing, F=8.0)
        traj.append(state.numpy().copy())
    return {"x0": x0, "trajectory": np.stack(traj), "dt": 0.001, "NO": 8, "J": 4, "F": 8.0}


def case_lorenz63(fm: Path) -> dict:
    sys.path.insert(0, str(fm))
    import torch
    from models.lorenz63_dynamics import Lorenz63Dynamics

    dyn = Lorenz63Dynamics(dt=0.01, c1=0.0, clip_range=None)
    state = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    forcing = torch.zeros((), dtype=torch.float64)
    traj = [state.numpy().copy()]
    for _ in range(300):
        state = dyn.step(state, forcing, sigma=10.0, rho=28.0, beta=8.0 / 3.0)
        traj.append(state.numpy().copy())
    return {"x0": np.array([1.0, 1.0, 1.0]), "trajectory": np.stack(traj), "dt": 0.01}


def case_gaspari_cohn(fm: Path) -> dict:
    sys.path.insert(0, str(fm))
    from evaluation.baselines import _gaspari_cohn

    z = np.linspace(0, 2.5, 51)
    return {"z": z, "values": np.array([_gaspari_cohn(float(v)) for v in z])}


CASES = {"lorenz96_2scale": case_lorenz96_2scale, "lorenz63": case_lorenz63, "gaspari_cohn": case_gaspari_cohn}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fm", type=Path, required=True, help="path to the 4dvarnet-fm-opencode checkout")
    p.add_argument("--case", default="all", choices=["all", *CASES])
    a = p.parse_args()
    GOLDEN.mkdir(parents=True, exist_ok=True)
    for name in (CASES if a.case == "all" else [a.case]):
        data = CASES[name](a.fm)
        out = GOLDEN / f"{name}.npz"
        np.savez(out, legacy_repo=str(a.fm), legacy_commit=_commit(a.fm), case=name, **data)
        print(f"wrote {out} (legacy commit {_commit(a.fm)[:8]})")


if __name__ == "__main__":
    main()
