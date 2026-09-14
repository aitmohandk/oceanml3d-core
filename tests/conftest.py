"""
Pytest fixtures.

Two disjoint families: the Lorenz-63 DA fixtures (configs, datasets, device) used by the toy tests,
and the synthetic-ocean fixtures (`synthetic_dir`, `catalog`, `variables`) the transplanted NOSC
tests build on — a 40-day, 48x48 synthetic surface dataset written once per session, so the ocean
tests need no data on disk.
"""
import sys
from pathlib import Path

import pytest
import torch

from oceanml3d.data.lorenz63 import Lorenz63Config, Lorenz63Dataset

ROOT = Path(__file__).resolve().parents[1]
# `scripts/` is not a package: the synthetic-data generator is imported by module name.
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture
def device():
    """CPU device for testing (ensures reproducibility)."""
    return torch.device("cpu")


@pytest.fixture
def simple_config():
    """Small, fast config for unit tests."""
    return Lorenz63Config(
        case=1,
        seed=42,
        num_windows=5,
        T_max=1.0,
        dt=0.01,
        obs_interval=10,
        spinup_steps=1000,
        R_var=0.5,
        B_var=2.0,
    )


@pytest.fixture
def cs1_config():
    """Case Study 1: noise-free forcing, correct params."""
    return Lorenz63Config(
        case=1,
        param_bias=0.0,
        seed=123,
        num_windows=5,
        T_max=5.0,
        dt=0.01,
        obs_interval=20,
        R_var=0.5,
        B_var=2.0,
        spinup_steps=10000,
    )


@pytest.fixture
def cs2_config():
    """Case Study 2: noisy forcing, biased params."""
    return Lorenz63Config(
        case=2,
        param_bias=0.05,
        seed=123,
        num_windows=5,
        T_max=5.0,
        dt=0.01,
        obs_interval=20,
        R_var=0.5,
        B_var=2.0,
        spinup_steps=10000,
    )


@pytest.fixture
def cs1_dataset(cs1_config):
    """Dataset instance for Case Study 1."""
    return Lorenz63Dataset(cs1_config)


@pytest.fixture
def cs2_dataset(cs2_config):
    """Dataset instance for Case Study 2."""
    return Lorenz63Dataset(cs2_config)


@pytest.fixture
def tiny_config():
    """Ultra-small config for quick sanity checks."""
    return Lorenz63Config(
        case=1,
        seed=42,
        num_windows=2,
        T_max=0.5,
        dt=0.01,
        obs_interval=5,
        spinup_steps=500,
    )


# -- synthetic ocean fixtures (transplanted with the NOSC port) ---------------------------------


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory) -> Path:
    from make_synthetic_data import make

    out = tmp_path_factory.mktemp("synthetic")
    make(out, days=40, n=48)
    return out


@pytest.fixture(scope="session")
def catalog(synthetic_dir):
    from oceanml3d.catalog import Catalog

    return Catalog({
        "ssh": str(synthetic_dir / "synthetic_surface.nc"),
        "era5": str(synthetic_dir / "synthetic_surface.nc"),
        "bathy": str(synthetic_dir / "synthetic_static.nc"),
        "drifters": str(synthetic_dir / "synthetic_surface.nc"),
    })


@pytest.fixture(scope="session")
def variables():
    from oceanml3d.variables import VariableSet

    return VariableSet.from_config({
        "ssh": {"source": "ssh", "var_name": "zos", "role": "input"},
        "ugos": {"source": "ssh", "role": "input"},
        "vgos": {"source": "ssh", "role": "input"},
        "u10": {"source": "era5", "role": "input"},
        "v10": {"source": "era5", "role": "input"},
        "lat": {"source": "ssh", "role": "static"},
        "bathy": {"source": "bathy", "var_name": "deptho", "role": "static"},
        "u_drifter": {"source": "drifters", "role": "target"},
        "v_drifter": {"source": "drifters", "role": "target"},
    })
