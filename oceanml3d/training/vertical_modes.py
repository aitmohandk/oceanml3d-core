"""Vertical EOF basis for the vertical-modes head (port of NOSC vertical_modes.py, numpy part)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from oceanml3d.data.open import normalise_dims


def eofs_from_profiles(profiles: np.ndarray, n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    """profiles (n_samples, n_levels) -> (components (n_modes, n_levels) orthonormal, explained variance ratio)."""
    profiles = np.asarray(profiles, dtype=np.float64)
    profiles = profiles[np.isfinite(profiles).all(axis=1)]
    if profiles.shape[0] < profiles.shape[1]:
        raise ValueError(f"need more samples ({profiles.shape[0]}) than levels ({profiles.shape[1]})")
    anom = profiles - profiles.mean(axis=0, keepdims=True)
    eigval, eigvec = np.linalg.eigh(anom.T @ anom / (anom.shape[0] - 1))
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    n_modes = min(n_modes, eigvec.shape[1])
    return eigvec[:, :n_modes].T, eigval[:n_modes] / eigval.sum()


def compute_vertical_eofs(truth: str | Path, var: str, depth_indices: list[int], n_modes: int, output_npz: str | Path,
                          time_slice: slice | None = None, domain: dict | None = None, sample_stride: int = 4) -> Path:
    da = normalise_dims(xr.open_dataset(truth))[var].isel(depth=list(depth_indices))
    if time_slice is not None:
        da = da.sel(time=time_slice)
    if domain:
        da = da.sel({k: v for k, v in domain.items() if k in da.dims})
    da = da.isel(time=slice(None, None, sample_stride), lat=slice(None, None, sample_stride), lon=slice(None, None, sample_stride))
    prof = da.transpose("time", "lat", "lon", "depth").values.reshape(-1, len(depth_indices))
    prof = prof[np.isfinite(prof).all(axis=1)]
    mean, std = prof.mean(0), prof.std(0)
    std = np.where(std > 0, std, 1.0)
    comp, evr = eofs_from_profiles((prof - mean) / std, n_modes)
    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_npz, components=comp, explained_variance_ratio=evr, level_mean=mean, level_std=std,
             depth_indices=np.asarray(depth_indices))
    return output_npz
