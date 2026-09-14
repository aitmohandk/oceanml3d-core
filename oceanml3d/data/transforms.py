"""Named per-variable transforms applied at load time."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import xarray as xr

_TRANSFORMS: dict[str, Callable[[xr.DataArray], xr.DataArray]] = {}


def register_transform(name: str):
    def deco(fn):
        _TRANSFORMS[name] = fn
        return fn
    return deco


def get_transform(name: str) -> Callable[[xr.DataArray], xr.DataArray]:
    try:
        return _TRANSFORMS[name]
    except KeyError as exc:
        raise KeyError(f"unknown transform '{name}', available: {sorted(_TRANSFORMS)}") from exc


@register_transform("log_grad")
def log_gradient_magnitude(da: xr.DataArray, eps: float = 1e-10) -> xr.DataArray:
    """log(|dT/dx| + |dT/dy|): the SST front proxy used in NOSC (``sst_transfo``)."""
    gx = np.abs(da.differentiate("lon"))
    gy = np.abs(da.differentiate("lat"))
    return np.log(gx + gy + eps)


@register_transform("log1p")
def log1p(da: xr.DataArray) -> xr.DataArray:
    return np.log1p(da.clip(min=0))


@register_transform("identity")
def identity(da: xr.DataArray) -> xr.DataArray:
    return da
