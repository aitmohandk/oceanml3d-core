"""Patch weights used both in the loss and when stitching patches back together."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def constant_crop_weight(patch: Mapping[str, int], crop: Mapping[str, int]) -> np.ndarray:
    """1 inside the patch minus a border of ``crop`` cells per dim, 0 on the border."""
    w = np.zeros(tuple(patch[d] for d in ("time", "lat", "lon")), dtype=np.float32)
    sl = tuple(slice(crop.get(d, 0), patch[d] - crop.get(d, 0)) for d in ("time", "lat", "lon"))
    w[sl] = 1.0
    return w


def triangular_time_weight(patch: Mapping[str, int], crop: Mapping[str, int], offset: int = 1) -> np.ndarray:
    """Constant crop multiplied by a triangle in time peaking at the window centre."""
    pw = constant_crop_weight(patch, crop)
    t = np.arange(patch["time"])
    tri = 1.0 - np.abs(offset + 2 * t - patch["time"]) / patch["time"]
    return (pw * tri[:, None, None]).astype(np.float32)


def patch_weight(kind: str, patch: Mapping[str, int], crop: Mapping[str, int], **kw) -> np.ndarray:
    if kind == "constant":
        return constant_crop_weight(patch, crop)
    if kind == "triangular":
        return triangular_time_weight(patch, crop, **kw)
    raise ValueError(f"unknown weight kind '{kind}' (constant|triangular)")
