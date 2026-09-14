"""Patch sampling over a ``(channel, time, lat, lon)`` array and weighted reconstruction.

Pure numpy/xarray so it can be unit-tested and reused without torch. The torch
``Dataset`` wrapper lives in :mod:`oceanml3d.data.datamodule`.
"""
from __future__ import annotations

import itertools
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import numpy as np
import xarray as xr

DIMS = ("time", "lat", "lon")


@dataclass(frozen=True)
class PatchSpec:
    patch: Mapping[str, int]       # e.g. {"time": 11, "lat": 240, "lon": 240}
    stride: Mapping[str, int]      # e.g. {"time": 1, "lat": 200, "lon": 200}
    pad_to_cover: bool = True      # add a last window flush to the end so that the full domain is covered

    def __post_init__(self):
        for d in DIMS:
            if d not in self.patch or d not in self.stride:
                raise ValueError(f"patch/stride must define {DIMS}")


def _starts(size: int, patch: int, stride: int, pad_to_cover: bool) -> np.ndarray:
    if size < patch:
        raise ValueError(f"domain size {size} smaller than patch {patch}")
    starts = np.arange(0, size - patch + 1, stride)
    if pad_to_cover and starts[-1] + patch < size:
        starts = np.append(starts, size - patch)
    return starts


class PatchIndex:
    """Enumerates patch windows over a domain; independent of the array contents."""

    def __init__(self, sizes: Mapping[str, int], spec: PatchSpec):
        self.sizes = dict(sizes)
        self.spec = spec
        self.starts = {d: _starts(sizes[d], spec.patch[d], spec.stride[d], spec.pad_to_cover) for d in DIMS}
        self._grid = list(itertools.product(*(range(len(self.starts[d])) for d in DIMS)))

    def __len__(self) -> int:
        return len(self._grid)

    def slices(self, i: int) -> dict[str, slice]:
        idx = self._grid[i]
        return {d: slice(int(self.starts[d][k]), int(self.starts[d][k]) + self.spec.patch[d])
                for d, k in zip(DIMS, idx, strict=True)}

    def __iter__(self) -> Iterator[dict[str, slice]]:
        for i in range(len(self)):
            yield self.slices(i)


class PatchArray:
    """Random access to patches of a lazily-loaded DataArray ``(channel, time, lat, lon)``."""

    def __init__(self, da: xr.DataArray, spec: PatchSpec,
                 norm_stats: tuple[np.ndarray, np.ndarray] | None = None,
                 drop_all_nan_targets: list[int] | None = None,
                 jitter: bool = False, augmentations: list | None = None, seed: int | None = None):
        """``jitter=True`` shifts each window by a random offset in ``[0, stride)`` at every access
        (moving patches: NOSC ``contrib/moving_patches``); ``augmentations`` are applied before
        normalisation. Both are meant for the train split only."""
        if tuple(da.dims) != ("channel", *DIMS):
            raise ValueError(f"expected dims ('channel', 'time', 'lat', 'lon'), got {da.dims}")
        self.da = da
        self.spec = spec
        self.index = PatchIndex({d: da.sizes[d] for d in DIMS}, spec)
        self.norm_stats = norm_stats
        self.jitter = jitter
        self.augmentations = list(augmentations or [])
        self._rng = np.random.default_rng(seed)
        self._valid = list(range(len(self.index)))
        if drop_all_nan_targets:
            self._valid = [i for i in self._valid if self._has_target(i, drop_all_nan_targets)]

    def _has_target(self, i: int, target_idx: list[int]) -> bool:
        sl = self.index.slices(i)
        sub = self.da.isel(channel=target_idx, **sl).values
        return bool(np.isfinite(sub).any())

    def __len__(self) -> int:
        return len(self._valid)

    def slices(self, i: int) -> dict[str, slice]:
        return self.index.slices(self._valid[i])

    def coords(self, i: int) -> dict[str, np.ndarray]:
        sl = self.slices(i)
        return {d: self.da[d].values[sl[d]] for d in DIMS}

    def _jittered(self, sl: dict[str, slice]) -> dict[str, slice]:
        out = {}
        for d, s in sl.items():
            room = self.da.sizes[d] - s.stop
            off = int(self._rng.integers(0, min(self.spec.stride[d], room) + 1)) if room > 0 else 0
            out[d] = slice(s.start + off, s.stop + off)
        return out

    def __getitem__(self, i: int) -> np.ndarray:
        sl = self._jittered(self.slices(i)) if self.jitter else self.slices(i)
        item = self.da.isel(**sl).values.astype(np.float32)
        for aug in self.augmentations:
            item = aug(item, self._rng)
        if self.norm_stats is not None:
            mean, std = self.norm_stats
            item = (item - mean[:, None, None, None]) / std[:, None, None, None]
        return item

    def reconstruct(self, items: np.ndarray, weight: np.ndarray | None = None,
                    variable_names: list[str] | None = None) -> xr.DataArray:
        """Merge patches ``(n_patches, n_vars, time, lat, lon)`` back onto the domain.

        Overlapping regions are averaged with ``weight`` (shape ``(time, lat, lon)``),
        typically :func:`oceanml3d.training.weights.patch_weight`.
        """
        n_vars = items.shape[1]
        full_shape = (n_vars, *(self.da.sizes[d] for d in DIMS))
        acc = np.zeros(full_shape, dtype=np.float64)
        cnt = np.zeros(full_shape, dtype=np.float64)
        w = np.ones(items.shape[2:], dtype=np.float64) if weight is None else np.asarray(weight, dtype=np.float64)
        for i, item in enumerate(items):
            sl = self.slices(i)
            region = (slice(None), sl["time"], sl["lat"], sl["lon"])
            acc[region] += np.nan_to_num(item) * w
            cnt[region] += w * np.isfinite(item)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = acc / cnt
        names = variable_names or [f"v{i}" for i in range(n_vars)]
        return xr.DataArray(
            out.astype(np.float32), dims=("channel", *DIMS),
            coords={"channel": names, **{d: self.da[d] for d in DIMS}},
        )
