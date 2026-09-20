"""Generic Lightning DataModule for gridded ocean data."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from oceanml3d.catalog import Catalog
from oceanml3d.data.open import compute_norm_stats, open_variable_set
from oceanml3d.data.patches import PatchArray, PatchSpec
from oceanml3d.variables import VariableSet


def to_slice(v: Any) -> slice:
    if isinstance(v, slice):
        return v
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return slice(*v)
    if isinstance(v, Mapping):
        return slice(v.get("start"), v.get("stop"))
    raise TypeError(f"cannot convert {v!r} to slice")


class PatchDataset(Dataset):
    def __init__(self, patches: PatchArray):
        self.patches = patches

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, i: int) -> torch.Tensor:
        return torch.from_numpy(self.patches[i])


class EvalPatchDataset(PatchDataset):
    """Validation / test patches, each paired with the ``(lat, lon)`` mask of ``eval_domain``.

    Patches are positional -- a batch carries no coordinates -- so the metric mask has to travel
    with the patch. Without ``eval_domain`` the mask is all ones. ``predict`` keeps the plain
    :class:`PatchDataset`: the export stitches the whole domain.
    """

    def __init__(self, patches: PatchArray, eval_domain: Mapping[str, slice] | None = None):
        super().__init__(patches)
        self.eval_domain = eval_domain or {}

    def mask(self, i: int) -> np.ndarray:
        coords = self.patches.coords(i)
        keep = np.ones((coords["lat"].size, coords["lon"].size), dtype=bool)
        for dim, axis in (("lat", 0), ("lon", 1)):
            sl = self.eval_domain.get(dim)
            if sl is None:
                continue
            c = coords[dim]
            inside = np.ones(c.size, bool)
            if sl.start is not None:
                inside &= c >= sl.start
            if sl.stop is not None:
                inside &= c <= sl.stop
            keep &= inside[:, None] if axis == 0 else inside[None, :]
        return keep.astype(np.float32)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return super().__getitem__(i), torch.from_numpy(self.mask(i))


class OceanDataModule(LightningDataModule):
    """Windows of ``(n_vars, T, H, W)`` sampled from a multi-variable lazily loaded field.

    Parameters mirror the ``datamodule`` block of NOSC experiments but with catalog keys
    instead of paths and a typed VariableSet instead of ad-hoc dicts.
    """

    def __init__(self, variables: VariableSet, catalog: Catalog,
                 domain: Mapping[str, Any], splits: Mapping[str, Any],
                 patch: Mapping[str, int], stride: Mapping[str, int],
                 batch_size: int = 1, num_workers: int = 4,
                 norm_stats: tuple[list[float], list[float]] | None = None,
                 chunks: Mapping[str, int] | None = None,
                 drop_empty_target_patches: bool = True,
                 cache: str | None = None, jitter: bool = False, augmentations: list | None = None,
                 eval_domain: Mapping[str, Any] | None = None):
        """``cache``: zarr store path; the stacked ``(channel, time, lat, lon)`` array is written once
        and re-opened from there (fast random access, bounded RAM, no eager load).
        ``eval_domain``: the sub-domain the val/test metrics are computed on (rim excluded)."""
        super().__init__()
        self.variables = variables
        self.catalog = catalog
        self.domain = {k: to_slice(v) for k, v in domain.items()}
        self.splits = {k: to_slice(v["time"] if isinstance(v, Mapping) else v) for k, v in splits.items()}
        self.spec = PatchSpec(dict(patch), dict(stride))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._norm_stats = None if norm_stats is None else (np.asarray(norm_stats[0], np.float32),
                                                            np.asarray(norm_stats[1], np.float32))
        self.chunks = chunks
        self.drop_empty = drop_empty_target_patches
        self.cache = cache
        self.jitter = jitter
        self.augmentations = augmentations or []
        self.eval_domain = {k: to_slice(v) for k, v in (eval_domain or {}).items()}
        self.da = None
        self.datasets: dict[str, PatchDataset] = {}

    def setup(self, stage: str | None = None) -> None:
        if self.da is None:
            starts = [s.start for s in self.splits.values()]
            stops = [s.stop for s in self.splits.values()]
            domain = {**self.domain, "time": slice(min(starts), max(stops))}
            self.da = open_variable_set(self.variables, self.catalog, domain, self.chunks)
            if self.cache:
                self.da = _cached(self.da, self.cache)
            if self._norm_stats is None:
                self._norm_stats = compute_norm_stats(self.da, self.splits["train"])
        drop = self.variables.target_indices if self.drop_empty else None
        for split, tsl in self.splits.items():
            sub = self.da.sel(time=tsl)
            train = split == "train"
            self.datasets[split] = PatchDataset(PatchArray(sub, self.spec, self._norm_stats,
                                                           drop_all_nan_targets=drop if train else None,
                                                           jitter=self.jitter and train,
                                                           augmentations=self.augmentations if train else None))
        self.eval_datasets = {split: EvalPatchDataset(ds.patches, self.eval_domain)
                              for split, ds in self.datasets.items() if split != "train"}

    def norm_stats(self) -> tuple[np.ndarray, np.ndarray]:
        return self._norm_stats

    def _loader(self, split: str, shuffle: bool, evaluation: bool = False) -> DataLoader:
        ds = self.eval_datasets[split] if evaluation else self.datasets[split]
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, pin_memory=True)

    def train_dataloader(self):
        return self._loader("train", True)

    def val_dataloader(self):
        """``(patch, eval_mask)`` batches: see :class:`EvalPatchDataset`."""
        return self._loader("val", False, evaluation=True)

    def test_dataloader(self):
        return self._loader("test", False, evaluation=True)

    def predict_dataloader(self):
        return self._loader("test", False)


def _cached(da, path: str):
    """Write the stacked array to zarr once (chunked per time step), then re-open lazily."""
    import os

    import xarray as xr

    if not os.path.exists(path):
        da.chunk({"channel": -1, "time": 1}).to_dataset(name="stack").to_zarr(path, mode="w")
        print(f"[oceanml3d] cached stacked array to {path}")
    return xr.open_zarr(path)["stack"]
