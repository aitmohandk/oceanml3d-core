"""Patch-wise prediction, stitching and export (replaces ``only_rec`` + concat_rec*.py in NOSC)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import Trainer

from oceanml3d.data.datamodule import OceanDataModule
from oceanml3d.inference.export import write_product
from oceanml3d.models.ocean.base import BaseOceanModel


def predict_field(model: BaseOceanModel, dm: OceanDataModule, trainer: Trainer, split: str = "test",
                  weight: np.ndarray | None = None):
    """Run the model over every patch of ``split`` and stitch a ``(channel, time, lat, lon)`` field."""
    dm.setup("predict")
    loader = dm._loader(split, shuffle=False)
    preds = trainer.predict(model, dataloaders=loader)
    items = torch.cat(preds).float().cpu().numpy()  # (n_patches, n_targets, T, H, W)
    w = weight if weight is not None else model.rec_weight.cpu().numpy()
    names = [s.name for s in model.variables.targets]
    return dm.datasets[split].patches.reconstruct(items, w, names)


def predict_and_export(model: BaseOceanModel, dm: OceanDataModule, trainer: Trainer, out_dir: str | Path,
                       name: str, split: str = "test", time_slice: slice | None = None,
                       depth_m: float | None = None, attrs: dict | None = None) -> Path:
    field = predict_field(model, dm, trainer, split)
    return write_product(field, model.variables, out_dir, name, time_slice, attrs, depth_m)
