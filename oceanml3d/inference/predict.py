"""Patch-wise prediction, stitching and export (replaces ``only_rec`` + concat_rec*.py in NOSC)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from pytorch_lightning import Callback, Trainer

from oceanml3d.data.datamodule import OceanDataModule
from oceanml3d.inference.export import write_product
from oceanml3d.models.ocean.base import BaseOceanModel


class _FoldIn(Callback):
    """Folds each predicted batch into the accumulator and drops it.

    ``on_predict_batch_end`` sees batches in loader order, and the predict loader is built with
    ``shuffle=False``, so batch ``b`` holds patches ``b * batch_size ...``. Lightning keeps
    handling device placement and precision; only the collecting is taken away from it.
    """

    def __init__(self, accumulator, batch_size: int):
        self.accumulator = accumulator
        self.batch_size = int(batch_size)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.accumulator.add_batch(batch_idx * self.batch_size,
                                   outputs.detach().float().cpu().numpy())


def _single_device_trainer(trainer: Trainer, callback: Callback) -> Trainer:
    """A one-device Trainer for the prediction pass, borrowing the caller's accelerator/precision.

    Prediction must not be sharded: ``trainer.predict`` splits the dataloader across ranks, and a
    field stitched from one rank's share is wrong without being obviously wrong. The accumulator
    would refuse to produce it, so this is belt and braces -- but it also means `training=ddp` runs
    end to end instead of erroring at the last step.
    """
    kw: dict = {"devices": 1, "logger": False, "enable_checkpointing": False,
                "enable_progress_bar": False, "callbacks": [callback]}
    for name in ("accelerator", "precision"):
        value = getattr(trainer, name, None)
        if value is not None:
            kw[name] = value
    # `inference_mode` matters and Lightning does not expose it publicly. It must be carried over:
    # the 4DVarNet solver differentiates inside its forward pass, and `config/training/default.yaml`
    # sets `inference_mode: false` for exactly that reason. Defaulting to Lightning's `True` here
    # would make its export fail with "inference tensors cannot be saved for backward".
    for name in ("inference_mode", "_inference_mode"):
        value = getattr(trainer, name, None)
        if isinstance(value, bool):
            kw["inference_mode"] = value
            break
    return Trainer(**kw)


def predict_field(model: BaseOceanModel, dm: OceanDataModule, trainer: Trainer, split: str = "test",
                  weight: np.ndarray | None = None, dtype: np.dtype | str = np.float64):
    """Run the model over every patch of ``split`` and stitch a ``(channel, time, lat, lon)`` field.

    Predictions are folded into the output field batch by batch rather than concatenated first. The
    old ``torch.cat(trainer.predict(...))`` held every patch at once: for the ``surface_currents_15m``
    test split that is 377 patches x 2 targets x 11 x 560 x 1440 float32, about 27 GB, on top of the
    accumulators. Peak memory is now the size of the output field, whatever the number of patches.
    """
    dm.setup("predict")
    loader = dm._loader(split, shuffle=False)
    patches = dm.datasets[split].patches
    names = [s.name for s in model.variables.targets]
    w = weight if weight is not None else model.rec_weight.cpu().numpy()

    if getattr(trainer, "world_size", 1) > 1:
        print("[oceanml3d] prediction runs on a single device; the distributed Trainer is used "
              "only for its accelerator and precision settings.")

    acc = patches.accumulator(len(names), w, dtype)
    batch_size = getattr(loader, "batch_size", None) or 1
    _single_device_trainer(trainer, _FoldIn(acc, batch_size)).predict(
        model, dataloaders=loader, return_predictions=False)
    return acc.result(names)


def predict_and_export(model: BaseOceanModel, dm: OceanDataModule, trainer: Trainer, out_dir: str | Path,
                       name: str, split: str = "test", time_slice: slice | None = None,
                       depth_m: float | None = None, attrs: dict | None = None) -> Path:
    field = predict_field(model, dm, trainer, split)
    return write_product(field, model.variables, out_dir, name, time_slice, attrs, depth_m)
