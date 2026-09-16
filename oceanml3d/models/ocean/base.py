"""Base class every ocean model derives from.

The contract is deliberately small so that very different approaches (direct U-Net,
4DVarNet solver, flow matching, ...) fit in:

* ``forward(batch)`` receives the normalised patch tensor ``(B, n_vars, T, H, W)``
  and returns the prediction for the *target* variables ``(B, n_targets, T, H, W)``;
* ``training_step`` / ``validation_step`` are provided (weighted loss on target channels);
* ``predict_step`` returns *denormalised* predictions so that :mod:`oceanml3d.inference`
  can stitch and export them without knowing anything about the model.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from pytorch_lightning import LightningModule

from oceanml3d.training.loss_grouping import combine_grouped_losses
from oceanml3d.training.losses import get_loss, gradient_loss
from oceanml3d.training.optim import OPTIMIZERS
from oceanml3d.variables import VariableSet


class BaseOceanModel(LightningModule):
    """Contract every model implements.

    ``stages``/``set_stage`` support multi-stage training (4dvarnet-fm-opencode's stage-1 prior /
    stage-2 conditional pipeline): a model declares the stages it needs, the CLI runs one fit per
    stage and calls ``set_stage`` in between, so a generative model (flow matching, score-based)
    fits in without a second training loop.
    """

    registry_name: str = "base"
    stages: tuple[str, ...] = ("main",)

    def __init__(self, variables: VariableSet, window: int,
                 rec_weight: np.ndarray, loss: str = "mae",
                 optimizer: str = "cosine_adam", optimizer_kw: dict[str, Any] | None = None,
                 norm_stats: tuple[np.ndarray, np.ndarray] | None = None,
                 loss_combine: str = "flat_sum", loss_group_weights: dict[str, float] | None = None,
                 grad_loss_weight: float = 0.0):
        """``loss_combine``: flat_sum | group_mean (groups = VariableSpec.group) | uncertainty
        (learned per-target log-variance, Kendall et al. 2018). ``grad_loss_weight`` adds a Sobel
        gradient MSE term per target (fine-scale preservation)."""
        super().__init__()
        self.variables = variables
        self.window = window
        self.register_buffer("rec_weight", torch.as_tensor(rec_weight, dtype=torch.float32))
        self.loss_fn = get_loss(loss)
        self.optimizer_name = optimizer
        self.optimizer_kw = optimizer_kw or {}
        self.norm_stats = norm_stats
        self._in_idx = torch.as_tensor(variables.input_indices)
        self._tgt_idx = torch.as_tensor(variables.target_indices)
        self.loss_combine = loss_combine
        self.loss_group_weights = dict(loss_group_weights or {})
        self.grad_loss_weight = float(grad_loss_weight)
        self.log_vars = torch.nn.Parameter(torch.zeros(len(variables.targets))) if loss_combine == "uncertainty" else None

    # -- checkpoint ----------------------------------------------------------
    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Carry the normalisation statistics and the channel layout in the checkpoint.

        ``predict_step`` denormalises with ``self.norm_stats``. Those numbers come from the *train*
        split, so a prediction run that recomputes them from whatever splits, domain or variable
        list the current config happens to declare will denormalise with statistics the model was
        never trained under -- and write a product that looks entirely normal. Storing them next to
        the weights makes the pair inseparable. Plain lists, so the checkpoint still loads under
        ``weights_only=True``.
        """
        if self.norm_stats is not None:
            mean, std = self.norm_stats
            checkpoint["oceanml3d"] = {
                "norm_stats": {"mean": np.asarray(mean).tolist(), "std": np.asarray(std).tolist()},
                "variables": list(self.variables.names),
            }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        meta = checkpoint.get("oceanml3d")
        if not meta:
            return
        names = list(meta.get("variables") or [])
        if names and names != list(self.variables.names):
            raise ValueError(
                "the checkpoint was trained on a different channel layout:\n"
                f"  checkpoint: {names}\n  config:     {list(self.variables.names)}"
            )
        ns = meta.get("norm_stats")
        if ns:
            self.norm_stats = (np.asarray(ns["mean"], dtype=np.float32),
                               np.asarray(ns["std"], dtype=np.float32))

    # -- helpers -------------------------------------------------------------
    def inputs(self, batch: torch.Tensor) -> torch.Tensor:
        """(B, n_inputs, T, H, W) with NaN replaced by 0 (masked observations)."""
        return batch.index_select(1, self._in_idx.to(batch.device)).nan_to_num()

    def targets(self, batch: torch.Tensor) -> torch.Tensor:
        """(B, n_targets, T, H, W), NaN kept (loss ignores them)."""
        return batch.index_select(1, self._tgt_idx.to(batch.device))

    # -- multi-stage training -------------------------------------------------
    def set_stage(self, stage: str) -> None:
        """Called before each ``trainer.fit``. Override to freeze/unfreeze sub-modules or switch loss."""
        if stage not in self.stages:
            raise ValueError(f"{type(self).__name__} declares stages {self.stages}, got '{stage}'")
        self.current_stage = stage

    # -- to implement --------------------------------------------------------
    def forward(self, batch: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- lightning -----------------------------------------------------------
    def per_target_losses(self, out: torch.Tensor, tgt: torch.Tensor, phase: str) -> list[torch.Tensor]:
        losses = []
        for i, spec in enumerate(self.variables.targets):
            li = self.loss_fn(out[:, i] - tgt[:, i], self.rec_weight)
            if self.grad_loss_weight > 0:
                gi = gradient_loss(out[:, i], tgt[:, i], self.rec_weight)
                self.log(f"{phase}/{spec.name}_gloss", gi, on_step=False, on_epoch=True, sync_dist=True)
                li = li + self.grad_loss_weight * gi
            self.log(f"{phase}/{spec.name}_loss", li, on_step=False, on_epoch=True, sync_dist=True)
            losses.append(li)
        return losses

    def combine_losses(self, losses: list[torch.Tensor], phase: str) -> torch.Tensor:
        names = [s.name for s in self.variables.targets]
        if self.loss_combine == "uncertainty":
            stacked = torch.stack(losses)
            weighted = torch.exp(-self.log_vars) * stacked + self.log_vars
            for n, lv in zip(names, self.log_vars, strict=True):
                self.log(f"{phase}/{n}_log_var", lv, on_step=False, on_epoch=True, sync_dist=True)
            return weighted.sum()
        groups = {s.name: s.group_name for s in self.variables.targets}
        total, per_group = combine_grouped_losses(losses, names, groups, self.loss_combine, self.loss_group_weights)
        if self.loss_combine == "group_mean":
            for g, gl in per_group.items():
                self.log(f"{phase}/group_{g}_loss", gl, on_step=False, on_epoch=True, sync_dist=True)
        return total

    def step(self, batch: torch.Tensor, phase: str) -> torch.Tensor:
        out = self(batch)
        tgt = self.targets(batch)
        total = self.combine_losses(self.per_target_losses(out, tgt, phase), phase)
        self.log(f"{phase}/loss", total, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return total

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> torch.Tensor:
        out = self(batch)
        if self.norm_stats is None:
            return out
        mean, std = self.norm_stats
        idx = self.variables.target_indices
        m = torch.as_tensor(mean[idx], device=out.device)[None, :, None, None, None]
        s = torch.as_tensor(std[idx], device=out.device)[None, :, None, None, None]
        return out * s + m

    def configure_optimizers(self):
        return OPTIMIZERS[self.optimizer_name](self.parameters(), **self.optimizer_kw)
