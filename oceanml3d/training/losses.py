"""Losses.

Two families, kept in one module because they are the same concept at two scales:

* :class:`StateMSELoss` — the toy/Lorenz path (`training/lightning_module.py`), a plain MSE on a
  ``(B, T, K)`` state with an optional finite-difference gradient term along time.
* ``weighted_mae`` / ``weighted_mse`` / ``gradient_loss`` — the gridded-ocean path
  (``models/ocean/base.py``), ported from NOSC. They take a ``(T, H, W)`` weight broadcast over
  batch and variable, and they ignore non-finite targets: an ocean target field is mostly NaN
  (drifter tracks, satellite ground tracks), so the loss has to be defined on the observed subset
  rather than on the full grid.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class StateMSELoss(nn.Module):
    def __init__(self, use_gradient_loss: bool = False, gradient_weight: float = 0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.use_gradient_loss = use_gradient_loss
        self.gradient_weight = gradient_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        loss = self.mse(pred, target)
        if self.use_gradient_loss and pred.shape[1] > 1:
            pred_grad = pred[:, 1:] - pred[:, :-1]
            target_grad = target[:, 1:] - target[:, :-1]
            loss = loss + self.gradient_weight * self.mse(pred_grad, target_grad)
        return loss


def weighted_mae(err: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over finite entries, weighted by ``weight`` (T, H, W)."""
    mask = torch.isfinite(err)
    w = weight.to(err.device)
    num = (err.abs().nan_to_num() * w * mask).sum()
    den = (w * mask).sum().clamp_min(1.0)
    return num / den


def weighted_mse(err: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(err)
    w = weight.to(err.device)
    num = (err.nan_to_num() ** 2 * w * mask).sum()
    den = (w * mask).sum().clamp_min(1.0)
    return num / den


def get_loss(name: str):
    return {"mae": weighted_mae, "mse": weighted_mse}[name]


def sobel(x: torch.Tensor) -> torch.Tensor:
    """Sobel gradient magnitude of a ``(..., H, W)`` tensor (replaces kornia.filters.sobel)."""
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=x.device, dtype=x.dtype)
    shape = x.shape
    flat = x.reshape(-1, 1, *shape[-2:])
    gx = torch.nn.functional.conv2d(flat, kx[None, None], padding=1)
    gy = torch.nn.functional.conv2d(flat, kx.t()[None, None], padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-12).reshape(shape)


def gradient_loss(out: torch.Tensor, tgt: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weighted MSE between Sobel gradients (fine-scale term); NaN targets ignored."""
    diff = sobel(out) - sobel(tgt.nan_to_num())
    diff = torch.where(torch.isfinite(tgt), diff, torch.full_like(diff, float("nan")))
    return weighted_mse(diff, weight)
