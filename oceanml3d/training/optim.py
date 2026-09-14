"""Optimizer factories (referenced by name from the config)."""
from __future__ import annotations

import torch


def cosine_adam(params, lr: float, t_max: int, weight_decay: float = 0.0) -> dict:
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    return {"optimizer": opt, "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max)}


def adamw_plateau(params, lr: float, weight_decay: float = 1e-5, patience: int = 10, **_) -> dict:
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=patience)
    return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val/loss"}}


OPTIMIZERS = {"cosine_adam": cosine_adam, "adamw_plateau": adamw_plateau}
