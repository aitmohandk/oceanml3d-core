"""4DVarNet: iterative variational solver with a learned prior (bilinear AE) and a learned
gradient step (ConvLSTM). Port of 4dvarnet-starter / NOSC ``src/models.py``
(``GradSolverZero``, ``BilinAEPriorCost``, ``BaseObsCost``, ``ConvLstmGradModel``) adapted to the
``VariableSet`` contract, so it is the "GradSolver ablation" of the OSSE-3D study.

State = the target channels ``(B, n_targets * T, H, W)``. Observations = input channels
mapped onto targets through ``obs_map`` (``{target: input}``; identity for ``role: both``
variables). Other inputs (SST, ARGO, statics) condition the gradient model.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from oceanml3d.models.ocean.base import BaseOceanModel
from oceanml3d.registry import register_model
from oceanml3d.variables import VariableSet


class BilinAEPriorCost(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int, kernel_size: int = 3, downsamp: int | None = None, bilin_quad: bool = True):
        super().__init__()
        p = kernel_size // 2
        self.bilin_quad = bilin_quad
        self.conv_in = nn.Conv2d(dim_in, dim_hidden, kernel_size, padding=p)
        self.conv_hidden = nn.Conv2d(dim_hidden, dim_hidden, kernel_size, padding=p)
        self.bilin_1 = nn.Conv2d(dim_hidden, dim_hidden, kernel_size, padding=p)
        self.bilin_21 = nn.Conv2d(dim_hidden, dim_hidden, kernel_size, padding=p)
        self.bilin_22 = nn.Conv2d(dim_hidden, dim_hidden, kernel_size, padding=p)
        self.conv_out = nn.Conv2d(2 * dim_hidden, dim_in, kernel_size, padding=p)
        self.down = nn.AvgPool2d(downsamp) if downsamp else nn.Identity()
        self.up = nn.UpsamplingBilinear2d(scale_factor=downsamp) if downsamp else nn.Identity()

    def forward_ae(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_hidden(F.relu(self.conv_in(self.down(x))))
        nonlin = self.bilin_21(x) ** 2 if self.bilin_quad else self.bilin_21(x) * self.bilin_22(x)
        return self.up(self.conv_out(torch.cat([self.bilin_1(x), nonlin], dim=1)))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(state, self.forward_ae(state))


class ConvLstmGradModel(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int, dim_cond: int = 0, kernel_size: int = 3, dropout: float = 0.1, downsamp: int | None = None):
        super().__init__()
        p = kernel_size // 2
        self.dim_hidden = dim_hidden
        self.gates = nn.Conv2d(dim_in + dim_cond + dim_hidden, 4 * dim_hidden, kernel_size, padding=p)
        self.conv_out = nn.Conv2d(dim_hidden, dim_in, kernel_size, padding=p)
        self.dropout = nn.Dropout(dropout)
        self.down = nn.AvgPool2d(downsamp) if downsamp else nn.Identity()
        self.up = nn.UpsamplingBilinear2d(scale_factor=downsamp) if downsamp else nn.Identity()
        self._state: list[torch.Tensor] = []
        self._grad_norm = None

    def reset_state(self, ref: torch.Tensor) -> None:
        size = [ref.shape[0], self.dim_hidden, *ref.shape[-2:]]
        self._grad_norm = None
        self._state = [self.down(torch.zeros(size, device=ref.device)), self.down(torch.zeros(size, device=ref.device))]

    def forward(self, grad: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        if self._grad_norm is None:
            self._grad_norm = (grad ** 2).mean().sqrt().clamp_min(1e-8)
        x = self.down(self.dropout(grad / self._grad_norm))
        if cond is not None:
            x = torch.cat([x, self.down(cond)], dim=1)
        hidden, cell = self._state
        i, f, o, g = self.gates(torch.cat([x, hidden], dim=1)).chunk(4, dim=1)
        cell = torch.sigmoid(f) * cell + torch.sigmoid(i) * torch.tanh(g)
        hidden = torch.sigmoid(o) * torch.tanh(cell)
        self._state = [hidden, cell]
        return self.up(self.conv_out(hidden))


@register_model("fourdvarnet")
class FourDVarNet(BaseOceanModel):
    def __init__(self, variables: VariableSet, window: int, rec_weight: np.ndarray,
                 obs_map: dict[str, str] | None = None, n_step: int = 15, lr_grad: float = 0.2,
                 prior_hidden: int = 64, grad_hidden: int = 96, downsamp: int | None = None,
                 obs_weight: float = 1.0, prior_weight: float = 1.0, init_from_obs: bool = True,
                 dropout: float = 0.1, **base_kw):
        super().__init__(variables, window, rec_weight, **base_kw)
        self.save_hyperparameters(ignore=["variables", "rec_weight", "norm_stats"])
        tgt = [s.name for s in variables.targets]
        obs_map = dict(obs_map or {})
        for s in variables.targets:                    # role: both -> observed by itself (masked input)
            if s.role.value == "both":
                obs_map.setdefault(s.name, s.name)
        if not obs_map:
            raise ValueError("fourdvarnet needs at least one observed target: obs_map {target: input} or role: both")
        in_names = [s.name for s in variables.inputs]
        self.obs_pairs = [(tgt.index(t), in_names.index(o)) for t, o in obs_map.items()]
        self.cond_idx = [i for i, n in enumerate(in_names) if n not in obs_map.values()]
        n_state = len(tgt) * window
        self.n_step, self.lr_grad = n_step, lr_grad
        self.obs_weight, self.prior_weight, self.init_from_obs = obs_weight, prior_weight, init_from_obs
        self.prior_cost = BilinAEPriorCost(n_state, prior_hidden, downsamp=downsamp)
        self.grad_mod = ConvLstmGradModel(n_state, grad_hidden, len(self.cond_idx) * window, dropout=dropout, downsamp=downsamp)

    # --- observation term -------------------------------------------------
    def _obs(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(obs, mask) shaped like the state (B, n_targets, T, H, W); NaN where unobserved."""
        raw = batch.index_select(1, self._in_idx.to(batch.device))
        b, _, t, h, w = raw.shape
        obs = torch.full((b, len(self.variables.targets), t, h, w), float("nan"), device=batch.device, dtype=batch.dtype)
        for ti, oi in self.obs_pairs:
            obs[:, ti] = raw[:, oi]
        return obs, torch.isfinite(obs)

    def obs_cost(self, state: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.sum() == 0:
            return state.sum() * 0.0
        return self.obs_weight * F.mse_loss(state[mask], obs.nan_to_num()[mask])

    # --- solver -----------------------------------------------------------
    def solve(self, batch: torch.Tensor) -> torch.Tensor:
        obs, mask = self._obs(batch)
        b, n, t, h, w = obs.shape
        obs2, mask2 = obs.reshape(b, n * t, h, w), mask.reshape(b, n * t, h, w)
        cond = self.inputs(batch)[:, self.cond_idx].reshape(b, -1, h, w) if self.cond_idx else None
        with torch.set_grad_enabled(True):
            state = (obs2.nan_to_num() if self.init_from_obs else torch.zeros_like(obs2)).detach().requires_grad_(True)
            self.grad_mod.reset_state(state)
            for step in range(self.n_step):
                cost = self.prior_weight * self.prior_cost(state) + self.obs_cost(state, obs2, mask2)
                grad = torch.autograd.grad(cost, state, create_graph=self.training)[0]
                update = self.grad_mod(grad, cond) / (step + 1) + self.lr_grad * (step + 1) / self.n_step * grad
                state = state - update
                if not self.training:
                    state = state.detach().requires_grad_(True)
            if not self.training:
                state = self.prior_cost.forward_ae(state)
        return state.reshape(b, n, t, h, w)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.solve(batch)

    def step(self, batch: torch.Tensor, phase: str) -> torch.Tensor:
        out = self(batch)
        tgt = self.targets(batch)
        total = self.combine_losses(self.per_target_losses(out, tgt, phase), phase)
        b, n, t, h, w = out.shape
        prior = self.prior_cost(out.reshape(b, n * t, h, w))
        self.log(f"{phase}/prior_cost", prior, on_step=False, on_epoch=True, sync_dist=True)
        total = total + self.prior_weight * prior
        self.log(f"{phase}/loss", total, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return total
