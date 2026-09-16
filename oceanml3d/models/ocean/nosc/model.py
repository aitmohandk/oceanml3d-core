"""NOSC — Neural Ocean Surface Currents.

Direct U-Net mapping from a window of surface inputs (SSH/geostrophic currents, ERA5
winds, latitude, bathymetry, optional SST) to drifter-derived currents at 0 or 15 m.
Port of ``MultivarUNet_mae`` + ``contrib/4dvarnet_latent/unet.UNetModel`` usage in NOSC.

The same class covers the OSSE-3D setting (branch ``first_implementation``) through options:
``head`` = ``single`` | ``grouped`` (one conv head per target group) | ``vertical_modes`` (EOF
projection per group), ``attention_levels`` (self-attention at coarse levels), and the loss
options of :class:`BaseOceanModel` (``loss_combine``, ``grad_loss_weight``).
"""
from __future__ import annotations

import numpy as np
import torch

from oceanml3d.models.ocean.base import BaseOceanModel
from oceanml3d.models.ocean.nn.heads import GroupedHeads, VerticalModesHead
from oceanml3d.models.ocean.nn.unet_nosc import UNetNosc
from oceanml3d.registry import register_model
from oceanml3d.variables import VariableSet


@register_model("nosc_unet")
class NOSCUNet(BaseOceanModel):
    def __init__(self, variables: VariableSet, window: int, rec_weight: np.ndarray,
                 widths: tuple[int, ...] = (64, 128, 256, 512, 1024), dropout: float = 0.1,
                 bilinear: bool = True, attention_levels: tuple[int, ...] = (), attention_heads: int = 4,
                 head: str = "single", neck_channels: int = 64, head_hidden: int = 32, head_layers: int = 2,
                 mode_specs: dict | None = None, time_mode: str = "channels", temporal_channels: int = 16,
                 trunk: str = "monai", num_res_blocks: int = 2, norm_num_groups: int | None = None, **base_kw):
        """``time_mode``: ``channels`` (days folded into channels, NOSC default) or ``conv3d`` (a 3D
        convolutional stem/head mixes time explicitly before/after the 2D trunk).

        ``trunk``: ``monai`` (MONAI's ``DiffusionModelUNet``, the default) or ``nosc`` (NOSC's own
        residual U-Net, which every run published before the switch used — ``ablation=trunk_nosc``).
        ``num_res_blocks``/``norm_num_groups`` apply to the MONAI trunk only, ``bilinear`` to the
        NOSC trunk only."""
        super().__init__(variables, window, rec_weight, **base_kw)
        self.save_hyperparameters(ignore=["variables", "rec_weight", "norm_stats"])
        self.n_in = len(variables.inputs)
        self.n_out = len(variables.targets)
        groups = variables.target_groups
        if head == "single":
            trunk_out, self.head = self.n_out * window, torch.nn.Identity()
        elif head == "grouped":
            trunk_out = neck_channels
            self.head = GroupedHeads(neck_channels, {g: len(v) * window for g, v in groups.items()}, head_hidden, head_layers)
        elif head == "vertical_modes":
            self.head = VerticalModesHead({g: len(v) for g, v in groups.items()}, window, mode_specs or {})
            trunk_out = self.head.trunk_channels
        else:
            raise ValueError(f"unknown head '{head}' (single|grouped|vertical_modes)")
        self.time_mode = time_mode
        if time_mode == "conv3d":
            self.stem = torch.nn.Sequential(torch.nn.Conv3d(self.n_in, temporal_channels, 3, padding=1), torch.nn.SiLU(),
                                            torch.nn.Conv3d(temporal_channels, temporal_channels, 3, padding=1), torch.nn.SiLU())
            self.temporal_head = torch.nn.Conv3d(self.n_out, self.n_out, (3, 1, 1), padding=(1, 0, 0))
            n_in_trunk = temporal_channels * window
        elif time_mode == "channels":
            n_in_trunk = self.n_in * window
        else:
            raise ValueError(f"unknown time_mode '{time_mode}' (channels|conv3d)")
        self.trunk = trunk
        trunk_kw = dict(attention_levels=tuple(attention_levels), attention_heads=attention_heads)
        if trunk == "monai":
            from oceanml3d.models.ocean.nn.unet_monai import MonaiUNet2d  # optional at import time
            self.net = MonaiUNet2d(n_in_trunk, trunk_out, tuple(widths), bilinear, dropout,
                                   num_res_blocks=num_res_blocks, norm_num_groups=norm_num_groups, **trunk_kw)
        elif trunk == "nosc":
            self.net = UNetNosc(n_in_trunk, trunk_out, tuple(widths), bilinear, dropout, **trunk_kw)
        else:
            raise ValueError(f"unknown trunk '{trunk}' (monai|nosc)")

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Checkpoints written before MONAI became the default carry no ``trunk`` hyperparameter, so
        Lightning rebuilds them as MONAI and ``load_state_dict`` fails on unreadable size mismatches.
        ``net.inc.`` only ever exists in :class:`UNetNosc`, so it identifies them exactly."""
        state = checkpoint.get("state_dict") or {}
        if self.trunk != "nosc" and any(k.startswith("net.inc.") for k in state):
            raise RuntimeError(
                "this checkpoint was trained with the NOSC trunk (its state_dict has 'net.inc.*'), "
                f"but the model was built with trunk='{self.trunk}'. Reload it with "
                "'ablation=trunk_nosc' (CLI) or trunk='nosc' (Python).")
        super().on_load_checkpoint(checkpoint)      # channel layout + normalisation statistics

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        x = self.inputs(batch)                                  # (B, n_in, T, H, W)
        b, _, t, h, w = x.shape
        if self.time_mode == "conv3d":
            x = self.stem(x)
        y = self.head(self.net(x.reshape(b, -1, h, w)))         # time folded into channels for the 2D trunk
        y = y.reshape(b, self.n_out, t, h, w)
        if self.time_mode == "conv3d":
            y = y + self.temporal_head(y)
        return y
