"""Adapter exposing MONAI's DiffusionModelUNet with the same call signature as
models.unet.UNet1D, for prototyping (see /homes/rfablet/.claude/plans/monai-diffunet-prototype.md).

monai is now a hard dependency, pinned `>=1.5,<1.6`. The torch conflict this
module's header used to describe is real but belongs to monai==1.6.0, which
requires torch>=2.8.0; monai 1.5.x pins torch>=2.4.1,<2.7.0 and installs into
the project env without moving torch off 2.4.1 (measured, 2026-09-14).
requirements-monai.txt is kept only to reproduce this 1-D prototype on the
1.6/torch-2.8 combination.

Verified directly against the installed monai==1.6.0 wheel, and re-checked
against 1.5.2:
- DiffusionModelUNet.forward(x, timesteps, context=None, class_labels=None, ...)
  takes a raw (N,) float tensor for `timesteps` and embeds it with a plain
  sinusoidal encoding (args = timesteps * freqs, max_period=10000) -- the same
  formula as this repo's own SinusoidalEmbedding. A continuous tau in [0, 1]
  is passed through directly; no num_train_timesteps / integer-step config
  needed.
- Constructor kwarg is `channels`, not `num_channels`.

Upstream bug patched here (not a MONAI config issue): monai.networks.nets.
diffusion_model_unet.DiffusionUNetResnetBlock.forward hardcodes the temb
broadcast as `[:, :, None, None]` for spatial_dims == 2 and unconditionally
`[:, :, None, None, None]` otherwise -- there is no spatial_dims == 1 branch,
so with spatial_dims=1 the addition broadcasts incorrectly and corrupts the
output shape (confirmed: forward crashes/produces garbage without this
patch). The patch itself now lives with the 2-D trunk
(oceanml3d.models.ocean.nn.unet_monai.patch_diffusion_resblock) because both
need it -- it also teaches the block the `dropout` submodule attached below.
"""

import torch
import torch.nn as nn
from monai.networks.nets import DiffusionModelUNet
from monai.networks.nets import diffusion_model_unet as _dmu

from oceanml3d.models.ocean.nn.unet_monai import patch_diffusion_resblock


class MonaiUNet1D(nn.Module):
    """Drop-in-shaped replacement for models.unet.UNet1D backed by MONAI's
    DiffusionModelUNet, restricted to spatial_dims=1 with attention disabled
    (see plan: MONAI's attention reshape has no spatial_dims==1 branch
    either -- a second, separate gap from the ResBlock bug patched above).

    Conditioning: obs is concatenated onto the state along the channel axis
    before the backbone call (matching FDV's channel-concat convention),
    rather than using UNet1D's separate ConditionEncoder-style branch.
    """

    def __init__(
        self,
        state_dim: int = 3,
        obs_dim: int = None,
        hidden_channels: list = None,
        num_res_blocks: int = 2,
        norm_num_groups: int = 8,
        use_obs: bool = True,
        output_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        patch_diffusion_resblock()

        if hidden_channels is None:
            hidden_channels = [32, 64, 128]
        self.state_dim = state_dim
        self.obs_dim = obs_dim if obs_dim is not None else state_dim
        self.use_obs = use_obs
        self.output_dim = output_dim if output_dim is not None else state_dim

        in_channels = state_dim + (self.obs_dim if use_obs else 0)

        self.backbone = DiffusionModelUNet(
            spatial_dims=1,
            in_channels=in_channels,
            out_channels=self.output_dim,
            channels=tuple(hidden_channels),
            attention_levels=tuple(False for _ in hidden_channels),
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
        )
        if dropout > 0:
            for module in self.backbone.modules():
                if isinstance(module, _dmu.DiffusionUNetResnetBlock):
                    module.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        obs: torch.Tensor = None,
        x_tau: torch.Tensor = None,
        tau: torch.Tensor = None,
        energy_terms: list = None,
    ) -> torch.Tensor:
        B = x.shape[0]
        inp = x
        if self.use_obs:
            if obs is None:
                raise ValueError("use_obs=True but obs is None")
            inp = torch.cat([x, obs], dim=1)
        timesteps = tau if tau is not None else torch.zeros(B, device=x.device, dtype=x.dtype)
        return self.backbone(inp, timesteps)


class MonaiDirectUNet(nn.Module):
    """MonaiUNet1D-backed drop-in for models.direct_unet.DirectUNet: same
    FlowMatchingBatch-taking forward() (obs/forcing/params -> single-pass
    state regression via a zeroed state input + channel-concat conditioning),
    so it plugs into train.py's model_type dispatch (model_factory /
    evaluate_model / save_trajectories) exactly where "direct_unet" does.
    """

    def __init__(self, state_dim=24, hidden_channels=None, dropout=0.1, param_dim=0,
                 cond_extra_dim=0, num_res_blocks=2, norm_num_groups=32):
        super().__init__()
        self.state_dim = state_dim
        self.param_dim = param_dim
        self.cond_extra_dim = cond_extra_dim
        if hidden_channels is None:
            hidden_channels = [64, 128, 256]
        self.monai_unet = MonaiUNet1D(
            state_dim=state_dim,
            obs_dim=state_dim + cond_extra_dim,
            hidden_channels=hidden_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            use_obs=True,
            dropout=dropout,
        )

    def forward(self, batch):
        obs = batch.obs
        forcing = batch.forcing
        params = batch.params
        B, T, D = obs.shape
        obs_clean = torch.nan_to_num(obs, nan=0.0)
        if self.cond_extra_dim > 0:
            cond = torch.cat([obs_clean, forcing.unsqueeze(-1)], dim=-1)
            if self.param_dim > 0:
                params_t = params.unsqueeze(1).expand(B, T, -1)
                cond = torch.cat([cond, params_t], dim=-1)
        else:
            cond = obs_clean
        x = torch.zeros(B, D, T, device=obs.device)
        tau = torch.zeros(B, device=obs.device)
        out = self.monai_unet(x, obs=cond.transpose(1, 2), tau=tau)
        return out.transpose(1, 2)
