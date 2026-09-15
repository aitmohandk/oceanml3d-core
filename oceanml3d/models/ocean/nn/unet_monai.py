"""MONAI-backed 2D trunk for the gridded models — the default since MONAI replaced NOSC's own
U-Net (:mod:`oceanml3d.models.ocean.nn.unet_nosc`, still reachable as ``trunk: nosc``).

The backbone is ``monai.networks.nets.DiffusionModelUNet`` with ``spatial_dims=2``. It is the only
U-Net MONAI ships that carries per-level self-attention natively, which is what
:class:`~oceanml3d.models.ocean.nn.unet_nosc.UNetNosc` already exposed through ``attention_levels``.
Three things it does *not* do the way this repository's own U-Net did, handled here rather than left
to bite at run time:

* **Dropout.** ``DiffusionUNetResnetBlock`` has no dropout at all — no constructor argument, no
  layer. :func:`patch_diffusion_resblock` teaches its ``forward`` to honour a ``dropout`` submodule,
  and the wrapper attaches one to every block, so ``dropout`` is real regularisation and not a
  silently dropped config key.
* **Spatial size.** The backbone concatenates encoder skips with decoder features, so every spatial
  dimension must survive ``len(widths) - 1`` exact halvings; ``30 x 30`` with three levels raises a
  shape mismatch deep inside MONAI. ``UNetNosc`` tolerated any size (it padded in ``Up``). The
  wrapper restores that tolerance by padding up to the next multiple and cropping the output back.
* **Upsampling.** There is no ``bilinear`` switch; MONAI interpolates (nearest) then convolves.
  ``bilinear`` is accepted and ignored so that the existing ``config/model/nosc_unet.yaml`` keeps
  composing, and says so once per process.

``DiffusionModelUNet.forward`` takes ``timesteps`` positionally. This is a direct regression, so a
zero vector is passed, exactly as the 1-D prototype does. The timestep-embedding MLP is still built
and still adds its (constant) bias to every residual block — harmless, and cheaper to leave in place
than to surgically remove.

One upstream initialisation choice is kept deliberately and is worth knowing: MONAI wraps both the
final convolution and every residual block's second convolution in ``zero_module``, so a freshly
built trunk is *exactly* the zero map — every residual branch starts as an identity and the head
outputs zeros. ``UNetNosc`` has no such thing (its ``outc`` gets the usual Kaiming init). Nothing is
broken by it — the zeroed convs still receive gradient on the first step, and the layers above them
from the second onwards — it is the standard diffusion-model convention, and with
``residual_input_channel`` it means the model starts as an exact identity. It is left untouched
rather than silently diverging from upstream, but it does mean "the output is all zeros" and
"dropout changes nothing" are expected at step 0, not bugs.
"""
from __future__ import annotations

import math
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DiffusionModelUNet
from monai.networks.nets import diffusion_model_unet as _dmu

_PATCHED_RESBLOCK = False
_WARNED_BILINEAR = False


def patch_diffusion_resblock() -> None:
    """Replace ``DiffusionUNetResnetBlock.forward`` with a version that adds two things upstream
    lacks: a ``spatial_dims == 1`` branch for the timestep broadcast, and an optional ``dropout``
    submodule applied before the residual add.

    Upstream hardcodes the broadcast as ``[:, :, None, None]`` for ``spatial_dims == 2`` and
    ``[:, :, None, None, None]`` otherwise, so there is no 1-D case at all and the addition
    broadcasts wrongly (verified again against monai 1.5.2). Patching the one method is preferred to
    editing the installed package, and it is idempotent.
    """
    global _PATCHED_RESBLOCK
    if _PATCHED_RESBLOCK:
        return

    def patched_forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.nonlinearity(self.norm1(x))
        if self.upsample is not None:
            x = self.upsample(x)
            h = self.upsample(h)
        elif self.downsample is not None:
            x = self.downsample(x)
            h = self.downsample(h)
        h = self.conv1(h)
        proj = self.time_emb_proj(self.nonlinearity(emb))
        if self.spatial_dims == 1:
            temb = proj[:, :, None]
        elif self.spatial_dims == 2:
            temb = proj[:, :, None, None]
        else:
            temb = proj[:, :, None, None, None]
        h = self.conv2(self.nonlinearity(self.norm2(h + temb)))
        if hasattr(self, "dropout"):
            h = self.dropout(h)
        out: torch.Tensor = self.skip_connection(x) + h
        return out

    _dmu.DiffusionUNetResnetBlock.forward = patched_forward
    _PATCHED_RESBLOCK = True


def default_norm_num_groups(widths: tuple[int, ...]) -> int:
    """MONAI requires every width to be a multiple of ``norm_num_groups`` and defaults it to 32,
    which rejects any narrow model (``widths=(8, 16, 32)`` is used by the smoke tests). The largest
    group count that always works is the gcd of the widths, capped at MONAI's own 32."""
    return min(32, reduce(math.gcd, widths))


class MonaiUNet2d(nn.Module):
    """Call-compatible replacement for :class:`~oceanml3d.models.ocean.nn.unet_nosc.UNetNosc`.

    Same positional signature, same ``(B, C, H, W) -> (B, C', H, W)`` contract; the extra keywords
    (``num_res_blocks``, ``norm_num_groups``) are MONAI's own.

    ``attention_levels`` keeps this repository's convention — 1-based encoder levels, ``1`` being
    the first downsampling — and is translated to MONAI's per-level boolean sequence. The two
    indexings coincide because ``UNetNosc.downs[i - 1]`` produces ``widths[i]``.
    """

    def __init__(self, in_channels: int, out_channels: int, widths: tuple[int, ...] = (64, 128, 256, 512, 1024),
                 bilinear: bool = True, dropout: float = 0.0, residual_input_channel: int | None = None,
                 attention_levels: tuple[int, ...] = (), attention_heads: int = 4,
                 num_res_blocks: int = 2, norm_num_groups: int | None = None):
        super().__init__()
        patch_diffusion_resblock()
        widths = tuple(int(w) for w in widths)
        if len(widths) < 2:
            raise ValueError(f"widths needs at least two levels, got {widths}")
        n = len(widths)

        global _WARNED_BILINEAR
        if not _WARNED_BILINEAR:
            print("[oceanml3d] trunk 'monai' ignores 'bilinear': DiffusionModelUNet upsamples by "
                  "nearest interpolation + convolution. Use trunk='nosc' for the bilinear U-Net.")
            _WARNED_BILINEAR = True

        attn = [False] * n
        for level in attention_levels:
            level = int(level)
            if not 1 <= level <= n - 1:
                raise ValueError(f"attention level {level} out of range for {n} widths "
                                 f"(1 = first downsampling, {n - 1} = coarsest)")
            if widths[level] % attention_heads:
                raise ValueError(f"attention_heads={attention_heads} does not divide widths[{level}]="
                                 f"{widths[level]}; MONAI splits channels into heads exactly")
            attn[level] = True

        self.widths = widths
        self.downsample_factor = 2 ** (n - 1)
        self.residual_input_channel = residual_input_channel
        self.backbone = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=widths,
            attention_levels=tuple(attn),
            num_res_blocks=int(num_res_blocks),
            norm_num_groups=int(norm_num_groups) if norm_num_groups else default_norm_num_groups(widths),
            num_head_channels=[max(w // attention_heads, 1) for w in widths],
        )
        if dropout > 0:
            for module in self.backbone.modules():
                if isinstance(module, _dmu.DiffusionUNetResnetBlock):
                    module.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        f = self.downsample_factor
        ph, pw = (-h) % f, (-w) % f
        inp = F.pad(x, [0, pw, 0, ph], mode="replicate") if ph or pw else x
        timesteps = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        out = self.backbone(inp, timesteps)
        if ph or pw:
            out = out[..., :h, :w]
        if self.residual_input_channel is not None:
            out = out + x[:, self.residual_input_channel : self.residual_input_channel + 1]
        return out
