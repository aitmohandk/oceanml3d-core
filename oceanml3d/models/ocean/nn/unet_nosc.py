"""NOSC's own residual 2D U-Net (cleaned from NOSC ``contrib/multivar/parts_drop.py``, itself
adapted from milesial/Pytorch-UNet).

This is no longer the default trunk of :class:`~oceanml3d.models.ocean.nosc.model.NOSCUNet` — MONAI's
``DiffusionModelUNet`` is (:mod:`oceanml3d.models.ocean.nn.unet_monai`). It is kept, unchanged, as
``trunk: nosc`` / ``ablation=trunk_nosc``, which is how every run published before that switch is
reproduced.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, mid: int | None = None, kernel_size: int = 3,
                 scale: float = 1.0, dropout: float = 0.0):
        super().__init__()
        mid = mid or cout
        pad = kernel_size // 2
        self.scale = scale
        self.body = nn.Sequential(
            nn.Conv2d(cin, mid, kernel_size, padding=pad, bias=False), nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(mid, cout, kernel_size, padding=pad, bias=False), nn.BatchNorm2d(cout),
            nn.Dropout2d(dropout),
        )
        self.proj = None if cin == cout else nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        if self.proj is not None:
            x = self.proj(x)
        return F.relu(h * self.scale + x)


class Down(nn.Module):
    def __init__(self, cin: int, cout: int, **kw):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), ResBlock(cin, cout, **kw))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, cin: int, cout: int, bilinear: bool = True, **kw):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = ResBlock(cin, cout, cin // 2, **kw)
        else:
            self.up = nn.ConvTranspose2d(cin, cin // 2, 2, stride=2)
            self.conv = ResBlock(cin, cout, **kw)

    def forward(self, x, skip):
        x = self.up(x)
        dy, dx = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class SelfAttention2d(nn.Module):
    """Multi-head self-attention over spatial positions (used at coarse U-Net levels)."""

    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        t = self.norm(x).flatten(2).transpose(1, 2)
        a, _ = self.attn(t, t, t, need_weights=False)
        return x + self.proj(a.transpose(1, 2).reshape(b, c, h, w))


class UNetNosc(nn.Module):
    """Encoder/decoder with ``len(widths)-1`` down/up levels. Input/output ``(B, C, H, W)``."""

    def __init__(self, in_channels: int, out_channels: int, widths: tuple[int, ...] = (64, 128, 256, 512, 1024),
                 bilinear: bool = True, dropout: float = 0.0, residual_input_channel: int | None = None,
                 attention_levels: tuple[int, ...] = (), attention_heads: int = 4):
        """``attention_levels``: encoder levels (1 = first downsampling, ...) that get self-attention."""
        super().__init__()
        factor = 2 if bilinear else 1
        n = len(widths)
        scales = 1.0 / torch.arange(1, 2 * n).sqrt()
        self.inc = ResBlock(in_channels, widths[0], dropout=dropout)
        self.downs = nn.ModuleList()
        for i in range(1, n):
            cout = widths[i] // factor if i == n - 1 else widths[i]
            self.downs.append(Down(widths[i - 1], cout, scale=float(scales[i]), dropout=dropout))
        self.attn = nn.ModuleDict({str(i): SelfAttention2d(self.downs[i - 1].net[1].body[-3].out_channels, attention_heads)
                                   for i in attention_levels})
        self.ups = nn.ModuleList()
        for i in range(n - 1, 0, -1):
            cout = widths[i - 1] // factor if i > 1 else widths[0]
            self.ups.append(Up(widths[i], cout, bilinear, scale=float(scales[n - 1 + (n - i)]), dropout=dropout))
        self.outc = nn.Conv2d(widths[0], out_channels, 1, bias=False)
        self.residual_input_channel = residual_input_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.inc(x)]
        for i, d in enumerate(self.downs, start=1):
            h = d(skips[-1])
            if str(i) in self.attn:
                h = self.attn[str(i)](h)
            skips.append(h)
        h = skips.pop()
        for u in self.ups:
            h = u(h, skips.pop())
        out = self.outc(h)
        if self.residual_input_channel is not None:
            out = out + x[:, self.residual_input_channel : self.residual_input_channel + 1]
        return out
