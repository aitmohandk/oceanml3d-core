"""Output heads shared by models: single conv, grouped conv heads, vertical-modes projection."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
import torch.nn as nn


class GroupedHeads(nn.Module):
    """One small conv stack per target group on a shared feature map, concatenated in group order."""

    def __init__(self, in_channels: int, group_channels: Mapping[str, int], hidden: int = 32, n_layers: int = 2):
        super().__init__()
        self.group_channels = dict(group_channels)
        self.heads = nn.ModuleDict()
        for name, n_out in self.group_channels.items():
            layers, c = [], in_channels
            for _ in range(max(n_layers - 1, 0)):
                layers += [nn.Conv2d(c, hidden, 3, padding=1), nn.SiLU()]
                c = hidden
            last = nn.Conv2d(c, n_out, 3, padding=1)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            self.heads[name] = nn.Sequential(*layers, last)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.heads[n](x) for n in self.group_channels], dim=1)


class VerticalModesHead(nn.Module):
    """The trunk predicts K EOF coefficients per (group, time step); a frozen basis maps them to levels.

    ``mode_specs``: group -> {"npz": path (from oceanml3d.training.vertical_modes), "n_modes": K}.
    Groups without a spec are passed through untouched.
    """

    def __init__(self, group_levels: Mapping[str, int], window: int, mode_specs: Mapping[str, Mapping]):
        super().__init__()
        self.window = window
        self.layout: list[tuple[str, int, str | None, int]] = []
        self.trunk_channels = 0
        for name, n_levels in group_levels.items():
            if name in mode_specs:
                spec = mode_specs[name]
                comp = np.load(spec["npz"])["components"][: int(spec["n_modes"])]
                if comp.shape[1] != n_levels:
                    raise ValueError(f"group '{name}': EOF basis has {comp.shape[1]} levels, config declares {n_levels}")
                self.register_buffer(f"proj_{name}", torch.from_numpy(comp.T.astype(np.float32)))
                k = comp.shape[0]
                self.layout.append((name, k * window, f"proj_{name}", n_levels))
                self.trunk_channels += k * window
            else:
                self.layout.append((name, n_levels * window, None, n_levels))
                self.trunk_channels += n_levels * window

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        outs, start = [], 0
        for _name, n, proj_name, n_levels in self.layout:
            block = h[:, start:start + n]
            start += n
            if proj_name is None:
                outs.append(block)
            else:
                proj = getattr(self, proj_name)
                b, _, hh, ww = block.shape
                coeffs = block.view(b, -1, self.window, hh, ww)
                outs.append(torch.einsum("lk,bkthw->blthw", proj, coeffs).reshape(b, n_levels * self.window, hh, ww))
        return torch.cat(outs, dim=1)
