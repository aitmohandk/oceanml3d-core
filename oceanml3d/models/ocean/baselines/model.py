"""Trivial, non-neural reference models.

They exist to keep the framework honest: if an interface only fits ``nosc_unet``, these break.
They are also the cheapest possible baselines in a benchmark (a learned model that does not beat
``persistence`` or ``passthrough`` is not learning anything useful) and they run on CPU in seconds.

* ``passthrough`` — copy a designated input channel onto each target (e.g. DUACS geostrophy as a
  prediction of the drifter currents). Zero parameters; ``train`` just measures it.
* ``linear`` — one learned affine combination of all input channels per target, shared over space
  and time (a per-pixel linear regression). The natural "is the U-Net worth it?" control.
* ``climatology`` — learned constant per target and per day-of-window position; ignores the inputs.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from oceanml3d.models.ocean.base import BaseOceanModel
from oceanml3d.registry import register_model
from oceanml3d.variables import VariableSet


@register_model("passthrough")
class Passthrough(BaseOceanModel):
    def __init__(self, variables: VariableSet, window: int, rec_weight: np.ndarray,
                 source: dict[str, str] | None = None, **base_kw):
        """``source``: {target name: input name}; defaults to the i-th input for the i-th target."""
        super().__init__(variables, window, rec_weight, **base_kw)
        self.save_hyperparameters(ignore=["variables", "rec_weight", "norm_stats"])
        in_names = [s.name for s in variables.inputs]
        tgt = variables.targets
        source = source or {t.name: in_names[min(i, len(in_names) - 1)] for i, t in enumerate(tgt)}
        missing = [v for v in source.values() if v not in in_names]
        if missing:
            raise ValueError(f"passthrough source(s) {missing} are not input variables")
        self.register_buffer("pick", torch.as_tensor([in_names.index(source[t.name]) for t in tgt]))
        self.scale = nn.Parameter(torch.ones(len(tgt)))       # 1 parameter per target: keeps Lightning happy
        self.bias = nn.Parameter(torch.zeros(len(tgt)))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        x = self.inputs(batch).index_select(1, self.pick)
        return x * self.scale[None, :, None, None, None] + self.bias[None, :, None, None, None]


@register_model("linear")
class LinearBaseline(BaseOceanModel):
    def __init__(self, variables: VariableSet, window: int, rec_weight: np.ndarray, use_time_window: bool = False, **base_kw):
        """``use_time_window=True`` lets each target see the whole window (n_in * T -> n_out * T weights)."""
        super().__init__(variables, window, rec_weight, **base_kw)
        self.save_hyperparameters(ignore=["variables", "rec_weight", "norm_stats"])
        self.n_in, self.n_out, self.use_time_window = len(variables.inputs), len(variables.targets), use_time_window
        c_in = self.n_in * window if use_time_window else self.n_in
        c_out = self.n_out * window if use_time_window else self.n_out
        self.lin = nn.Conv2d(c_in, c_out, 1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        x = self.inputs(batch)
        b, _, t, h, w = x.shape
        if self.use_time_window:
            return self.lin(x.reshape(b, self.n_in * t, h, w)).reshape(b, self.n_out, t, h, w)
        y = self.lin(x.permute(0, 2, 1, 3, 4).reshape(b * t, self.n_in, h, w))
        return y.reshape(b, t, self.n_out, h, w).permute(0, 2, 1, 3, 4)


@register_model("climatology")
class Climatology(BaseOceanModel):
    def __init__(self, variables: VariableSet, window: int, rec_weight: np.ndarray, **base_kw):
        super().__init__(variables, window, rec_weight, **base_kw)
        self.save_hyperparameters(ignore=["variables", "rec_weight", "norm_stats"])
        self.value = nn.Parameter(torch.zeros(len(variables.targets), window))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        b, _, t, h, w = batch.shape
        return self.value[None, :, :t, None, None].expand(b, -1, -1, h, w).contiguous()
