"""Combine per-target losses into one scalar (port of NOSC contrib/multivar/loss_grouping.py).

flat_sum   : sum over targets (each depth channel weighs as much as the whole SSH).
group_mean : mean inside each group then weighted sum across groups, so each physical
             quantity contributes equally whatever its number of depth levels.
Pure Python: works on torch scalars and on floats (unit tests).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def combine_grouped_losses(per_var: Sequence[Any], names: Sequence[str], groups: Mapping[str, str] | None = None,
                           mode: str = "flat_sum", group_weights: Mapping[str, float] | None = None):
    if mode not in ("flat_sum", "group_mean"):
        raise ValueError(f"unknown loss combination mode '{mode}'")
    if mode == "flat_sum":
        total = None
        for loss in per_var:
            total = loss if total is None else total + loss
        return total, {"all": total}
    if groups is None:
        raise ValueError("mode='group_mean' requires groups")
    group_weights = group_weights or {}
    sums: dict[str, Any] = {}
    counts: dict[str, int] = {}
    for name, loss in zip(names, per_var, strict=True):
        g = groups.get(name, name)
        sums[g] = loss if g not in sums else sums[g] + loss
        counts[g] = counts.get(g, 0) + 1
    total, weighted = None, {}
    for g, s in sums.items():
        w = float(group_weights.get(g, 1.0)) / counts[g] * s
        weighted[g] = w
        total = w if total is None else total + w
    return total, weighted
