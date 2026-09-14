"""Typed description of the stable config groups + eager validation.

Hydra composes untyped YAML: a typo in a group name, a stride larger than a patch or a test
window outside the split only shows up hours into a job. ``validate_config`` runs at the very
start of every command (and in ``command=show-config``) and reports *all* problems at once.

The dataclasses are the documentation of what each group must contain; they are also
registered in Hydra's ConfigStore so ``--cfg job`` shows resolved types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

DIMS = ("time", "lat", "lon")


@dataclass
class TrainerConfig:
    accelerator: str = "auto"
    devices: Any = 1
    max_epochs: int = 100
    precision: Any = 32


@dataclass
class TrainingConfig:
    batch_size: int = 1
    num_workers: int = 4
    loss: str = "mae"                     # mae | mse
    loss_combine: str = "flat_sum"        # flat_sum | group_mean | uncertainty
    grad_loss_weight: float = 0.0
    save_top_k: int = 3
    moving_patches: bool = False
    cache: bool = False
    loss_group_weights: dict[str, float] = field(default_factory=dict)
    optimizer: dict[str, Any] = field(default_factory=dict)
    rec_weight: dict[str, Any] = field(default_factory=dict)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)


@dataclass
class ExportConfig:
    enabled: bool = True
    split: str = "test"
    time: Any = None
    depth_m: float | None = None


VALID = {"loss": {"mae", "mse"}, "loss_combine": {"flat_sum", "group_mean", "uncertainty"},
         "rec_weight_kind": {"constant", "triangular"}, "split": {"train", "val", "test"}}


def _slice_bounds(v: Any) -> tuple[Any, Any]:
    if isinstance(v, dict):
        v = v.get("time", v)
    return (v[0], v[1]) if isinstance(v, (list, tuple)) else (None, None)


def validate_config(cfg: DictConfig, catalog=None, variables=None) -> list[str]:
    """Return every problem found in a composed config (empty list = valid)."""
    p: list[str] = []
    d, t = cfg.data, cfg.training

    for dim in DIMS:
        if dim not in d.patch or dim not in d.stride:
            p.append(f"data.patch/stride must define '{dim}'")
            continue
        if int(d.stride[dim]) > int(d.patch[dim]):
            p.append(f"data.stride.{dim} ({d.stride[dim]}) > data.patch.{dim} ({d.patch[dim]}): gaps in coverage")
    if "time" in d.patch and int(d.patch.time) < 1:
        p.append("data.patch.time must be >= 1")

    crop = t.rec_weight.get("crop", {})
    for dim in DIMS:
        if 2 * int(crop.get(dim, 0)) >= int(d.patch.get(dim, 1)):
            p.append(f"training.rec_weight.crop.{dim} removes the whole patch")
    if t.rec_weight.get("kind") not in VALID["rec_weight_kind"]:
        p.append(f"training.rec_weight.kind must be one of {sorted(VALID['rec_weight_kind'])}")

    for key in ("loss", "loss_combine"):
        if t.get(key) not in VALID[key]:
            p.append(f"training.{key}='{t.get(key)}' must be one of {sorted(VALID[key])}")

    splits = {k: _slice_bounds(v) for k, v in OmegaConf.to_container(d.splits, resolve=True).items()}
    for name in ("train", "val", "test"):
        if name not in splits:
            p.append(f"data.splits.{name} is required")
    for name, (start, stop) in splits.items():
        if start is None or stop is None:
            p.append(f"data.splits.{name} must be a [start, stop] pair")
        elif pd.Timestamp(start) >= pd.Timestamp(stop):
            p.append(f"data.splits.{name}: start {start} is not before stop {stop}")
    if "train" in splits and "val" in splits and all(splits["train"]) and all(splits["val"]):
        if pd.Timestamp(splits["val"][0]) < pd.Timestamp(splits["train"][1]):
            p.append(f"data.splits.val starts inside the train window ({splits['val'][0]} < {splits['train'][1]}): leakage")

    e = cfg.get("export", {})
    if e.get("enabled", False):
        if e.get("split") not in VALID["split"]:
            p.append(f"export.split='{e.get('split')}' must be one of {sorted(VALID['split'])}")
        if e.get("time") and e.split in splits and all(splits[e.split]):
            s0, s1 = pd.Timestamp(e.time[0]), pd.Timestamp(e.time[1])
            w0, w1 = pd.Timestamp(splits[e.split][0]), pd.Timestamp(splits[e.split][1])
            if s0 < w0 or s1 > w1:
                p.append(f"export.time {list(e.time)} is outside the '{e.split}' split [{splits[e.split][0]}, {splits[e.split][1]}]")

    if variables is not None:
        if d.get("norm_stats"):
            n = len(OmegaConf.to_container(d.norm_stats, resolve=True)[0])
            if n != len(variables):
                p.append(f"data.norm_stats has {n} entries for {len(variables)} variables")
        groups = variables.target_groups
        if t.get("loss_combine") == "group_mean" and len(groups) == 1:
            p.append("training.loss_combine='group_mean' with a single target group has no effect")
        for spec in variables:
            if spec.depth_index is not None and spec.depth_index < 0:
                p.append(f"variable '{spec.name}': depth_index must be >= 0")
    if catalog is not None and variables is not None:
        for spec in variables:
            try:
                path = catalog.resolve(spec.source)
            except KeyError:
                p.append(f"variable '{spec.name}': source '{spec.source}' is not in the catalog (paths=<site>)")
                continue
            if not path.exists():
                p.append(f"variable '{spec.name}': {path} does not exist "
                         f"(run the prepare recipes, or `oceanml3d command=prepare-obs`)")
    return p


def check_config(cfg: DictConfig, catalog=None, variables=None) -> None:
    problems = validate_config(cfg, catalog, variables)
    if problems:
        raise SystemExit("invalid configuration:\n  - " + "\n  - ".join(problems))


def register_schemas() -> None:
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(group="training", name="_schema", node=TrainingConfig)
    cs.store(name="_export_schema", node=ExportConfig)
