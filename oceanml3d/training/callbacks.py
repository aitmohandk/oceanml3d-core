"""Callbacks: reproducibility stamp (git hash + config) written next to checkpoints."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


class VersioningCallback(Callback):
    def __init__(self, cfg=None):
        self.cfg = cfg

    def on_fit_start(self, trainer, pl_module) -> None:
        if trainer.logger is None or trainer.logger.log_dir is None:
            return
        out = Path(trainer.logger.log_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "git_hash.txt").write_text(git_hash())
        if self.cfg is not None:
            (out / "config.yaml").write_text(OmegaConf.to_yaml(self.cfg))
        if hasattr(pl_module, "norm_stats") and pl_module.norm_stats is not None:
            mean, std = pl_module.norm_stats
            (out / "norm_stats.json").write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist()}))
