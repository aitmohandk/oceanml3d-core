"""Data catalog: logical dataset keys -> local paths, per site.

NOSC hard-codes ``/Odyssey/private/...`` paths inside every experiment YAML. Here an
experiment only references logical keys (``ssh_l4_cmems_4th``) and the mapping to a
path lives in a per-site file (``config/paths/<site>.yaml``) or in ``$OCEANML3D_CATALOG``.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


class Catalog:
    def __init__(self, entries: Mapping[str, Any], root: str | os.PathLike | None = None):
        self.root = Path(root) if root else None
        self.entries: dict[str, dict[str, Any]] = {}
        for key, value in entries.items():
            if isinstance(value, str):
                value = {"path": value}
            self.entries[key] = dict(value)

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> Catalog:
        path = path or os.environ.get("OCEANML3D_CATALOG")
        if path is None:
            return cls({})
        with open(path) as f:
            doc = yaml.safe_load(f) or {}
        return cls(doc.get("datasets", {}), root=doc.get("root"))

    def resolve(self, key_or_path: str) -> Path:
        """Return a path for a catalog key; a real path is passed through unchanged."""
        if key_or_path in self.entries:
            p = Path(os.path.expandvars(self.entries[key_or_path]["path"]))
            if not p.is_absolute() and self.root is not None:
                p = self.root / p
            return p
        p = Path(os.path.expandvars(key_or_path))
        if p.exists() or any(ch in key_or_path for ch in "/\\."):
            return p
        raise KeyError(
            f"'{key_or_path}' is neither a catalog key nor an existing path. "
            f"Known keys: {sorted(self.entries)}"
        )

    def meta(self, key: str) -> dict[str, Any]:
        return self.entries.get(key, {})

    def __contains__(self, key: str) -> bool:
        return key in self.entries
