"""Model registry.

A model is registered either with the ``@register_model("name")`` decorator or through the
``oceanml3d.models`` entry point group in a third-party package's ``pyproject.toml``.
Experiments then refer to it by name: ``model: {name: nosc_unet, ...}``.
"""
from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points
from typing import TypeVar

_MODELS: dict[str, type] = {}
_IMPORT_ERROR: Exception | None = None
T = TypeVar("T", bound=type)


def set_import_error(exc: Exception) -> None:
    """Recorded by :mod:`oceanml3d.models.ocean` when the built-in models cannot be imported, so that
    ``get_model`` explains *why* the registry is empty instead of just saying 'unknown model'."""
    global _IMPORT_ERROR
    _IMPORT_ERROR = exc


def register_model(name: str) -> Callable[[T], T]:
    def deco(cls: T) -> T:
        if name in _MODELS and _MODELS[name] is not cls:
            raise ValueError(f"model '{name}' already registered by {_MODELS[name]}")
        _MODELS[name] = cls
        cls.registry_name = name  # type: ignore[attr-defined]
        return cls
    return deco


def _load_entry_points() -> None:
    try:
        eps = entry_points(group="oceanml3d.models")
    except TypeError:  # python < 3.10 API
        eps = entry_points().get("oceanml3d.models", [])
    for ep in eps:
        if ep.name not in _MODELS:
            try:
                _MODELS[ep.name] = ep.load()
            except Exception as exc:  # noqa: BLE001 - a broken plugin must not break the core
                print(f"[oceanml3d] could not load model plugin '{ep.name}': {exc}")


def _ensure_builtins() -> None:
    import importlib

    if not _MODELS:
        importlib.import_module("oceanml3d.models.ocean")
    _load_entry_points()


def get_model(name: str) -> type:
    _ensure_builtins()
    if name not in _MODELS:
        hint = (f"; the built-in models could not be imported ({_IMPORT_ERROR}) - "
                'install the training extra: pip install -e ".[torch]"') if _IMPORT_ERROR else ""
        raise KeyError(f"unknown model '{name}'. Available: {sorted(_MODELS)}{hint}")
    return _MODELS[name]


def list_models() -> list[str]:
    _ensure_builtins()
    return sorted(_MODELS)
