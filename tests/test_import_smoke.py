"""Every module of the project must import.

`PLAN.md` §1: the safety net required before any rename or move. A rename that misses one file
usually fails only when that code path first runs — sometimes hours into a training job, sometimes
only in the one evaluation script nobody runs on CI. This turns that into a one-second failure, and
it covers the modules no other test imports.

Discovery is dynamic on purpose: a new module is covered the moment it is added, without anyone
remembering to list it here.
"""
from __future__ import annotations

import importlib
import pkgutil

import pytest

ROOT = "oceanml3d"
SUBPACKAGES = ("models", "models.ocean", "data", "training", "inference", "obs",
               "legacy", "legacy.models", "legacy.data", "legacy.training", "legacy.evaluation",
               "legacy.conf")

# Modules whose import genuinely requires an optional dependency. Anything else that fails to
# import is a bug, not a missing extra — keep this list short and justified.
OPTIONAL = {
    "oceanml3d.legacy.models.monai_unet_adapter": "monai",
}


def _discover() -> list[str]:
    pkg = importlib.import_module(ROOT)
    names = [ROOT]
    names += [m.name for m in pkgutil.walk_packages(pkg.__path__, prefix=f"{ROOT}.")]
    return sorted(set(names))


@pytest.mark.parametrize("module", _discover())
def test_module_imports(module: str) -> None:
    if module in OPTIONAL:
        pytest.importorskip(OPTIONAL[module], reason=f"{module} needs {OPTIONAL[module]}")
    importlib.import_module(module)


def test_discovery_actually_found_the_modules() -> None:
    """Guards the guard: a wrong `ROOT` would make every test above vacuously pass."""
    found = _discover()
    assert len(found) > 40, found
    for sub in SUBPACKAGES:
        assert f"{ROOT}.{sub}" in found, sub
