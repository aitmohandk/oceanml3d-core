"""THE product-format contract, shared verbatim by ``oceanml3d-core`` and ``oceanml3d-eval``.

This file is the single source of truth for the interface between the two repos: the writer
(:mod:`oceanml3d.inference.export`) and the readers (:mod:`oceanml3d_eval.product`) both validate
against it, so a format drift fails a test instead of producing silently unreadable products.

Version 2 adds the optional ``ensemble_size`` + ``coords.member`` pair, so probabilistic methods
(ensemble Kalman filters, flow-matching / diffusion samplers) are describable by the same manifest
as a deterministic reconstruction. Readers of v1 files keep working: no key became mandatory.

The file is duplicated *byte for byte* rather than turned into a third package, so that
``oceanml3d-eval`` stays installable without the training stack (torch, lightning, hydra).
Each repo pins its SHA-256 in ``docs/product_format.md`` and a test enforces it; when the
format changes, edit here, bump ``PRODUCT_FORMAT_VERSION``, copy to the other repo and update
both hashes.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

PRODUCT_FORMAT_VERSION = 2

REQUIRED_KEYS = ("name", "path", "pattern", "variables")
OPTIONAL_KEYS = ("format_version", "label", "coords", "depth_m", "depth_index", "time_coverage_hours",
                 "first_date", "last_date", "attrs", "scale", "lon_0_360", "ensemble_size")
CANONICAL_VARIABLES = ("u", "v", "ssh", "sst", "thetao", "so", "mld")
DEPTH_SUFFIX = re.compile(r"^(?P<base>[a-zA-Z][a-zA-Z0-9]*)_d(?P<index>\d{2})$")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_variable_name(name: str) -> tuple[str, int | None]:
    """``thetao_d12`` -> ``("thetao", 12)``; ``u`` -> ``("u", None)``."""
    m = DEPTH_SUFFIX.match(name)
    return (m.group("base"), int(m.group("index"))) if m else (name, None)


def validate_manifest(doc: Mapping[str, Any], strict: bool = False) -> list[str]:
    """Return a list of problems with a ``product.yaml`` document (empty = valid).

    ``strict=True`` also reports non-canonical variable base names, which are allowed but make
    a product incomparable with the built-in benchmarks.
    """
    problems: list[str] = []
    for key in REQUIRED_KEYS:
        if key not in doc:
            problems.append(f"missing required key '{key}'")
    if problems:
        return problems

    version = doc.get("format_version", PRODUCT_FORMAT_VERSION)
    if int(version) > PRODUCT_FORMAT_VERSION:
        problems.append(f"format_version {version} is newer than this reader ({PRODUCT_FORMAT_VERSION})")

    if not isinstance(doc["variables"], Mapping) or not doc["variables"]:
        problems.append("'variables' must be a non-empty mapping of canonical name -> description")
    else:
        for name, spec in doc["variables"].items():
            base, index = parse_variable_name(name)
            if strict and base not in CANONICAL_VARIABLES:
                problems.append(f"variable '{name}': base name '{base}' is not canonical {CANONICAL_VARIABLES}")
            if isinstance(spec, Mapping):
                if index is not None and spec.get("depth_index") not in (None, index):
                    problems.append(f"variable '{name}': depth_index {spec['depth_index']} contradicts the name suffix")
            elif spec is not None and not isinstance(spec, str):
                problems.append(f"variable '{name}': description must be a mapping or a string")

    try:
        re.compile(doc["pattern"])
    except re.error as exc:
        problems.append(f"'pattern' is not a valid regex: {exc}")

    coords = doc.get("coords", {"lat": "lat", "lon": "lon", "time": "time"})
    if not isinstance(coords, Mapping) or not {"lat", "lon", "time"} <= set(coords):
        problems.append("'coords' must map lat, lon and time to their names in the files")
    elif not isinstance(coords, Mapping):
        coords = {}

    for key in ("first_date", "last_date"):
        if key in doc and doc[key] is not None and not DATE.match(str(doc[key])):
            problems.append(f"'{key}' must be YYYY-MM-DD, got {doc[key]!r}")

    for key in ("depth_m", "time_coverage_hours"):
        if doc.get(key) is not None and not isinstance(doc[key], (int, float)):
            problems.append(f"'{key}' must be a number or null")

    n = doc.get("ensemble_size")
    if n is not None:
        if not isinstance(n, int) or n < 1:
            problems.append("'ensemble_size' must be a positive integer (a deterministic product omits it)")
        elif n > 1 and "member" not in coords:
            problems.append("an ensemble product must declare the member dimension in 'coords', e.g. coords.member: member")

    unknown = set(doc) - set(REQUIRED_KEYS) - set(OPTIONAL_KEYS)
    if unknown:
        problems.append(f"unknown keys {sorted(unknown)} (allowed: {sorted(set(REQUIRED_KEYS) | set(OPTIONAL_KEYS))})")
    return problems


def check_manifest(doc: Mapping[str, Any], source: str = "product.yaml", strict: bool = False) -> None:
    problems = validate_manifest(doc, strict)
    if problems:
        raise ValueError(f"invalid product manifest ({source}):\n  - " + "\n  - ".join(problems))
