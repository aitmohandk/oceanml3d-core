"""Variable specifications: the single description of what a model reads and predicts.

This generalises the ``multivar`` dictionary of NOSC (``var_path``, ``var_name``,
``input_arch``, ``output_arch``, ``broadcast_time``) into a typed, validated object
shared by the data module, the model and the export step.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    INPUT = "input"          # fed to the network only
    TARGET = "target"        # supervised only (may contain NaN)
    BOTH = "both"            # fed as (masked) input and supervised
    STATIC = "static"        # time-invariant field broadcast over the window (lat, bathy...)
    AUX = "aux"              # loaded for augmentations / diagnostics only (never fed nor supervised)


@dataclass
class VariableSpec:
    name: str
    source: str                       # catalog key or path (resolved by oceanml3d.catalog)
    var_name: str | None = None       # name inside the file; defaults to `name`
    role: Role = Role.INPUT
    mask: str | None = None           # optional catalog key of an observation mask
    transform: str | None = None      # registered transform name (e.g. "log_grad")
    fill_nan: float | None = None
    standard_name: str | None = None  # CF-like name used at export (e.g. "eastward_sea_water_velocity")
    units: str | None = None
    depth_index: int | None = None    # exact position in the file's depth axis (isel); None = surface/2D
    depth_m: float | None = None      # informative, filled by `oceanml3d depths` or by hand
    group: str | None = None          # loss / head group (e.g. "temperature"); default = own name
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.role = Role(self.role)
        if self.var_name is None:
            self.var_name = self.name

    @property
    def is_input(self) -> bool:
        return self.role in (Role.INPUT, Role.BOTH, Role.STATIC)

    @property
    def is_target(self) -> bool:
        return self.role in (Role.TARGET, Role.BOTH)

    @property
    def is_static(self) -> bool:
        return self.role is Role.STATIC

    @property
    def group_name(self) -> str:
        return self.group or self.name

    @property
    def base_name(self) -> str:
        """`thetao_d05` -> `thetao` (name without the depth suffix)."""
        return self.name.rsplit("_d", 1)[0] if self.depth_index is not None and "_d" in self.name else self.name


class VariableSet:
    """Ordered collection of VariableSpec with channel-layout helpers."""

    def __init__(self, specs: Iterable[VariableSpec]):
        self.specs: list[VariableSpec] = list(specs)
        names = [s.name for s in self.specs]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate variable names: {names}")
        if not self.targets:
            raise ValueError("a VariableSet needs at least one target variable")

    @classmethod
    def from_config(cls, cfg: Mapping[str, Mapping[str, Any]]) -> VariableSet:
        """Build from a mapping ``name -> spec``.

        A spec with ``depth_indices: [i, j, ...]`` expands into one variable per level named
        ``<name>_d<ii>`` (``depth_index=i``, same group), replacing NOSC's generated
        ``config/vars/*_gs21.yaml`` fragments and ``depth_fragments.py``.
        """
        specs = []
        for name, v in cfg.items():
            if v is None:          # `name: null` in an override removes the variable
                continue
            v = dict(v)
            if v.get("depth_indices") is not None:
                indices = list(v.pop("depth_indices"))
                depth_values = v.pop("depth_values", None) or {}
                if len(set(indices)) != len(indices):
                    raise ValueError(f"{name}: duplicate depth indices {indices}")
                v.setdefault("group", name)
                v.setdefault("var_name", name)
                for i in indices:
                    specs.append(VariableSpec(name=f"{name}_d{int(i):02d}", depth_index=int(i),
                                              depth_m=depth_values.get(int(i)), **v))
            else:
                v.pop("depth_values", None)
                specs.append(VariableSpec(name=name, **v))
        return cls(specs)

    def __iter__(self):
        return iter(self.specs)

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, name: str) -> VariableSpec:
        for s in self.specs:
            if s.name == name:
                return s
        raise KeyError(name)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.specs]

    @property
    def inputs(self) -> list[VariableSpec]:
        return [s for s in self.specs if s.is_input]

    @property
    def targets(self) -> list[VariableSpec]:
        return [s for s in self.specs if s.is_target]

    def index(self, name: str) -> int:
        return self.names.index(name)

    @property
    def input_indices(self) -> list[int]:
        return [i for i, s in enumerate(self.specs) if s.is_input]

    @property
    def target_indices(self) -> list[int]:
        return [i for i, s in enumerate(self.specs) if s.is_target]

    def n_input_channels(self, window: int) -> int:
        return len(self.inputs) * window

    def n_target_channels(self, window: int) -> int:
        return len(self.targets) * window

    @property
    def target_groups(self) -> dict[str, list[str]]:
        """Ordered ``group -> [target names]`` (groups are contiguous in target order by construction
        when built from config; heads rely on this)."""
        out: dict[str, list[str]] = {}
        for s in self.targets:
            out.setdefault(s.group_name, []).append(s.name)
        return out
