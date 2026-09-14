"""Training-time augmentations applied on numpy patches ``(n_channels, T, H, W)`` *before*
normalisation. Declared in ``data.augment`` and only active on the train split.

* :class:`MissionDropout` — remove 1..k altimetry missions from the observed SSH (and its
  mask channel) of a sample, using the per-mission masks written by
  ``build_mask_dataset(per_mission=True)`` and loaded as ``role: aux`` variables. This is the
  "what does losing Jason-3 cost?" OSSE ablation and an operational-robustness regulariser.
* :class:`ObsNoise` — re-draw Gaussian noise on observed cells of the listed channels each time
  a sample is drawn (validation/test keep the baked deterministic noise).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from oceanml3d.variables import VariableSet


class Augmentation:
    def bind(self, variables: VariableSet) -> None:  # noqa: D401
        pass

    def __call__(self, item: np.ndarray, rng: np.random.Generator) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


@dataclass
class MissionDropout(Augmentation):
    obs: list[str]                       # observation channels to thin (e.g. [ssh_obs])
    mask: str | None = "obs_mask"        # explicit presence channel to update (or None)
    missions: list[str] = field(default_factory=list)   # aux variable names (mask_<mission>)
    n_drop: tuple[int, int] = (1, 2)
    p: float = 0.5
    _idx: dict = field(default_factory=dict, init=False)

    def bind(self, variables: VariableSet) -> None:
        self._idx = {"obs": [variables.index(n) for n in self.obs],
                     "mask": variables.index(self.mask) if self.mask else None,
                     "missions": [variables.index(n) for n in self.missions]}

    def __call__(self, item: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() > self.p or not self._idx["missions"]:
            return item
        k = int(rng.integers(self.n_drop[0], min(self.n_drop[1], len(self._idx["missions"])) + 1))
        drop = rng.choice(self._idx["missions"], size=k, replace=False)
        keep_idx = [i for i in self._idx["missions"] if i not in drop]
        kept = np.any(item[keep_idx] > 0, axis=0) if keep_idx else np.zeros(item.shape[1:], bool)
        lost = np.any(item[drop] > 0, axis=0) & ~kept
        for i in self._idx["obs"]:
            item[i][lost] = np.nan
        if self._idx["mask"] is not None:
            item[self._idx["mask"]][lost] = 0.0
        return item


@dataclass
class ObsNoise(Augmentation):
    channels: dict[str, float]           # channel name -> noise std (physical units)
    _idx: dict = field(default_factory=dict, init=False)

    def bind(self, variables: VariableSet) -> None:
        self._idx = {variables.index(n): s for n, s in self.channels.items()}

    def __call__(self, item: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        for i, std in self._idx.items():
            obs = np.isfinite(item[i])
            item[i][obs] += rng.normal(0.0, std, obs.sum()).astype(item.dtype)
        return item


AUGMENTATIONS = {"mission_dropout": MissionDropout, "obs_noise": ObsNoise}


def build_augmentations(cfg: dict | None, variables: VariableSet) -> list[Augmentation]:
    out = []
    for name, kw in (cfg or {}).items():
        if kw is None:
            continue
        kw = dict(kw)
        kind = kw.pop("kind", name)
        if "n_drop" in kw:
            kw["n_drop"] = tuple(kw["n_drop"])
        aug = AUGMENTATIONS[kind](**kw)
        aug.bind(variables)
        out.append(aug)
    return out
