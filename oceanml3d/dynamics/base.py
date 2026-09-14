from __future__ import annotations

from collections.abc import Callable

import numpy as np

_DYNAMICS: dict[str, type] = {}


def register_dynamics(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        _DYNAMICS[name] = cls
        cls.registry_name = name
        return cls
    return deco


def get_dynamics(name: str, **kw) -> Dynamics:
    if name not in _DYNAMICS:
        raise KeyError(f"unknown dynamics '{name}'. Available: {list_dynamics()}")
    return _DYNAMICS[name](**kw)


def list_dynamics() -> list[str]:
    return sorted(_DYNAMICS)


class Dynamics:
    """Deterministic dynamical system integrated with RK4.

    ``forcing`` is an optional external term (the two-scale Lorenz-96 of
    ``4dvarnet-fm-opencode`` drives the slow variables with it); ``clip_range`` mirrors the
    stability clamp of the original implementation.
    """

    state_dim: int = 0
    dt: float = 0.01
    clip_range: float | None = None

    def rhs(self, state: np.ndarray, forcing: np.ndarray | float = 0.0) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def step(self, state: np.ndarray, dt: float | None = None, forcing: np.ndarray | float = 0.0) -> np.ndarray:
        h = self.dt if dt is None else dt
        k1 = self.rhs(state, forcing)
        k2 = self.rhs(state + h / 2 * k1, forcing)
        k3 = self.rhs(state + h / 2 * k2, forcing)
        k4 = self.rhs(state + h * k3, forcing)
        out = state + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return np.clip(out, -self.clip_range, self.clip_range) if self.clip_range else out

    def simulate(self, n_steps: int, x0: np.ndarray | None = None, spinup: int = 0,
                 dt: float | None = None, seed: int = 0, forcing: np.ndarray | float = 0.0) -> np.ndarray:
        """Trajectory of shape ``(n_steps, state_dim)``. ``forcing`` may be a scalar or a
        ``(spinup + n_steps,)`` series."""
        rng = np.random.default_rng(seed)
        x = self.initial_state(rng) if x0 is None else np.asarray(x0, float).copy()
        series = np.asarray(forcing, float)
        get = (lambda i: float(series[min(i, series.size - 1)])) if series.ndim else (lambda i: float(series))
        for i in range(spinup):
            x = self.step(x, dt, get(i))
        out = np.empty((n_steps, self.state_dim))
        for i in range(n_steps):
            out[i] = x
            x = self.step(x, dt, get(spinup + i))
        return out

    def initial_state(self, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal(self.state_dim)
