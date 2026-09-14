"""Lorenz-63 and Lorenz-96 (ported from 4dvarnet-fm-opencode ``models/lorenz*_dynamics.py``)."""
from __future__ import annotations

import numpy as np

from oceanml3d.dynamics.base import Dynamics, register_dynamics


@register_dynamics("lorenz63")
class Lorenz63(Dynamics):
    state_dim = 3

    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0, dt: float = 0.01):
        self.sigma, self.rho, self.beta, self.dt = sigma, rho, beta, dt

    def rhs(self, s: np.ndarray, forcing: np.ndarray | float = 0.0) -> np.ndarray:
        x, y, z = s[..., 0], s[..., 1], s[..., 2]
        return np.stack([self.sigma * (y - x) + forcing, x * (self.rho - z) - y, x * y - self.beta * z], axis=-1)

    def initial_state(self, rng):
        return np.array([1.0, 1.0, 1.0]) + 0.01 * rng.standard_normal(3)


@register_dynamics("lorenz96")
class Lorenz96(Dynamics):
    def __init__(self, state_dim: int = 40, forcing: float = 8.0, dt: float = 0.01):
        self.state_dim, self.forcing, self.dt = state_dim, forcing, dt

    def rhs(self, s: np.ndarray, forcing: np.ndarray | float = 0.0) -> np.ndarray:
        return (np.roll(s, -1, axis=-1) - np.roll(s, 2, axis=-1)) * np.roll(s, 1, axis=-1) - s + self.forcing + forcing

    def initial_state(self, rng):
        x = np.full(self.state_dim, self.forcing)
        x[0] += 0.01
        return x


@register_dynamics("lorenz96_2scale")
class Lorenz96TwoScale(Dynamics):
    """Two-scale Lorenz-96 (Lorenz 1996): NO slow variables X coupled to NO x J fast variables Y.

    Port of ``4dvarnet-fm-opencode/models/lorenz96_dynamics.py::Lorenz96Dynamics`` (numpy instead of
    torch, same equations, same RK4 and the same ``clip_range`` clamp). This is the system the
    fm case studies actually use — ``lorenz96`` above is the classic single-scale version.

        dX_i/dt = X_{i-1} (X_{i+1} - X_{i-2}) - X_i + F - h * sum_j Y_{i,j} + c1 * W
        dY_{i,j}/dt = [ Y_{i,j+1} (Y_{i,j-1} - Y_{i,j-2}) - Y_{i,j} + hx * X_i ] / eps

    ``fast_weights`` optionally weights the sum over the fast variables; ``coupling_exponent``
    applies ``c1 * sign(W) |W|^p`` to the external forcing.
    """

    def __init__(self, NO: int = 8, J: int = 4, forcing: float = 8.0, h: float = 1.0, hx: float = 1.0,
                 eps: float = 0.1, c1: float = 1.0, coupling_exponent: float = 1.0, dt: float = 0.001,
                 clip_range: float | None = 50.0, fast_weights: list[float] | None = None):
        self.NO, self.J, self.forcing = NO, J, forcing
        self.h, self.hx, self.eps, self.c1 = h, hx, eps, c1
        self.coupling_exponent, self.dt, self.clip_range = coupling_exponent, dt, clip_range
        self.fast_weights = None if fast_weights is None else np.asarray(fast_weights, float)
        self.state_dim = NO + NO * J

    def _coupling(self, w: np.ndarray | float) -> np.ndarray | float:
        if self.coupling_exponent == 1.0:
            return self.c1 * w
        return self.c1 * np.sign(w) * np.abs(w) ** self.coupling_exponent

    def rhs(self, s: np.ndarray, forcing: np.ndarray | float = 0.0) -> np.ndarray:
        NO, J = self.NO, self.J
        x = s[..., :NO]
        y = s[..., NO:].reshape(*s.shape[:-1], NO, J)
        y_sum = (y * self.fast_weights).sum(-1) if self.fast_weights is not None else y.sum(-1)
        dx = (np.roll(x, 1, -1) * (np.roll(x, -1, -1) - np.roll(x, 2, -1)) - x + self.forcing
              - self.h * y_sum + self._coupling(forcing))
        dy = (np.roll(y, -1, -1) * (np.roll(y, 1, -1) - np.roll(y, 2, -1)) - y + self.hx * x[..., None]) / self.eps
        return np.concatenate([dx, dy.reshape(*s.shape[:-1], NO * J)], axis=-1)

    def initial_state(self, rng):
        x = np.full(self.NO, self.forcing)
        x[0] += 0.01
        return np.concatenate([x, 0.01 * rng.standard_normal(self.NO * self.J)])

    def slow(self, trajectory: np.ndarray) -> np.ndarray:
        """The observable part of a trajectory: the NO slow variables."""
        return trajectory[..., : self.NO]
