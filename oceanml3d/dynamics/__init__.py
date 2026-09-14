"""Known dynamical models, used by classical data assimilation and by the toy systems on which
methods are developed before touching ocean data (Lorenz-63, Lorenz-96 — from 4dvarnet-fm-opencode).

A dynamics is a plain callable ``step(state, dt) -> state`` on numpy arrays plus a ``simulate``;
no torch, so it can generate data and drive an EnKF without the training stack.
"""
from oceanml3d.dynamics import lorenz  # noqa: F401,E402
from oceanml3d.dynamics.base import (  # noqa: F401
    Dynamics,
    get_dynamics,
    list_dynamics,
    register_dynamics,
)
