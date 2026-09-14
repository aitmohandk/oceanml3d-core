"""Classical data assimilation as ordinary oceanml3d models.

Ported from ``4dvarnet-fm-opencode/evaluation/baselines.py`` (EnKF with multiplicative inflation
and Gaspari-Cohn localisation) and from the OI placeholder of ``4dvarnet-ocean-reanalyses``.
They are the references every learned method is judged against, so they must produce products
through the same path — hence they are registered models, not a separate script.

They carry no learnable weight: ``training`` is a no-op pass (Lightning still needs one
parameter, kept as a scalar bias that stays at zero unless you actually optimise it).
"""
from __future__ import annotations

import numpy as np
import torch

from oceanml3d.models.ocean.base import BaseOceanModel
from oceanml3d.registry import register_model
from oceanml3d.variables import VariableSet


def gaspari_cohn(z: np.ndarray) -> np.ndarray:
    """Compactly supported correlation function (Gaspari & Cohn 1999), z = distance / radius."""
    z = np.abs(np.asarray(z, float))
    out = np.zeros_like(z)
    m = z <= 1
    out[m] = (((-0.25 * z[m] + 0.5) * z[m] + 0.625) * z[m] - 5.0 / 3.0) * z[m] ** 2 + 1
    m = (z > 1) & (z <= 2)
    out[m] = ((((z[m] / 12 - 0.5) * z[m] + 0.625) * z[m] + 5.0 / 3.0) * z[m] - 5) * z[m] + 4 - 2.0 / (3 * z[m])
    return out


def periodic_distance(n: int) -> np.ndarray:
    i = np.arange(n)
    d = np.abs(i[:, None] - i[None, :])
    return np.minimum(d, n - d)


class _NoTrainModel(BaseOceanModel):
    """A model whose forward is a fixed algorithm: gradients are irrelevant but Lightning needs one parameter."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, "train")
        return loss.detach() + 0.0 * self.dummy.sum()


@register_model("optimal_interpolation")
class OptimalInterpolation(_NoTrainModel):
    """Gaussian-kernel OI of the observed channel onto each target (the classical L4 mapping).

    ``length_scale`` and ``time_scale`` are in grid cells / time steps; ``obs_var`` and ``bg_var``
    are the observation and background error variances in normalised units.
    """

    def __init__(self, variables: VariableSet, window: int, rec_weight: np.ndarray,
                 obs_map: dict[str, str] | None = None, length_scale: float = 4.0, time_scale: float = 2.0,
                 obs_var: float = 0.05, bg_var: float = 1.0, **base_kw):
        super().__init__(variables, window, rec_weight, **base_kw)
        self.save_hyperparameters(ignore=["variables", "rec_weight", "norm_stats"])
        in_names = [s.name for s in variables.inputs]
        tgt = variables.targets
        obs_map = dict(obs_map or {})
        for s in tgt:
            if s.role.value == "both":
                obs_map.setdefault(s.name, s.name)
        if not obs_map:
            obs_map = {t.name: in_names[min(i, len(in_names) - 1)] for i, t in enumerate(tgt)}
        names = [t.name for t in tgt]
        unknown = [s for s in obs_map.values() if s not in in_names]
        if unknown:
            raise ValueError(f"optimal_interpolation: observed input(s) {unknown} are not input variables")
        self.pairs = [(names.index(t), in_names.index(src)) for t, src in obs_map.items() if t in names]
        self.length_scale, self.time_scale = length_scale, time_scale
        self.obs_var, self.bg_var = obs_var, bg_var

    def _kernel(self, t: int, h: int, w: int, device) -> torch.Tensor:
        ti = torch.arange(t, device=device, dtype=torch.float32)
        yi = torch.arange(h, device=device, dtype=torch.float32)
        xi = torch.arange(w, device=device, dtype=torch.float32)
        k = torch.exp(-0.5 * ((ti[:, None] - ti[None, :]) / max(self.time_scale, 1e-6)) ** 2)
        ky = torch.exp(-0.5 * ((yi[:, None] - yi[None, :]) / max(self.length_scale, 1e-6)) ** 2)
        kx = torch.exp(-0.5 * ((xi[:, None] - xi[None, :]) / max(self.length_scale, 1e-6)) ** 2)
        return k, ky, kx

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        raw = batch.index_select(1, self._in_idx.to(batch.device))
        b, _, t, h, w = raw.shape
        out = torch.zeros(b, len(self.variables.targets), t, h, w, device=batch.device, dtype=batch.dtype)
        kt, ky, kx = self._kernel(t, h, w, batch.device)
        for ti_, oi in self.pairs:
            obs = raw[:, oi]
            mask = torch.isfinite(obs).float()
            filled = obs.nan_to_num()
            # separable Gaussian smoothing of (masked field, mask) = Nadaraya-Watson / diagonal-B OI
            num = torch.einsum("ts,bshw->bthw", kt, filled)
            den = torch.einsum("ts,bshw->bthw", kt, mask)
            num = torch.einsum("ij,btjw->btiw", ky, num)
            den = torch.einsum("ij,btjw->btiw", ky, den)
            num = torch.einsum("ij,bthj->bthi", kx, num)
            den = torch.einsum("ij,bthj->bthi", kx, den)
            gain = self.bg_var / (self.bg_var + self.obs_var)
            out[:, ti_] = gain * num / den.clamp_min(1e-6)
        return out


@register_model("enkf")
class EnsembleKalmanFilter(_NoTrainModel):
    """Stochastic EnKF over a known dynamics, with multiplicative inflation and GC localisation.

    Meant for the toy tasks (``data=lorenz96``), where the state is stored as ``(time, 1, K)``: the
    filter runs along the window's time axis, one analysis per step, and returns the analysis
    trajectory. ``predict_step`` can return the whole ensemble (``return_ensemble=True``), which the
    exporter writes as a member dimension (product format v2).
    """

    def __init__(self, variables: VariableSet, window: int, rec_weight: np.ndarray,
                 dynamics: str = "lorenz96", dynamics_kw: dict | None = None, n_members: int = 20,
                 inflation: float = 1.02, obs_noise: float = 1.0, localization_radius: float | None = 4.0,
                 steps_per_window: int = 1, return_ensemble: bool = False, seed: int = 0, **base_kw):
        super().__init__(variables, window, rec_weight, **base_kw)
        self.save_hyperparameters(ignore=["variables", "rec_weight", "norm_stats"])
        from oceanml3d.dynamics import get_dynamics

        self.dyn = get_dynamics(dynamics, **(dynamics_kw or {}))
        self.n_members, self.inflation, self.obs_noise = n_members, inflation, obs_noise
        self.localization_radius, self.steps_per_window = localization_radius, steps_per_window
        self.return_ensemble, self.seed = return_ensemble, seed
        in_names = [s.name for s in variables.inputs]
        self.obs_idx = in_names.index("obs") if "obs" in in_names else 0

    def _localisation(self, k: int) -> np.ndarray:
        if not self.localization_radius:
            return np.ones((k, k))
        return gaspari_cohn(periodic_distance(k) / self.localization_radius)

    def _assimilate(self, obs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """obs: (T, K) in normalised units with NaN where unobserved; returns (n_members, T, K)."""
        t, k = obs.shape
        rng = np.random.default_rng(self.seed)
        phys = obs * std + mean                                  # the filter works in physical units
        loc = self._localisation(k)
        ens = phys[0][None] + rng.normal(0, self.obs_noise, (self.n_members, k))
        ens = np.where(np.isfinite(ens), ens, rng.normal(0, 1, ens.shape))
        out = np.empty((self.n_members, t, k))
        for i in range(t):
            if i > 0:
                for _ in range(self.steps_per_window):
                    ens = self.dyn.step(ens)
            m = ens.mean(0)
            ens = m + self.inflation * (ens - m)
            seen = np.isfinite(phys[i])
            if seen.any():
                y = phys[i][seen]
                hx = ens[:, seen]
                anom = ens - ens.mean(0)
                hanom = hx - hx.mean(0)
                cov = (anom.T @ hanom) / (self.n_members - 1) * loc[:, seen]
                hcov = (hanom.T @ hanom) / (self.n_members - 1) * loc[np.ix_(seen, seen)]
                gain = cov @ np.linalg.inv(hcov + self.obs_noise ** 2 * np.eye(seen.sum()))
                pert = rng.normal(0, self.obs_noise, (self.n_members, seen.sum()))
                ens = ens + (y[None] + pert - hx) @ gain.T
            out[:, i] = ens
        return (out - mean) / std                                # back to normalised units

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        ens = self._ensemble(batch)
        return ens.mean(0)

    def _ensemble(self, batch: torch.Tensor) -> torch.Tensor:
        raw = batch.index_select(1, self._in_idx.to(batch.device))[:, self.obs_idx]     # (B, T, H, W)
        b, t, h, w = raw.shape
        mean, std = (0.0, 1.0) if self.norm_stats is None else (
            float(self.norm_stats[0][self.variables.target_indices[0]]),
            float(self.norm_stats[1][self.variables.target_indices[0]]))
        arr = raw.detach().cpu().numpy()
        n_tgt = len(self.variables.targets)
        out = np.empty((self.n_members, b, n_tgt, t, h, w), dtype=np.float32)
        for bi in range(b):
            for hi in range(h):
                a = self._assimilate(arr[bi, :, hi, :], mean, std)                     # (M, T, K)
                out[:, bi, :, :, hi, :] = a[:, None]
        return torch.as_tensor(out, device=batch.device)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        ens = self._ensemble(batch) if self.return_ensemble else self._ensemble(batch).mean(0, keepdim=True)
        if self.norm_stats is not None:
            mean, std = self.norm_stats
            idx = self.variables.target_indices
            m = torch.as_tensor(mean[idx], device=ens.device)[None, None, :, None, None, None]
            s = torch.as_tensor(std[idx], device=ens.device)[None, None, :, None, None, None]
            ens = ens * s + m
        return ens[0] if not self.return_ensemble else ens
