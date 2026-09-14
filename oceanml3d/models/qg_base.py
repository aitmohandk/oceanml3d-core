"""Behaviour shared by every quasi-geostrophic dynamics.

Extracted verbatim from `qg_dynamics.QGDynamics`, where `qg1l_dynamics.QG1LDynamics` held a
byte-identical copy of each of these methods (measured: 100% identical, 68 lines). Only the methods
that were *exactly* the same are here; the ones that merely resemble each other (`_tendency`,
`_invert`, `_rk4_step`, `rollout_*`, `step`, `kinetic_energy`, `enstrophy`) genuinely differ between
the one- and two-layer formulations and stay in their subclasses.

The methods below are polymorphic through `self._grid`, `self._invert` and the layer attributes, so
each subclass keeps its own layer geometry.
"""
from __future__ import annotations

import math

import torch

from oceanml3d.models.dynamics import DynamicsBase


class QGBaseDynamics(DynamicsBase):
    @property
    def device(self) -> torch.device:
        return self.K2.device

    def generate_wind_state(self, num_steps: int,
                            seed: int | None = None,
                            x0: float | None = None,
                            y0: float | None = None) -> torch.Tensor:
        out = torch.zeros(num_steps, 3, dtype=torch.float64)
        if self.wind_amp == 0.0:
            return out.to(self.dtype).to(self.device)
        gen = torch.Generator(device="cpu").manual_seed(seed or self.wind_seed)
        dt = self.dt
        tau_a = self.wind_tau_days * 86400.0
        coeff_a = self.wind_amp * math.sqrt(2.0 / tau_a * dt)
        tau_d = self.wind_drift_tau_days * 86400.0
        coeff_d = self.wind_drift_sigma * math.sqrt(2.0 / tau_d * dt)
        a = torch.zeros((), dtype=torch.float64)
        wx = torch.zeros((), dtype=torch.float64)
        wy = torch.zeros((), dtype=torch.float64)
        x0 = self.L / 2.0 if x0 is None else float(x0)
        y0 = self.W / 2.0 if y0 is None else float(y0)
        for k in range(num_steps):
            a = a - (1.0 / tau_a) * a * dt + coeff_a * torch.randn((), generator=gen)
            wx = wx - (1.0 / tau_d) * wx * dt + coeff_d * torch.randn((), generator=gen)
            wy = wy - (1.0 / tau_d) * wy * dt + coeff_d * torch.randn((), generator=gen)
            xc = (x0 + self.wind_cx * dt * k + wx) % self.L
            yc = (y0 + self.wind_cy * dt * k + wy) % self.W
            out[k, 0] = a
            out[k, 1] = xc
            out[k, 2] = yc
        return out.to(self.dtype).to(self.device)

    def wind_curl_field(self, wind_state: torch.Tensor) -> torch.Tensor:
        wind_state = wind_state.double().to(self.device)
        a = wind_state[..., 0]
        xc = wind_state[..., 1]
        yc = wind_state[..., 2]
        x = self.x_grid.double()
        y = self.y_grid.double()
        n_im = (-1, 0, 1)
        field = torch.zeros(*wind_state.shape[:-1], self.ny, self.nx,
                            dtype=torch.float64, device=self.device)
        sig2 = self.wind_sigma ** 2
        for ix in n_im:
            for iy in n_im:
                dx = x[None, None, :] - (xc - ix * self.L)[..., None, None]
                dy = y[None, :, None] - (yc - iy * self.W)[..., None, None]
                r2 = dx ** 2 + dy ** 2
                field = field + ((1.0 - r2 / (2.0 * sig2))
                                 * torch.exp(-r2 / (2.0 * sig2)))
        return (a[..., None, None] * field).to(self.dtype)

    def _wind_curl_spectral(self, qh: torch.Tensor,
                            wind_state_t) -> torch.Tensor:
        if self.wind_amp == 0.0 or wind_state_t is None \
                or float(wind_state_t[0]) == 0.0:
            return torch.zeros(self.ny, qh.shape[-1],
                               device=qh.device, dtype=qh.dtype)
        curl = self.wind_curl_field(wind_state_t.unsqueeze(0)).squeeze(0)
        curlh = torch.fft.rfft2(curl, dim=(-2, -1))
        cdtype = torch.complex64 if curl.real.dtype == torch.float32 \
            else torch.complex128
        return curlh.to(cdtype)

    def streamfunctions(self, state: torch.Tensor) -> torch.Tensor:
        single = state.dim() == 1
        state_b = state.unsqueeze(0) if single else state
        q = self._grid(state_b)
        qh = torch.fft.rfft2(q, dim=(-2, -1))
        ph = self._invert(qh)
        psi = torch.fft.irfft2(ph, s=(self.ny, self.nx), dim=(-2, -1))
        return psi.squeeze(0) if single else psi
