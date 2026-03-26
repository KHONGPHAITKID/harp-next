from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .axial_mamba import CircularBiMamba1D
from .depth_gate import DepthAwareGate
from .geometry_context import GeometryContextBuilder


class SELite(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.se1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.se2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.se1(w), inplace=True)
        w = F.hardsigmoid(self.se2(w), inplace=True)
        return x * w


class RangeMambaSECore(nn.Module):
    """Context+gating core applied after the local conv stem (stage resolution)."""

    def __init__(
        self,
        channels: int = 128,
        use_col_mamba: bool = False,
        reduction: int = 16,
        d_state: int = 16,
        expand: int = 2,
        backend: str = "mamba",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.use_col_mamba = use_col_mamba

        self.geom_builder = GeometryContextBuilder(out_channels=channels)
        self.row = CircularBiMamba1D(
            channels=channels, d_state=d_state, expand=expand, backend=backend, seam_shift=True
        )
        if use_col_mamba:
            self.col = CircularBiMamba1D(
                channels=channels, d_state=d_state, expand=expand, backend=backend, seam_shift=False
            )
            self.mix = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False)
        else:
            self.col = None
            self.mix = None

        self.gate = DepthAwareGate(channels=channels)
        self.se = SELite(channels=channels, reduction=reduction)

    def forward(self, u_local: torch.Tensor, range_aux: dict) -> torch.Tensor:
        # u_local: [B,C,H,W]
        if range_aux is None:
            raise ValueError("range_aux is required for RangeMambaSECore")
        depth0 = range_aux.get("depth")
        intensity0 = range_aux.get("intensity")
        valid0 = range_aux.get("valid")
        if depth0 is None or intensity0 is None or valid0 is None:
            raise ValueError("range_aux must contain depth/intensity/valid")

        b, c, h, w = u_local.shape
        if c != self.channels:
            raise ValueError(f"u_local channels {c} must match configured channels {self.channels}")

        depth0 = depth0.to(device=u_local.device, dtype=torch.float32)
        intensity0 = intensity0.to(device=u_local.device, dtype=torch.float32)
        valid0 = valid0.to(device=u_local.device, dtype=torch.float32)

        geom, gradmag, valid_s = self.geom_builder(depth0, intensity0, valid0, out_hw=(h, w))

        z = self.row(u_local, geom, valid_s, axis="row")
        if self.use_col_mamba:
            assert self.col is not None and self.mix is not None
            z_col = self.col(z, geom, valid_s, axis="col")
            z = self.mix(torch.cat([z, z_col], dim=1))

        y = self.gate(u_local, z, valid_s, gradmag)
        y = self.se(y)
        return y

