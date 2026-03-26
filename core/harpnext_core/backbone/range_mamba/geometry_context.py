from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryContextBuilder(nn.Module):
    def __init__(self, out_channels: int = 128) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.Hardswish(inplace=True),
        )

    def forward(
        self,
        depth0: torch.Tensor,
        intensity0: torch.Tensor,
        valid0: torch.Tensor,
        out_hw: Tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Inputs expected as full-resolution maps: [B,1,H0,W0]
        if depth0.ndim != 4 or intensity0.ndim != 4 or valid0.ndim != 4:
            raise ValueError("range_aux maps must be shaped [B,1,H,W]")

        depth0 = depth0.to(dtype=torch.float32)
        intensity0 = intensity0.to(dtype=torch.float32)
        valid0 = valid0.to(dtype=torch.float32).clamp(0.0, 1.0)

        # Downsample valid using max pooling semantics (keep any-point indicator).
        valid = F.adaptive_max_pool2d(valid0, out_hw)

        # Valid-weighted averages (ratio of adaptive averages == ratio of sums).
        depth_num = F.adaptive_avg_pool2d(depth0 * valid0, out_hw)
        depth_den = F.adaptive_avg_pool2d(valid0, out_hw).clamp_min(1e-6)
        depth = depth_num / depth_den

        inten_num = F.adaptive_avg_pool2d(intensity0 * valid0, out_hw)
        inten_den = F.adaptive_avg_pool2d(valid0, out_hw).clamp_min(1e-6)
        intensity = inten_num / inten_den

        # Masked finite differences at stage resolution.
        dx = depth[:, :, :, 1:] - depth[:, :, :, :-1]
        valid_dx = valid[:, :, :, 1:] * valid[:, :, :, :-1]
        dx = F.pad(dx, (1, 0, 0, 0))
        valid_dx = F.pad(valid_dx, (1, 0, 0, 0))
        dx = dx * valid_dx

        dy = depth[:, :, 1:, :] - depth[:, :, :-1, :]
        valid_dy = valid[:, :, 1:, :] * valid[:, :, :-1, :]
        dy = F.pad(dy, (0, 0, 1, 0))
        valid_dy = F.pad(valid_dy, (0, 0, 1, 0))
        dy = dy * valid_dy

        gradmag = (dx.abs() + dy.abs()) * valid

        geom_in = torch.cat([depth, intensity, valid, dx.abs(), dy.abs()], dim=1)
        geom = self.proj(geom_in)
        return geom, gradmag, valid

