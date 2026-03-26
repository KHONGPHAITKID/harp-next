from __future__ import annotations

import torch
import torch.nn as nn


class DepthAwareGate(nn.Module):
    def __init__(self, channels: int, init_bias: float = -2.0) -> None:
        super().__init__()
        self.gate_conv = nn.Conv2d(2 * channels + 2, channels, kernel_size=1, bias=True)
        nn.init.constant_(self.gate_conv.bias, init_bias)

    def forward(
        self,
        u_local: torch.Tensor,
        z_ctx: torch.Tensor,
        valid: torch.Tensor,
        gradmag: torch.Tensor,
    ) -> torch.Tensor:
        gate_in = torch.cat([u_local, z_ctx, valid, gradmag], dim=1)
        gate = torch.sigmoid(self.gate_conv(gate_in))
        return u_local + gate * z_ctx

