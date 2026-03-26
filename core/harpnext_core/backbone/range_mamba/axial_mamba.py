from __future__ import annotations

import torch
import torch.nn as nn

from .mamba_adapter import SelectiveScan1D


class CircularBiMamba1D(nn.Module):
    def __init__(
        self,
        channels: int,
        d_state: int = 16,
        expand: int = 2,
        backend: str = "mamba",
        seam_shift: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.seam_shift = seam_shift

        self.ln = nn.LayerNorm(channels)
        self.mamba_f = SelectiveScan1D(d_model=channels, d_state=d_state, expand=expand, backend=backend)
        self.mamba_b = SelectiveScan1D(d_model=channels, d_state=d_state, expand=expand, backend=backend)
        self.proj = nn.Linear(2 * channels, channels, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        geom: torch.Tensor,
        valid: torch.Tensor,
        axis: str,
    ) -> torch.Tensor:
        # x, geom: [B,C,H,W], valid: [B,1,H,W]
        b, c, h, w = x.shape
        if geom.shape != x.shape:
            raise ValueError(f"geom shape {geom.shape} must match x shape {x.shape}")
        if valid.shape[:2] != (b, 1) or valid.shape[2:] != (h, w):
            raise ValueError(f"valid shape {valid.shape} must be [B,1,H,W] == [{b},1,{h},{w}]")

        s = (x + geom).permute(0, 2, 3, 1).contiguous()  # [B,H,W,C]
        s = self.ln(s)
        valid_hw = valid.permute(0, 2, 3, 1).contiguous()  # [B,H,W,1]

        if axis == "row":
            seq = s.view(b * h, w, c)
            mask = valid_hw.view(b * h, w, 1)
            seq = seq * mask

            if self.seam_shift:
                shift = w // 2
                seq = torch.roll(seq, shifts=shift, dims=1)

            zf = self.mamba_f(seq)
            zb = torch.flip(self.mamba_b(torch.flip(seq, dims=[1])), dims=[1])
            z = self.proj(torch.cat([zf, zb], dim=-1))

            if self.seam_shift:
                z = torch.roll(z, shifts=-shift, dims=1)

            z = z * mask
            z = z.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
            return z

        if axis == "col":
            s_col = s.permute(0, 2, 1, 3).contiguous()  # [B,W,H,C]
            valid_col = valid_hw.permute(0, 2, 1, 3).contiguous()  # [B,W,H,1]

            seq = s_col.view(b * w, h, c)
            mask = valid_col.view(b * w, h, 1)
            seq = seq * mask

            zf = self.mamba_f(seq)
            zb = torch.flip(self.mamba_b(torch.flip(seq, dims=[1])), dims=[1])
            z = self.proj(torch.cat([zf, zb], dim=-1))
            z = z * mask

            z = z.view(b, w, h, c).permute(0, 2, 1, 3).contiguous()  # [B,H,W,C]
            z = z.permute(0, 3, 1, 2).contiguous()  # [B,C,H,W]
            return z

        raise ValueError(f"axis must be 'row' or 'col', got: {axis}")

