from typing import Optional
import torch
import torch.nn as nn
from timm.models.layers import DropPath


class HARPNeXtAttentionBlock(nn.Module):
    """Transformer encoder block matching ConvSENeXt's [B, C, H, W] interface.

    Used in stages 3-4 of the hybrid backbone for global context via MHSA.
    """

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        attn_drop: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()

        # Downsample if stride > 1 or channel mismatch
        if downsample is not None:
            self.downsample = downsample
        elif stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01),
            )
        else:
            self.downsample = None

        # Learnable 2D positional embedding (initialized lazily in forward)
        self.pos_embed = None
        self._pos_h = 0
        self._pos_w = 0

        # Attention
        self.norm1 = nn.LayerNorm(planes)
        self.attn = nn.MultiheadAttention(
            embed_dim=planes,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # FFN
        self.norm2 = nn.LayerNorm(planes)
        mlp_hidden = int(planes * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(planes, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, planes),
        )

    def _get_pos_embed(self, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return additive positional embedding [1, H*W, C]."""
        if self.pos_embed is None or self._pos_h != H or self._pos_w != W:
            C = self.norm1.normalized_shape[0]
            self.pos_embed = nn.Parameter(
                torch.zeros(1, H * W, C, device=device, dtype=dtype)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            self._pos_h = H
            self._pos_w = W
        return self.pos_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Downsample first (reduces tokens before attention)
        if self.downsample is not None:
            x = self.downsample(x)

        B, C, H, W = x.shape

        # Reshape to sequence: [B, H*W, C]
        tokens = x.flatten(2).transpose(1, 2)

        # Add positional embedding
        tokens = tokens + self._get_pos_embed(H, W, x.device, x.dtype)

        # MHSA
        residual = tokens
        tokens = self.norm1(tokens)
        tokens = residual + self.drop_path(self.attn(tokens, tokens, tokens, need_weights=False)[0])

        # FFN
        residual = tokens
        tokens = self.norm2(tokens)
        tokens = residual + self.drop_path(self.ffn(tokens))

        # Reshape back to [B, C, H, W]
        out = tokens.transpose(1, 2).view(B, C, H, W)
        return out
