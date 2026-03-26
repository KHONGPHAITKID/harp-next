from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn


def _import_mamba():
    try:
        from mamba_ssm import Mamba  # type: ignore

        return Mamba
    except Exception:
        pass
    try:
        from mamba_ssm.modules.mamba_simple import Mamba  # type: ignore

        return Mamba
    except Exception as e:
        raise ImportError(
            "mamba-ssm is required for RangeMamba blocks. "
            "Install it (optionally with causal-conv1d) then retry."
        ) from e


class SelectiveScan1D(nn.Module):
    """Thin adapter to decouple the codebase from the concrete mamba-ssm class."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        backend: str = "mamba",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.d_model = d_model
        self._block: Optional[nn.Module]

        if backend == "identity":
            self._block = nn.Identity()
        elif backend == "mamba":
            Mamba = _import_mamba()
            self._block = Mamba(d_model=d_model, d_state=d_state, expand=expand, **kwargs)
        else:
            raise ValueError(f"Unknown SelectiveScan1D backend: {backend}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        if self._block is None:
            raise RuntimeError("SelectiveScan1D is not initialized")
        return self._block(x)

