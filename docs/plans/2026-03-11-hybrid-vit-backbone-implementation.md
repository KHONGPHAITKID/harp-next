# Hybrid ViT Backbone Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `HARPNeXtAttentionBlock` that replaces ConvSENeXt in stages 3–4, and wire it into the backbone via a new `block_type="hybrid"` option.

**Architecture:** Stages 1–2 use ConvSENeXt (local features at high resolution). Stages 3–4 use a transformer encoder block with multi-head self-attention + FFN + learnable 2D positional embeddings (global context at low resolution: 2048 and 512 tokens). The block matches ConvSENeXt's `[B, C, H, W]` → `[B, C, H, W]` interface so the ETP point↔pixel fusion, multi-scale fusion, and decode head remain untouched.

**Tech Stack:** PyTorch (`nn.MultiheadAttention`, `nn.LayerNorm`), timm (`DropPath`)

**Design doc:** `docs/plans/2026-03-11-hybrid-vit-backbone-design.md`

---

### Task 1: HARPNeXtAttentionBlock — Failing Tests

**Files:**
- Create: `tests/backbone/__init__.py`
- Create: `tests/backbone/test_attention_block.py`

**Step 1: Create test directory and test file**

Create `tests/backbone/__init__.py` (empty).

Create `tests/backbone/test_attention_block.py`:

```python
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def test_attention_block_no_downsample():
    """Block with stride=1 preserves spatial dims and channels."""
    from core.harpnext_core.backbone.attention_block import HARPNeXtAttentionBlock
    block = HARPNeXtAttentionBlock(
        inplanes=128, planes=128, stride=1,
        num_heads=8, mlp_ratio=4.0, drop_path=0.0,
    )
    x = torch.randn(2, 128, 16, 128)
    out = block(x)
    assert out.shape == (2, 128, 16, 128)


def test_attention_block_with_downsample():
    """Block with stride=2 halves spatial dims."""
    from core.harpnext_core.backbone.attention_block import HARPNeXtAttentionBlock
    block = HARPNeXtAttentionBlock(
        inplanes=128, planes=128, stride=2,
        num_heads=8, mlp_ratio=4.0, drop_path=0.0,
    )
    x = torch.randn(2, 128, 32, 256)
    out = block(x)
    assert out.shape == (2, 128, 16, 128)


def test_attention_block_channel_change():
    """Block handles inplanes != planes via downsample."""
    from core.harpnext_core.backbone.attention_block import HARPNeXtAttentionBlock
    block = HARPNeXtAttentionBlock(
        inplanes=64, planes=128, stride=1,
        num_heads=8, mlp_ratio=4.0, drop_path=0.0,
    )
    x = torch.randn(2, 64, 16, 128)
    out = block(x)
    assert out.shape == (2, 128, 16, 128)


def test_attention_block_gradient_flows():
    """Verify gradients flow through the block."""
    from core.harpnext_core.backbone.attention_block import HARPNeXtAttentionBlock
    block = HARPNeXtAttentionBlock(
        inplanes=128, planes=128, stride=1,
        num_heads=8, mlp_ratio=4.0, drop_path=0.0,
    )
    x = torch.randn(2, 128, 8, 64, requires_grad=True)
    out = block(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_attention_block_stage3_shapes():
    """Simulate stage 3: input 32x256 with stride=2 → 16x128."""
    from core.harpnext_core.backbone.attention_block import HARPNeXtAttentionBlock
    block = HARPNeXtAttentionBlock(
        inplanes=128, planes=128, stride=2,
        num_heads=8, mlp_ratio=4.0, drop_path=0.1,
    )
    x = torch.randn(1, 128, 32, 256)
    out = block(x)
    assert out.shape == (1, 128, 16, 128)


def test_attention_block_stage4_shapes():
    """Simulate stage 4: input 16x128 with stride=2 → 8x64."""
    from core.harpnext_core.backbone.attention_block import HARPNeXtAttentionBlock
    block = HARPNeXtAttentionBlock(
        inplanes=128, planes=128, stride=2,
        num_heads=8, mlp_ratio=4.0, drop_path=0.1,
    )
    x = torch.randn(1, 128, 16, 128)
    out = block(x)
    assert out.shape == (1, 128, 8, 64)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/backbone/test_attention_block.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.harpnext_core.backbone.attention_block'`

**Step 3: Commit**

```bash
git add tests/backbone/
git commit -m "test: add failing tests for HARPNeXtAttentionBlock"
```

---

### Task 2: Implement HARPNeXtAttentionBlock

**Files:**
- Create: `core/harpnext_core/backbone/attention_block.py`

**Step 1: Write the implementation**

Create `core/harpnext_core/backbone/attention_block.py`:

```python
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
```

**Step 2: Run tests to verify they pass**

Run: `pytest tests/backbone/test_attention_block.py -v`
Expected: All 6 tests PASS

**Step 3: Commit**

```bash
git add core/harpnext_core/backbone/attention_block.py
git commit -m "feat: add HARPNeXtAttentionBlock with MHSA + FFN + 2D pos embed"
```

---

### Task 3: Wire Hybrid Block Type into Backbone — Failing Tests

**Files:**
- Create: `tests/backbone/test_hybrid_backbone.py`

**Step 1: Write failing test**

Create `tests/backbone/test_hybrid_backbone.py`:

```python
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def _make_voxel_dict(batch_size=2, num_points=500, num_voxels=200):
    """Create a minimal voxel_dict that mirrors FeaturesEncoder output."""
    voxel_feats = torch.randn(num_voxels, 16)

    # Coordinates: [batch_idx, y, x] with valid ranges for 64x512
    coors_list = []
    for b in range(batch_size):
        n = num_points // batch_size
        ys = torch.randint(0, 64, (n,))
        xs = torch.randint(0, 512, (n,))
        bs = torch.full((n,), b, dtype=torch.long)
        coors_list.append(torch.stack([bs, ys, xs], dim=1))
    coors = torch.cat(coors_list, dim=0)

    # Voxel coors: unique subset
    voxel_coors_list = []
    for b in range(batch_size):
        nv = num_voxels // batch_size
        ys = torch.randint(0, 64, (nv,))
        xs = torch.randint(0, 512, (nv,))
        bs = torch.full((nv,), b, dtype=torch.long)
        voxel_coors_list.append(torch.stack([bs, ys, xs], dim=1))
    voxel_coors = torch.cat(voxel_coors_list, dim=0)

    N = coors.shape[0]
    point_feats = [
        torch.randn(N, 64),
        torch.randn(N, 128),
        torch.randn(N, 256),
        torch.randn(N, 256),
    ]

    return {
        'voxel_feats': voxel_feats,
        'voxel_coors': voxel_coors,
        'coors': coors,
        'point_feats': point_feats,
    }


def test_hybrid_backbone_forward_shapes():
    """Hybrid backbone produces correct output shapes."""
    from core.harpnext_core.backbone.harpnext_backbone import HARPNeXtBackbone
    backbone = HARPNeXtBackbone(
        in_channels=16,
        point_in_channels=384,
        output_shape=[64, 512],
        depth=10,
        stem_channels=128,
        num_stages=4,
        out_channels=[128, 128, 128, 128],
        strides=[1, 2, 2, 2],
        dilations=[3, 3, 3, 3],
        fuse_channels=[256, 128],
        block_type="hybrid",
        block_cfg={
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "drop_path": 0.0,
            "attn_drop": 0.0,
        },
    )
    backbone = backbone.cpu()
    voxel_dict = _make_voxel_dict()
    result = backbone(voxel_dict)

    # voxel_feats[0] is fused: [B, 128, 64, 512]
    assert result['voxel_feats'][0].shape == (2, 128, 64, 512)
    # 5 entries: fused + stem + 4 stages
    assert len(result['voxel_feats']) == 5
    # point_feats_backbone[0] is fused: [N, 128]
    assert result['point_feats_backbone'][0].shape[1] == 128


def test_hybrid_backbone_uses_both_block_types():
    """Stages 1-2 use ConvSENeXt, stages 3-4 use AttentionBlock."""
    from core.harpnext_core.backbone.harpnext_backbone import HARPNeXtBackbone
    from core.harpnext_core.backbone.attention_block import HARPNeXtAttentionBlock

    backbone = HARPNeXtBackbone(
        in_channels=16,
        point_in_channels=384,
        output_shape=[64, 512],
        depth=10,
        block_type="hybrid",
        block_cfg={"num_heads": 8, "mlp_ratio": 4.0, "drop_path": 0.0},
    )

    # Check block types by inspecting the res_layers
    layer1 = getattr(backbone, 'layer1')  # stage 1 → ConvSENeXt
    layer2 = getattr(backbone, 'layer2')  # stage 2 → ConvSENeXt
    layer3 = getattr(backbone, 'layer3')  # stage 3 → AttentionBlock
    layer4 = getattr(backbone, 'layer4')  # stage 4 → AttentionBlock

    assert not isinstance(layer1[0], HARPNeXtAttentionBlock)
    assert not isinstance(layer2[0], HARPNeXtAttentionBlock)
    assert isinstance(layer3[0], HARPNeXtAttentionBlock)
    assert isinstance(layer4[0], HARPNeXtAttentionBlock)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/backbone/test_hybrid_backbone.py -v`
Expected: FAIL — `KeyError: "invalid block_type hybrid for HARPNeXtBackbone."`

**Step 3: Commit**

```bash
git add tests/backbone/test_hybrid_backbone.py
git commit -m "test: add failing tests for hybrid backbone integration"
```

---

### Task 4: Wire Hybrid Block Type into Backbone — Implementation

**Files:**
- Modify: `core/harpnext_core/backbone/harpnext_backbone.py`

**Step 1: Modify backbone to support `block_type="hybrid"`**

In `harpnext_backbone.py`, make these changes:

1. **In `__init__`**, after the TinyViM check (line 179-183), add hybrid handling:

Replace:
```python
        if self.block_type in ("tinyvim", "tvim"):
            from core.tinyvim_core.tvimblock import HARPNeXtTinyViMBlock
            self.block = HARPNeXtTinyViMBlock
        elif self.block_type not in ("convsenext", "convsennext", "convse"):
            raise KeyError(f"invalid block_type {block_type} for HARPNeXtBackbone.")
```

With:
```python
        if self.block_type in ("tinyvim", "tvim"):
            from core.tinyvim_core.tvimblock import HARPNeXtTinyViMBlock
            self.block = HARPNeXtTinyViMBlock
        elif self.block_type == "hybrid":
            from core.harpnext_core.backbone.attention_block import HARPNeXtAttentionBlock
            self.attn_block = HARPNeXtAttentionBlock
        elif self.block_type not in ("convsenext", "convsennext", "convse"):
            raise KeyError(f"invalid block_type {block_type} for HARPNeXtBackbone.")
```

2. **In `_make_res_layer`**, add hybrid handling. The hybrid mode picks the block class based on the `index` parameter (0-1 → ConvSENeXt, 2-3 → AttentionBlock).

Replace the entire `_make_res_layer` method with:

```python
    def _make_res_layer(self, block: nn.Module, inplanes, planes, num_blocks, stride, dilation, dw_conv_kernel, dw_conv_bias, index: int = 0):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False, device=self.device),
                nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01, device=self.device),
            )

        # Hybrid mode: use attention blocks for later stages (index >= 2)
        use_attention = self.block_type == "hybrid" and index >= 2

        layers = []
        if use_attention:
            layers.append(
                self.attn_block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    dilation=dilation,
                    downsample=downsample,
                    **self.block_cfg))
        elif self.block_type in ("tinyvim", "tvim"):
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    dilation=dilation,
                    downsample=downsample,
                    index=index,
                    **self.block_cfg))
        else:
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    dilation=dilation,
                    downsample=downsample,
                    norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
                    act_cfg=dict(type='HSwish', inplace=True),
                    dw_conv_kernel=dw_conv_kernel,
                    dw_conv_bias=dw_conv_bias))
        inplanes = planes
        for _ in range(1, num_blocks):
            if use_attention:
                layers.append(
                    self.attn_block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        dilation=dilation,
                        downsample=None,
                        **self.block_cfg))
            elif self.block_type in ("tinyvim", "tvim"):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        dilation=dilation,
                        downsample=None,
                        index=index,
                        **self.block_cfg))
            else:
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        dilation=dilation,
                        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
                        act_cfg=dict(type='HSwish', inplace=True),
                        dw_conv_kernel=dw_conv_kernel,
                        dw_conv_bias=dw_conv_bias))

        return nn.Sequential(*layers)
```

**Step 2: Run all backbone tests**

Run: `pytest tests/backbone/ -v`
Expected: All tests PASS (both attention_block and hybrid_backbone tests)

**Step 3: Run existing tests to check for regressions**

Run: `pytest tests/ -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add core/harpnext_core/backbone/harpnext_backbone.py
git commit -m "feat: wire hybrid block_type into HARPNeXtBackbone"
```

---

### Task 5: Add Hybrid Config File

**Files:**
- Create: `configs/net/harpnext-semantickitti-hybrid.yaml`

**Step 1: Create the config**

Copy from `configs/net/harpnext-semantickitti-tinyvim.yaml` and change `block_type` and `block_cfg`:

```yaml
# Copyright 2025 CEA LIST - Samir Abou Haidar

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

model:
  name: harpnext

  voxel_encoder:
    in_channels: 4
    feat_channels: [64, 128, 256, 256]
    with_distance: True
    with_cluster_center: True
    with_pre_norm: True
    feat_compression: 16

  backbone:
    in_channels: 16
    point_in_channels: 384
    output_shape: [64, 512]
    depth: 10
    block_type: hybrid
    block_cfg:
      num_heads: 8
      mlp_ratio: 4.0
      drop_path: 0.1
      attn_drop: 0.0
    stem_channels: 128
    num_stages: 4
    out_channels: [128, 128, 128, 128]
    strides: [1, 2, 2, 2]
    dilations: [3, 3, 3, 3]
    fuse_channels: [256, 128]
    dw_conv_kernel: 7
    dw_conv_bias: True
    inter_align_corners: True

  decode_head:
    in_channels: 128
    middle_channels: [128, 256, 128, 64]
    channels: 64
    dropout_ratio: 0
    num_classes: 19
    conv_seg_kernel_size: 1

  auxiliary: True
  auxiliary_heads:
    channels: 128
    dropout_ratio: 0
    conv_seg_kernel_size: 1

classif:
  nb_class: 19
  ignore_class: 255

augmentations:
  pointsample: 0.9
  randomflip:
    sync_2d: False
    flip_ratio_bev_horizontal: 0.5
    flip_ratio_bev_vertical: 0.5
  GlobalRotScaleTrans:
    rot_range: [-3.1415926, 3.1415926]
    scale_ratio_range: [0.95, 1.05]
    translation_std: [0.1, 0.1, 0.1]

input_feat:
  - "intensity"

range_proj:
  range_H: 64
  range_W: 512
  fov_up: 3.0
  fov_down: -25.0

train:
  lamda: 1.0

preproc:
  gpu: False #Set to False when Training (PreProc on CPU)
```

**Step 2: Commit**

```bash
git add configs/net/harpnext-semantickitti-hybrid.yaml
git commit -m "feat: add SemanticKITTI config for hybrid backbone"
```

---

### Task 6: Update CHANGELOG

**Files:**
- Modify: `CHANGELOG.md`

**Step 1: Add entry**

Append to `CHANGELOG.md`:

```markdown
## 2026-03-11

### Added: Hybrid ViT Backbone (Conv + Attention)
- Added `HARPNeXtAttentionBlock` — a transformer encoder block (MHSA + FFN + learnable 2D positional embeddings) that matches the ConvSENeXt `[B, C, H, W]` interface
- New `block_type="hybrid"` for `HARPNeXtBackbone`: stages 1–2 use ConvSENeXt, stages 3–4 use attention blocks for global context
- Added SemanticKITTI config: `configs/net/harpnext-semantickitti-hybrid.yaml`

**Files:**
- `core/harpnext_core/backbone/attention_block.py` (new)
- `core/harpnext_core/backbone/harpnext_backbone.py` (modified)
- `configs/net/harpnext-semantickitti-hybrid.yaml` (new)
- `tests/backbone/test_attention_block.py` (new)
- `tests/backbone/test_hybrid_backbone.py` (new)
```

**Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: update CHANGELOG with hybrid ViT backbone"
```
