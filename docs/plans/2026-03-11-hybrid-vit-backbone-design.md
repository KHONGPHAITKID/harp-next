# Hybrid ViT Backbone Design

**Date:** 2026-03-11
**Goal:** Improve mIoU by replacing ConvSENeXt blocks in stages 3–4 with multi-head self-attention blocks, while keeping ConvSENeXt for stages 1–2.

## Motivation

ConvSENeXt blocks have a limited receptive field (7×7 depthwise conv + SE). At lower-resolution stages (16×128 and 8×64), the token count is small enough (2048 and 512) that full self-attention is computationally tractable and provides global context that could improve segmentation of large objects and contextual reasoning.

## Architecture

```
Stem (unchanged)
  |
Stage 1: ConvSENeXt  [B, 128, 64, 512]   — local features, full res
Stage 2: ConvSENeXt  [B, 128, 32, 256]   — local features, 2x downsample
Stage 3: AttentionBlock [B, 128, 16, 128] — global context, 4x cumulative stride
Stage 4: AttentionBlock [B, 128, 8, 64]   — global context, 8x cumulative stride
  |
Multi-scale aggregation (unchanged)
```

## AttentionBlock

A transformer encoder block wrapped to match the ConvSENeXt interface: takes `[B, C, H, W]`, returns `[B, C, H, W]`.

### Internal Architecture

```
Input: [B, C, H, W]
  |
  ├─ downsample (if stride > 1 or channel mismatch):
  |    Conv2d(in, out, 1x1, stride) + BN
  |
  ├─ Reshape to [B, H*W, C]
  ├─ + learnable 2D positional embedding
  |
  ├─ LayerNorm
  ├─ Multi-Head Self-Attention (heads=8, dim=128)
  ├─ DropPath + Residual
  |
  ├─ LayerNorm
  ├─ FFN: Linear(C, 4C) → GELU → Linear(4C, C)
  ├─ DropPath + Residual
  |
  ├─ Reshape to [B, C, H', W']
  |
Output: [B, C, H', W']
```

### Design Decisions

1. **Positional encoding:** Learnable 2D positional embeddings decomposed into H and W components (added, not concatenated). Fixed grid size since range images always have the same resolution per stage.

2. **Downsampling:** Reuses the same 1×1 conv + BN downsample logic as ConvSENeXt. Applied before attention to reduce token count.

3. **Heads/dim:** 8 heads with 128-dim features → 16-dim per head. Standard for this feature size.

4. **FFN:** Simple 2-layer MLP with 4× expansion. No depthwise conv — keep it clean for v1. Can explore ConvFFN later if needed.

5. **DropPath:** Same rate as used in TinyViM blocks (configurable via `block_cfg`).

### Token Counts

| Stage | Feature Map | Tokens | MHSA Cost (relative) |
|-------|------------|--------|---------------------|
| 3     | 16 × 128   | 2,048  | 1.0×                |
| 4     | 8 × 64     | 512    | 0.06×               |

Both are well within practical limits for standard attention.

## Integration

### New block_type: "hybrid"

```python
block_type="hybrid"
```

In `_make_res_layer`, the backbone will select:
- `ConvSENeXt` for stages 0–1 (index 0, 1)
- `HARPNeXtAttentionBlock` for stages 2–3 (index 2, 3)

The stage index determines which block class is instantiated.

### Config Example

```yaml
backbone:
  block_type: hybrid
  block_cfg:
    num_heads: 8
    mlp_ratio: 4.0
    drop_path: 0.1
    attn_drop: 0.0
```

### What Stays Unchanged

- Stem (pixel, point, fusion)
- ETP pipeline (point2cluster, cluster2pixel, pixel2point)
- Per-stage point fusion layers (Linear + BN1d + ReLU)
- Multi-scale pixel fusion + attention layers (Eq. 11-13)
- Final multi-scale aggregation
- Decode head and aux heads
- Loss computation

## New File

`core/harpnext_core/backbone/attention_block.py` — contains `HARPNeXtAttentionBlock` class.

## Modifications

- `core/harpnext_core/backbone/harpnext_backbone.py`:
  - Import `HARPNeXtAttentionBlock`
  - Handle `block_type="hybrid"` in `__init__` and `_make_res_layer`
  - For hybrid mode, use two different block classes depending on stage index

## Testing

- Unit test for `HARPNeXtAttentionBlock`: verify input/output shapes match ConvSENeXt interface
- Unit test for hybrid backbone: verify forward pass produces correct output shapes
- Integration test: verify full model (encoder → hybrid backbone → head) runs without errors
- Benchmark: compare parameter count and FLOPs against ConvSENeXt and TinyViM configs

## Expected Impact

- **mIoU:** +0.5–2.0% improvement from global context in later stages
- **Parameters:** ~10-15% increase (attention layers are parameter-efficient at 128-dim)
- **Latency:** ~5-15% increase (attention on small feature maps is fast)
- **Training:** May need 10-20% more epochs for attention layers to converge
