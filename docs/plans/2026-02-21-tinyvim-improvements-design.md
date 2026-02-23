# HARP-NeXt TinyViM Improvement Design

**Date:** 2026-02-21
**Target:** SemanticKITTI (primary), then nuScenes
**Goal:** Fix TinyViM performance regression and exceed IROS 2025 baseline mIoU

---

## Problem Statement

After integrating Vision Mamba (TinyViM) into the HARP-NeXt backbone, a moderate mIoU drop of 1–3 points was observed on SemanticKITTI with otherwise stable training. Root cause analysis identified two confirmed bugs and one architectural opportunity.

---

## Root Causes Identified

### Bug 1 — Dilation silently dropped (High Impact)

**Location:** `core/tinyvim_core/tvimblock.py`, `HARPNeXtTinyViMBlock.__init__`

`HARPNeXtTinyViMBlock` accepts a `dilation` parameter but never passes it to `TViMBlock` or its inner `RepDW` layers. The SemanticKITTI config uses `dilations: [3, 3, 3, 3]`, meaning ConvSENeXt operates with an effective receptive field of 19 pixels (kernel 7 + dilation 3) while TinyViM is actually running at dilation 1 (effective receptive field of 7 pixels). This is a significant loss of spatial context on the 64×512 range image.

**Fix:** Propagate `dilation` through `HARPNeXtTinyViMBlock → TViMBlock → RepDW`.

### Bug 2 — Incorrect residual path on strided stages (Moderate Impact)

**Location:** `core/tinyvim_core/tvimblock.py`, `HARPNeXtTinyViMBlock.forward`

Current code mutates `x = self.downsample(x)` before passing it to the inner block. This means `TViMBlock`'s internal skip connection uses the already-projected (downsampled) tensor as both input and residual. ConvSENeXt correctly saves `residual = x`, applies transforms, then sets `residual = self.downsample(x)` for the skip path. Affects stages 1–3 where `stride != 1`.

**Fix:** Save original `x`, pass projected residual explicitly to `TViMBlock.forward`.

---

## Design

### Section 1: Bug Fixes

#### 1a. Dilation support in `RepDW`

`RepDW.__init__` gains a `dilation: int = 1` parameter. Its depthwise `nn.Conv2d` calls are updated to use `dilation=dilation` and `padding=dilation * (kernel_size // 2)` (same-padding formula for dilated convs).

#### 1b. Dilation propagation up the call chain

- `TViMBlock.__init__` accepts `dilation: int = 1` and passes it to each `RepDW` it instantiates.
- `SS2D.__init__` accepts `dilation: int = 1` and passes it to `TViMBlock` or its inner `Rep_Inception` module as appropriate.
- `HARPNeXtTinyViMBlock.__init__` passes its `dilation` argument to `TViMBlock`.

#### 1c. Residual path fix

`TViMBlock.forward` gains an optional `residual: Optional[torch.Tensor] = None` parameter. When provided, it replaces `x` in the skip-connection additions:
```python
# Before:
x = x + drop_path(...) + drop_path(...)

# After:
skip = residual if residual is not None else x
x = skip + drop_path(...) + drop_path(...)
```

`HARPNeXtTinyViMBlock.forward` saves original `x` before downsampling:
```python
def forward(self, x):
    residual = x
    if self.downsample is not None:
        residual = self.downsample(x)
    return self.block(x, residual=residual)
```

---

### Section 2: Hybrid Stage Placement

**Motivation:** ConvSENeXt excels at local feature extraction (edges, surfaces) at full/high resolution. TinyViM/SS2D excels at capturing global context via sequential scanning. Stages 0–1 operate at high spatial resolution (64×512 and 32×256) where local features are most important. Stages 2–3 operate at lower resolution (16×128 and 8×64) where global context is more valuable and SS2D's scanning cost is lower.

**New `block_type` value:** `hybrid`

**Backbone config change:**

```yaml
backbone:
  block_type: hybrid
  stage_block_types: [convsenext, convsenext, tinyvim, tinyvim]
  block_cfg:        # only used when a stage uses tinyvim
    ssm_d_state: 16
    ssm_ratio: 2.0
    ...
```

**Implementation in `HARPNeXtBackbone`:**

- `arch_settings` is unchanged (`depth=10` → `stage_blocks = (1,1,1,1)`).
- New attribute `self.stage_block_types: List[str]` parsed from config when `block_type == "hybrid"`.
- `_make_res_layer` receives `stage_block_type: str` and selects the appropriate block class.
- When `block_type` is `"convsenext"` or `"tinyvim"`, all stages use that single type (backward compatible).
- When `block_type` is `"hybrid"`, each stage uses its corresponding entry from `stage_block_types`.

**New config file:** `configs/net/harpnext-semantickitti-hybrid.yaml` — copy of `*-tinyvim.yaml` with `block_type: hybrid` and `stage_block_types`.

---

### Section 3: Lovász Loss on Main Head

**Motivation:** The main decode head currently uses only cross-entropy, which optimizes per-point accuracy rather than IoU. Auxiliary heads already use `1.5× Lovász + CE + Boundary`. Adding Lovász to the main head directly optimizes the mIoU metric we are measuring.

**Loss formula change** (`trainer/manager.py`):

```
# Before:
loss = CE(main) + λ × Σ(CE + 1.5×Lov + BD)(aux_i)

# After:
loss = CE(main) + α×Lovász(main) + λ × Σ(CE + 1.5×Lov + BD)(aux_i)
```

**New config key:** `train.lovasz_main_weight` (float, default `1.0`) in the netconfig files.

No new imports or dependencies; `lovasz_softmax()` is already used in `manager.py`.

---

## Files Changed

| File | Change |
|------|--------|
| `core/tinyvim_core/tvimblock.py` | Fix dilation in `RepDW`, `TViMBlock`, `SS2D`, `HARPNeXtTinyViMBlock`; fix residual path in `HARPNeXtTinyViMBlock.forward` and `TViMBlock.forward` |
| `core/harpnext_core/backbone/harpnext_backbone.py` | Add `hybrid` block_type support in `__init__` and `_make_res_layer` |
| `configs/net/harpnext-semantickitti-tinyvim.yaml` | Add `lovasz_main_weight: 1.0` under `train:` |
| `configs/net/harpnext-semantickitti-hybrid.yaml` | New file: hybrid stage config |
| `configs/net/harpnext-nuscenes-tinyvim.yaml` | Add `lovasz_main_weight: 1.0` under `train:` (follow-up) |
| `trainer/manager.py` | Read `lovasz_main_weight`, add Lovász to main head loss |

---

## Verification Plan

1. **Sanity check** — run 5 epochs with `--fp16` on SemanticKITTI and confirm loss decreases normally (no NaN, no divergence).
2. **Short eval** — run full eval (`--eval --restart`) with the fixed TinyViM config and compare mIoU vs the broken baseline.
3. **Hybrid ablation** — run full eval with the hybrid config and compare vs pure TinyViM.
4. **Full training run** — 100 epochs with all improvements to get final mIoU.

---

## Expected Outcomes

- Bug Fix 1 (dilation) alone should recover most of the 1–3 mIoU gap.
- Bug Fix 2 (residual) provides correct gradient flow on downsampling stages.
- Hybrid placement may provide an additional 0.5–1 mIoU by using the right inductive biases at each scale.
- Lovász on main head historically provides +0.5–1 mIoU on segmentation benchmarks.

Total expected improvement: **+2–5 mIoU** over the broken TinyViM baseline, potentially exceeding the original IROS 2025 ConvSENeXt baseline.
