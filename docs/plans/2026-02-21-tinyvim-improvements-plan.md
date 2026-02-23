# TinyViM Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix two confirmed bugs in the TinyViM integration (dilation drop, residual path), add hybrid ConvSENeXt+TinyViM stage configuration, and add Lovász loss to the main decode head to recover and exceed the IROS 2025 baseline mIoU on SemanticKITTI.

**Architecture:** The changes are spread across three areas: (1) `tvimblock.py` — propagate `dilation` through the TinyViM block chain and fix the residual path; (2) `harpnext_backbone.py` — add a `hybrid` block_type that selects block class per stage; (3) `trainer/manager.py` — add Lovász loss on the main head. A new `harpnext-semantickitti-hybrid.yaml` config file is also created.

**Tech Stack:** PyTorch 2.5, CUDA 11.8, Python 3.12. No formal test suite — verification is done via short eval runs using `python main.py --eval --restart`. `conda activate harpnext` before all commands.

---

## Context

The project has **no formal test suite**. All verification steps use:
```bash
python main.py --eval --restart --gpu 0 --fp16 \
  --mainconfig ./configs/main/main-config.yaml \
  --netconfig ./configs/net/<config>.yaml \
  --log_path ./logs/<run-name> \
  --path_dataset /path/to/SemanticKITTI
```
A successful run means: no crash, loss prints correctly, mIoU is printed at the end.

SemanticKITTI labels are shifted by -1 internally (0–18). Ignore index is 255. See `CLAUDE.md` for full gotcha list.

---

### Task 1: Add dilation parameter to `RepDW`

**Files:**
- Modify: `core/tinyvim_core/tvimblock.py:44-55`

**What:** `RepDW` is a reparameterized depthwise conv. It currently hard-codes `padding=1` for a 3×3 conv (dilation=1). The SemanticKITTI config uses `dilations: [3,3,3,3]`, so we need `padding = dilation` for same-spatial-size output.

**Step 1: Open `core/tinyvim_core/tvimblock.py` and read lines 44–55**

Current code:
```python
class RepDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)
        self.apply(self._init_weights)
```

**Step 2: Edit `RepDW.__init__` to accept and use `dilation`**

Replace the `__init__` signature and `self.conv` line:
```python
class RepDW(torch.nn.Module):
    def __init__(self, ed, dilation: int = 1) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, dilation, dilation=dilation, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.dilation = dilation
        self.bn = torch.nn.BatchNorm2d(ed)
        self.apply(self._init_weights)
```

Note: `Conv2d_BN(a, b, ks, stride, pad, dilation, groups)` — positional args match the signature at line 20. `pad=dilation` ensures same-size output for dilated 3×3 conv. Store `self.dilation` for reference.

**Step 3: Commit**
```bash
git add core/tinyvim_core/tvimblock.py
git commit -m "fix: add dilation support to RepDW depthwise conv"
```

---

### Task 2: Propagate dilation through `SS2D.__init__`

**Files:**
- Modify: `core/tinyvim_core/tvimblock.py:330-415`

**What:** `SS2D` creates a `local_conv = RepDW(...)` at line 369. This local conv handles the non-SSM branch of features. It must receive the dilation so its receptive field matches ConvSENeXt.

**Step 1: Read `SS2D.__init__` (lines 330–422)**

Key lines to change:
- Line 331: `def __init__(self, ..., d_conv=3, conv_bias=True, ...)` — add `dilation: int = 1`
- Line 369: `self.local_conv = RepDW(d_expand-d_inner)` — add `dilation=dilation`

**Step 2: Add `dilation` parameter to `SS2D.__init__` signature**

Find the `def __init__` line of `SS2D` (line ~331) and add `dilation: int = 1` after `conv_bias=True`:
```python
    def __init__(
        self,
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        d_conv=3,
        conv_bias=True,
        dilation: int = 1,   # ← add this
        dropout=0.0,
        ...
    ):
```

**Step 3: Pass dilation to `RepDW` at line ~369**

Change:
```python
self.local_conv = RepDW(d_expand-d_inner)
```
To:
```python
self.local_conv = RepDW(d_expand-d_inner, dilation=dilation)
```

**Step 4: Commit**
```bash
git add core/tinyvim_core/tvimblock.py
git commit -m "fix: propagate dilation into SS2D local_conv"
```

---

### Task 3: Propagate dilation through `TViMBlock.__init__`

**Files:**
- Modify: `core/tinyvim_core/tvimblock.py:547-611`

**What:** `TViMBlock` creates `self.op = SS2D(...)` at line ~578. It must pass `dilation` through.

**Step 1: Add `dilation: int = 1` to `TViMBlock.__init__` signature**

Current signature (line ~548):
```python
class TViMBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: ... = ...,
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_simple_init=False,
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        use_checkpoint: bool = False,
        index = 0,
        **kwargs,
    ):
```

Add `dilation: int = 1` before `**kwargs`:
```python
        dilation: int = 1,
        **kwargs,
```

**Step 2: Pass `dilation` to `SS2D` instantiation (line ~578)**

Change:
```python
        self.op = SS2D(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_rank_ratio=ssm_rank_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            simple_init=ssm_simple_init,
            index = index,
        )
```
To:
```python
        self.op = SS2D(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_rank_ratio=ssm_rank_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dilation=dilation,
            dropout=ssm_drop_rate,
            simple_init=ssm_simple_init,
            index=index,
        )
```

**Step 3: Commit**
```bash
git add core/tinyvim_core/tvimblock.py
git commit -m "fix: propagate dilation through TViMBlock to SS2D"
```

---

### Task 4: Pass dilation from `HARPNeXtTinyViMBlock` to `TViMBlock`

**Files:**
- Modify: `core/tinyvim_core/tvimblock.py:614-675`

**What:** `HARPNeXtTinyViMBlock` already accepts `dilation` in its `__init__` (line ~621) but never forwards it to `TViMBlock` (line ~652). This is the root bug.

**Step 1: Read `HARPNeXtTinyViMBlock.__init__` (lines 614–674)**

At line ~652, `TViMBlock` is created:
```python
        self.block = TViMBlock(
            hidden_dim=planes,
            drop_path=drop_path,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_rank_ratio=ssm_rank_ratio,
            ssm_dt_rank=ssm_dt_rank,
            ssm_act_layer=ssm_act_layer,
            ssm_conv=ssm_conv,
            ssm_conv_bias=ssm_conv_bias,
            ssm_drop_rate=ssm_drop_rate,
            ssm_simple_init=ssm_simple_init,
            mlp_ratio=mlp_ratio,
            mlp_act_layer=mlp_act_layer,
            mlp_drop_rate=mlp_drop_rate,
            use_checkpoint=use_checkpoint,
            index=index,
        )
```

**Step 2: Add `dilation=dilation` to the `TViMBlock(...)` call**

```python
        self.block = TViMBlock(
            hidden_dim=planes,
            drop_path=drop_path,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_rank_ratio=ssm_rank_ratio,
            ssm_dt_rank=ssm_dt_rank,
            ssm_act_layer=ssm_act_layer,
            ssm_conv=ssm_conv,
            ssm_conv_bias=ssm_conv_bias,
            dilation=dilation,           # ← add this line
            ssm_drop_rate=ssm_drop_rate,
            ssm_simple_init=ssm_simple_init,
            mlp_ratio=mlp_ratio,
            mlp_act_layer=mlp_act_layer,
            mlp_drop_rate=mlp_drop_rate,
            use_checkpoint=use_checkpoint,
            index=index,
        )
```

**Step 3: Commit**
```bash
git add core/tinyvim_core/tvimblock.py
git commit -m "fix: pass dilation from HARPNeXtTinyViMBlock to TViMBlock (root bug fix)"
```

---

### Task 5: Fix residual path in `HARPNeXtTinyViMBlock.forward`

**Files:**
- Modify: `core/tinyvim_core/tvimblock.py:671-675`

**What:** The current `forward` mutates `x = self.downsample(x)` before passing to the block. This means TViMBlock's internal skip connection uses the already-projected tensor as the residual, which works but differs from ConvSENeXt semantics. In ConvSENeXt, the main transform starts from the original `x` and the skip is projected separately. We replicate this by saving the original `x` before downsampling and passing it separately.

**Step 1: Read `HARPNeXtTinyViMBlock.forward` (lines 671–675)**

Current:
```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample is not None:
            x = self.downsample(x)
        return self.block(x)
```

**Step 2: Update `TViMBlock._forward` to accept an optional `residual` argument (lines 600–605)**

Current `_forward`:
```python
    def _forward(self, input: torch.Tensor):
        if self.ssm_branch:
            x = input + self.drop_path(self.op(input))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(x))
        return x
```

Change to:
```python
    def _forward(self, input: torch.Tensor, residual: Optional[torch.Tensor] = None):
        skip = residual if residual is not None else input
        if self.ssm_branch:
            x = skip + self.drop_path(self.op(input))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(x))
        return x
```

**Step 3: Update `TViMBlock.forward` to accept and pass `residual` (lines 607–611)**

Change:
```python
    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)
```
To:
```python
    def forward(self, input: torch.Tensor, residual: Optional[torch.Tensor] = None):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input, residual)
        else:
            return self._forward(input, residual)
```

**Step 4: Update `HARPNeXtTinyViMBlock.forward` to save original x and pass projected residual**

Change:
```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample is not None:
            x = self.downsample(x)
        return self.block(x)
```
To:
```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.block(x, residual=residual)
```

Note: When `stride=1` and `inplanes==planes` (no downsample), `residual = x` = `input`, so behavior is identical to before for the non-strided case. Only strided stages (1, 2, 3) are affected.

**Step 5: Commit**
```bash
git add core/tinyvim_core/tvimblock.py
git commit -m "fix: correct residual path in TViMBlock to match ConvSENeXt semantics"
```

---

### Task 6: Verify the bug fixes don't break the model

**Files:** None (eval run only)

**Step 1: Run a quick eval with the fixed TinyViM config**

```bash
conda activate harpnext
python main.py --eval --restart --gpu 0 --fp16 \
  --mainconfig ./configs/main/main-config.yaml \
  --netconfig ./configs/net/harpnext-semantickitti-tinyvim.yaml \
  --log_path ./logs/<your-tinyvim-run> \
  --path_dataset /path/to/SemanticKITTI
```

Expected: Eval completes without error. mIoU is printed. Check that the value is >= the broken baseline (or better).

If the run crashes with a shape mismatch error:
- The most likely cause is a wrong padding value in Task 1. `padding` for a dilated depthwise conv of kernel=3 must equal `dilation`. Verify the `Conv2d_BN` call: `Conv2d_BN(ed, ed, 3, 1, dilation, dilation=dilation, groups=ed)` — 5th positional arg is `pad`, 6th is `dilation`.

---

### Task 7: Add hybrid block_type support to `HARPNeXtBackbone`

**Files:**
- Modify: `core/harpnext_core/backbone/harpnext_backbone.py:148-233` (init)
- Modify: `core/harpnext_core/backbone/harpnext_backbone.py:293-348` (_make_res_layer)

**What:** Add a `"hybrid"` value for `block_type` that reads a `stage_block_types` list from config and selects `ConvSENeXt` or `HARPNeXtTinyViMBlock` per stage. `["convsenext", "convsenext", "tinyvim", "tinyvim"]` uses ConvSENeXt for early high-resolution stages and TinyViM for later low-resolution stages.

**Step 1: Update `HARPNeXtBackbone.__init__` to accept `stage_block_types`**

Find the `def __init__` signature (line ~153) and add `stage_block_types` parameter after `block_cfg`:
```python
    def __init__(self,
                 in_channels: int = 16,
                 point_in_channels: int = 384,
                 output_shape: Sequence[int] = [],
                 depth: int = 34,
                 stem_channels: int = 128,
                 num_stages: int = 4,
                 out_channels: Sequence[int] = (128, 128, 128, 128),
                 strides: Sequence[int] = (1, 2, 2, 2),
                 dilations: Sequence[int] = (1, 1, 1, 1),
                 fuse_channels: Sequence[int] = (256, 128),
                 dw_conv_kernel = 7,
                 dw_conv_bias = True,
                 inter_align_corners = True,
                 block_type: str = "convsenext",
                 block_cfg: Optional[dict] = None,
                 stage_block_types: Optional[list] = None) -> None:   # ← add this
```

**Step 2: Handle `"hybrid"` block_type in `__init__` body (after line ~177)**

Currently:
```python
        self.block, stage_blocks = self.arch_settings[depth]
        self.block_type = block_type.lower()
        self.block_cfg = block_cfg or {}
        if self.block_type in ("tinyvim", "tvim"):
            from core.tinyvim_core.tvimblock import HARPNeXtTinyViMBlock
            self.block = HARPNeXtTinyViMBlock
        elif self.block_type not in ("convsenext", "convsennext", "convse"):
            raise KeyError(f"invalid block_type {block_type} for HARPNeXtBackbone.")
```

Replace with:
```python
        self.block, stage_blocks = self.arch_settings[depth]
        self.block_type = block_type.lower()
        self.block_cfg = block_cfg or {}
        if self.block_type in ("tinyvim", "tvim"):
            from core.tinyvim_core.tvimblock import HARPNeXtTinyViMBlock
            self.block = HARPNeXtTinyViMBlock
        elif self.block_type == "hybrid":
            from core.tinyvim_core.tvimblock import HARPNeXtTinyViMBlock
            self._hybrid_tinyvim_block = HARPNeXtTinyViMBlock
            self._hybrid_convsenext_block = ConvSENeXt
            assert stage_block_types is not None and len(stage_block_types) == num_stages, \
                f"block_type='hybrid' requires stage_block_types list of length {num_stages}"
            self.stage_block_types = [s.lower() for s in stage_block_types]
        elif self.block_type not in ("convsenext", "convsennext", "convse"):
            raise KeyError(f"invalid block_type {block_type} for HARPNeXtBackbone.")
```

**Step 3: Update `_make_res_layer` to accept per-stage block type override**

Change the signature (line ~293):
```python
    def _make_res_layer(self, block: nn.Module, inplanes, planes, num_blocks, stride, dilation, dw_conv_kernel, dw_conv_bias, index: int = 0):
```
To:
```python
    def _make_res_layer(self, block: nn.Module, inplanes, planes, num_blocks, stride, dilation, dw_conv_kernel, dw_conv_bias, index: int = 0, stage_block_type: str = None):
```

Add at the top of `_make_res_layer` body:
```python
        # Resolve block class for this stage (hybrid mode overrides self.block_type)
        if stage_block_type is not None:
            effective_block_type = stage_block_type
            if effective_block_type in ("tinyvim", "tvim"):
                block = self._hybrid_tinyvim_block
            else:
                block = self._hybrid_convsenext_block
        else:
            effective_block_type = self.block_type
```

Then replace every `if self.block_type in ("tinyvim", "tvim"):` check inside `_make_res_layer` with `if effective_block_type in ("tinyvim", "tvim"):`.

**Step 4: Pass `stage_block_type` in the stage construction loop**

In the loop at line ~207:
```python
        for i, num_blocks in enumerate(stage_blocks):
```

Change the `_make_res_layer` call to pass the stage override:
```python
            res_layer = self._make_res_layer(
                block=self.block,
                inplanes=inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                dw_conv_kernel=dw_conv_kernel,
                dw_conv_bias=dw_conv_bias,
                index=i,
                stage_block_type=self.stage_block_types[i] if self.block_type == "hybrid" else None,
            )
```

**Step 5: Commit**
```bash
git add core/harpnext_core/backbone/harpnext_backbone.py
git commit -m "feat: add hybrid block_type to backbone for per-stage ConvSENeXt/TinyViM selection"
```

---

### Task 8: Create the hybrid config file

**Files:**
- Create: `configs/net/harpnext-semantickitti-hybrid.yaml`

**What:** A new config file that uses `block_type: hybrid` with ConvSENeXt for stages 0–1 and TinyViM for stages 2–3.

**Step 1: Create the file**

Copy `configs/net/harpnext-semantickitti-tinyvim.yaml` and change only the backbone block config:

```yaml
# Copyright 2025 CEA LIST - Samir Abou Haidar
# Licensed under the Apache License, Version 2.0

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
    stage_block_types: [convsenext, convsenext, tinyvim, tinyvim]
    block_cfg:
      ssm_d_state: 16
      ssm_ratio: 2.0
      ssm_rank_ratio: 2.0
      ssm_dt_rank: auto
      ssm_conv: 3
      ssm_conv_bias: True
      ssm_drop_rate: 0.0
      ssm_simple_init: False
      mlp_ratio: 4.0
      mlp_drop_rate: 0.0
      use_checkpoint: False
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
  lovasz_main_weight: 1.0

preproc:
  gpu: False
```

**Step 2: Verify the hybrid config parses correctly by running eval**

```bash
python main.py --eval --restart --gpu 0 --fp16 \
  --mainconfig ./configs/main/main-config.yaml \
  --netconfig ./configs/net/harpnext-semantickitti-hybrid.yaml \
  --log_path ./logs/<any-existing-run-with-matching-arch> \
  --path_dataset /path/to/SemanticKITTI
```

Expected: No `KeyError` or `AssertionError`. If it fails with `KeyError: invalid block_type hybrid`, check Task 7 Step 2 is correctly applied.

**Step 3: Commit**
```bash
git add configs/net/harpnext-semantickitti-hybrid.yaml
git commit -m "feat: add SemanticKITTI hybrid ConvSENeXt+TinyViM config"
```

---

### Task 9: Add Lovász loss to main decode head in trainer

**Files:**
- Modify: `trainer/manager.py:366-373`
- Modify: `configs/net/harpnext-semantickitti-tinyvim.yaml`
- Modify: `configs/net/harpnext-semantickitti.yaml`

**What:** Currently `loss_points` (main head loss) is only cross-entropy. Adding Lovász here directly optimizes IoU for the final prediction layer. The weight is controlled by `train.lovasz_main_weight` in the netconfig (defaults to 0 if key is absent = no change for existing configs).

**Step 1: Read the loss computation block in `trainer/manager.py` (lines 366–373)**

```python
                if training:
                    lamda = self.netconfig["train"]["lamda"]
                    loss_points = self.loss["ce"](out_losses["HARPNeXtHead.seg_logit"], labels["pt_labels"])
                    loss_aux_0 = ...
                    ...
                    loss = loss_points + lamda*loss_aux_0 + lamda*loss_aux_1 + lamda*loss_aux_2 + lamda*loss_aux_3
```

**Step 2: Update the training loss block to include Lovász on main head**

Change lines 366–373:
```python
                if training:
                    lamda = self.netconfig["train"]["lamda"]
                    lovasz_main_weight = self.netconfig["train"].get("lovasz_main_weight", 0.0)
                    loss_points = self.loss["ce"](out_losses["HARPNeXtHead.seg_logit"], labels["pt_labels"])
                    if lovasz_main_weight > 0:
                        loss_points = loss_points + lovasz_main_weight * self.loss["lovasz"](out_losses["HARPNeXtHead.seg_logit"], labels["pt_labels"])
                    loss_aux_0 = self.loss["ce"](out_losses["AuxHead_0.seg_logit"], labels["proj_labels"]) + 1.5 * self.loss["lovasz"](out_losses["AuxHead_0.seg_logit"], labels["proj_labels"]) + self.loss["bd"](out_losses["AuxHead_0.seg_logit"], labels["proj_labels"])
                    loss_aux_1 = self.loss["ce"](out_losses["AuxHead_1.seg_logit"], labels["proj_labels"]) + 1.5 * self.loss["lovasz"](out_losses["AuxHead_1.seg_logit"], labels["proj_labels"]) + self.loss["bd"](out_losses["AuxHead_1.seg_logit"], labels["proj_labels"])
                    loss_aux_2 = self.loss["ce"](out_losses["AuxHead_2.seg_logit"], labels["proj_labels"]) + 1.5 * self.loss["lovasz"](out_losses["AuxHead_2.seg_logit"], labels["proj_labels"]) + self.loss["bd"](out_losses["AuxHead_2.seg_logit"], labels["proj_labels"])
                    loss_aux_3 = self.loss["ce"](out_losses["AuxHead_3.seg_logit"], labels["proj_labels"]) + 1.5 * self.loss["lovasz"](out_losses["AuxHead_3.seg_logit"], labels["proj_labels"]) + self.loss["bd"](out_losses["AuxHead_3.seg_logit"], labels["proj_labels"])
                    loss = loss_points + lamda*loss_aux_0 + lamda*loss_aux_1 + lamda*loss_aux_2 + lamda*loss_aux_3
```

Key: using `.get("lovasz_main_weight", 0.0)` means existing configs without this key default to 0 (no change in behavior = backward compatible).

**Step 3: Add `lovasz_main_weight` to the TinyViM config**

In `configs/net/harpnext-semantickitti-tinyvim.yaml`, under `train:`:
```yaml
train:
  lamda: 1.0
  lovasz_main_weight: 1.0
```

**Step 4: Add `lovasz_main_weight` to the base SemanticKITTI config (optional)**

In `configs/net/harpnext-semantickitti.yaml`, under `train:`:
```yaml
train:
  lamda: 1.0
  lovasz_main_weight: 1.0
```

This also enables the improvement for the original ConvSENeXt model.

**Step 5: Commit**
```bash
git add trainer/manager.py configs/net/harpnext-semantickitti-tinyvim.yaml configs/net/harpnext-semantickitti.yaml
git commit -m "feat: add Lovász loss on main decode head (lovasz_main_weight config key)"
```

---

### Task 10: Final end-to-end verification

**Files:** None (verification runs only)

**Step 1: Verify all three configs parse and eval without errors**

Run each config (replace `<run-name>` with an actual trained checkpoint directory):

```bash
# 1. Pure TinyViM with all fixes
python main.py --eval --restart --gpu 0 --fp16 \
  --mainconfig ./configs/main/main-config.yaml \
  --netconfig ./configs/net/harpnext-semantickitti-tinyvim.yaml \
  --log_path ./logs/<tinyvim-run> \
  --path_dataset /path/to/SemanticKITTI

# 2. Hybrid
python main.py --eval --restart --gpu 0 --fp16 \
  --mainconfig ./configs/main/main-config.yaml \
  --netconfig ./configs/net/harpnext-semantickitti-hybrid.yaml \
  --log_path ./logs/<hybrid-run> \
  --path_dataset /path/to/SemanticKITTI

# 3. Original ConvSENeXt with Lovász on main head
python main.py --eval --restart --gpu 0 --fp16 \
  --mainconfig ./configs/main/main-config.yaml \
  --netconfig ./configs/net/harpnext-semantickitti.yaml \
  --log_path ./logs/<convsenext-run> \
  --path_dataset /path/to/SemanticKITTI
```

Expected: All three complete without error. mIoU values are printed.

**Step 2: Record mIoU numbers for comparison**

Note the mIoU values:
- Broken TinyViM baseline (before fixes): < baseline
- Fixed TinyViM: should be >= broken baseline, ideally >= ConvSENeXt baseline
- Hybrid: may be slightly different from pure TinyViM
- ConvSENeXt + Lovász: should be >= pure ConvSENeXt baseline

**Step 3: Start full training run with best config (choose based on eval results)**

```bash
python main.py \
  --net harpnext --dataset semantic_kitti \
  --path_dataset /path/to/SemanticKITTI \
  --mainconfig ./configs/main/main-config.yaml \
  --netconfig ./configs/net/harpnext-semantickitti-hybrid.yaml \
  --log_path ./logs/hybrid-v1 \
  --gpu 0 --seed 0 --fp16
```

---

## Summary of Changes

| Task | Files Changed | Impact |
|------|--------------|--------|
| 1–4 | `tvimblock.py` | Fix dilation drop (root bug) |
| 5 | `tvimblock.py` | Fix residual path for strided stages |
| 6 | — | Verify fixes |
| 7 | `harpnext_backbone.py` | Add hybrid block_type |
| 8 | `harpnext-semantickitti-hybrid.yaml` (new) | Hybrid config |
| 9 | `manager.py`, 2 YAML configs | Lovász on main head |
| 10 | — | End-to-end verification |
