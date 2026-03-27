Below is a coding-agent implementation spec for the three methods we converged on:

1. **Core method:** late-stage **RangeMamba-SE** replacement inside HARP-NeXt
2. **Optional training method:** **soft pixel supervision** for ambiguous projected pixels
3. **Optional model extension:** **uncertainty-guided sparse 3D refinement** on hard points

The goal is to preserve the HARP-NeXt contract: same feature encoder, same Pt2Px / Px2Pt mapping operators, same 4-stage backbone layout, same multi-scale fusion equations, and same fusion head interface. HARP-NeXt uses range images of **32×480** on nuScenes and **64×512** on SemanticKITTI, a **4-stage** backbone with strides **(1,2,2,2)**, and outputs **128 channels** at both pixel and point levels; its ablation also shows **4 stages with 1 block per stage** is the best speed/accuracy setting, so do not deepen the network for the first implementation.    

## 0. Non-negotiable baseline contract

Implement these rules first:

* Keep the **feature encoder** unchanged.
* Keep **Pt2Px** and **Px2Pt** unchanged.
* Keep the stage-level fusion equations unchanged:

  * pixel fusion uses (\tilde P_x^n), (\bar P_x^{n-1}), interpolation, attention, and residual update
  * point refinement uses **(M^{-1}(\tilde P_x^n))** concatenated with the previous point features
* Keep the final fusion head unchanged in the first paper version.
* Replace only the **pixel feature extractor block** in **stage 3** and **stage 4**. Keep **stage 1–2** as original Conv-SE-NeXt.   

Critical pitfall: **do not feed (P_x^n) into the point branch.** The baseline uses **(\tilde P_x^n)** for Px2Pt in Eq. (14). Changing that breaks the ablation fairness and alters the model’s fusion semantics. 

## 1. Dependencies, frameworks, and tools

Use this stack:

* **Python + PyTorch**
* **Official Mamba package** from `mamba-ssm`
* Optional: `causal-conv1d` for faster kernels
* Existing baseline CUDA / scatter mapping ops for Pt2Px and Px2Pt
* Optional for sparse refiner: `torch-cluster` or `torch-geometric` for radius graph / kNN
* Profiling: `torch.cuda.Event`, `torch.profiler`, and optionally Nsight Systems

The current official Mamba docs say to install **PyTorch first**, then optionally `causal-conv1d>=1.4.0`, then `mamba-ssm` or `mamba-ssm[causal-conv1d]` with `--no-build-isolation`; the official package also notes Linux + NVIDIA GPU support and lists **PyTorch 1.12+** and **CUDA 11.6+** as prerequisites. ([PyPI][1])

Implementation rule: wrap the external Mamba class behind your own adapter, e.g. `SelectiveScan1D`, so the rest of the code does not depend on whether the backend is `Mamba` or `Mamba2`. The official package exposes both a Mamba block and a Mamba-2 block. ([PyPI][1])

---

# IDEA 1 — Core model: RangeMamba-SE in late stages

## 1.1 What to implement

Replace the Conv-SE-NeXt block in:

* **Stage 3** with **row-wise bidirectional Mamba**
* **Stage 4** with **row-wise + column-wise bidirectional Mamba**

Keep:

* **Stage 1–2** as Conv-SE-NeXt
* the baseline input resolutions
* the same 128-channel stage outputs
* the same multi-scale range-point fusion equations
* the same training schedule initially.   

## 1.2 File/module breakdown

Suggested files:

* `models/range_mamba/geometry_context.py`
* `models/range_mamba/axial_mamba.py`
* `models/range_mamba/depth_gate.py`
* `models/range_mamba/range_mamba_se.py`
* `models/backbone/stage_wrapper.py`
* `configs/harp_mamba.yaml`

### Module A — `GeometryContextBuilder`

**Input**

* `depth`: `[B,1,H,W]`
* `intensity`: `[B,1,H,W]`
* `valid`: `[B,1,H,W]`

**Output**

* `geom`: `[B,C,H,W]`
* `gradmag`: `[B,1,H,W]`

**Functionality**

* Downsample raw depth/intensity/mask to stage resolution
* Compute masked depth gradients
* Produce a geometry embedding for gating and Mamba conditioning

**Implementation steps**

1. Downsample `valid` with **max pooling** or nearest; output must indicate whether any original point exists in the coarse cell.
2. Downsample `depth` and `intensity` with **valid-weighted average**, not plain average.
3. Compute `dx`, `dy` using finite differences on masked depth.
4. Build `geom_in = concat(depth, intensity, valid, abs(dx), abs(dy))`.
5. Apply `1x1 Conv -> BN or LN-free linear conv -> activation` to get `geom` with `C=128`.

**Pseudocode**

```python
def build_geometry_context(depth0, intensity0, valid0, out_hw, out_channels):
    valid = max_pool_or_nearest(valid0, out_hw)                  # [B,1,H,W]
    depth_num = avg_pool_to(depth0 * valid0, out_hw)
    depth_den = avg_pool_to(valid0, out_hw).clamp_min(1e-6)
    depth = depth_num / depth_den

    inten_num = avg_pool_to(intensity0 * valid0, out_hw)
    inten_den = avg_pool_to(valid0, out_hw).clamp_min(1e-6)
    intensity = inten_num / inten_den

    dx = finite_diff_x(depth) * valid
    dy = finite_diff_y(depth) * valid
    gradmag = (dx.abs() + dy.abs()) * valid

    geom_in = torch.cat([depth, intensity, valid, dx.abs(), dy.abs()], dim=1)
    geom = conv1x1(geom_in, out_channels)
    return geom, gradmag, valid
```

**Pitfalls**

* Do **not** use plain average pooling on depth/intensity; invalid zeros will bias the value.
* Do **not** compute gradients before masking.
* Keep depth/intensity normalization consistent with baseline preprocessing.

---

### Module B — `CircularBiMamba1D`

**Input**

* `x`: `[B,C,H,W]`
* `geom`: `[B,C,H,W]`
* `valid`: `[B,1,H,W]`
* `axis`: `"row"` or `"col"`

**Output**

* `z`: `[B,C,H,W]`

**Functionality**

* Convert each row or column into a sequence
* Apply bidirectional Mamba
* For rows, use circular handling along width

**Implementation steps**

1. Add `geom` to `x`, then apply **LayerNorm over channels**.
2. If `axis == "row"`, reshape to `[(B*H), W, C]`.
3. If `axis == "col"`, reshape to `[(B*W), H, C]`.
4. Multiply tokens by the valid mask before scan.
5. For row scan only, apply `torch.roll(seq, shifts=W//2, dims=1)` before Mamba and reverse the roll after. This removes a fixed seam at column 0.
6. Use **two independent** Mamba blocks:

   * forward sequence
   * reversed sequence
7. Concatenate forward and backward outputs, project back to `C`.
8. Zero invalid outputs after scan.

**Pseudocode**

```python
def circular_bimamba_1d(x, geom, valid, axis, mamba_f, mamba_b, proj):
    s = layer_norm_channels_last(x + geom)   # still [B,C,H,W]

    if axis == "row":
        seq  = rearrange(s,    "b c h w -> (b h) w c")
        mask = rearrange(valid, "b 1 h w -> (b h) w 1")
        L = seq.shape[1]
        seq = seq * mask
        seq = torch.roll(seq, shifts=L // 2, dims=1)

        zf = mamba_f(seq)
        zb = torch.flip(mamba_b(torch.flip(seq, dims=[1])), dims=[1])
        z  = proj(torch.cat([zf, zb], dim=-1))

        z  = torch.roll(z, shifts=-(L // 2), dims=1)
        z  = z * mask
        z  = rearrange(z, "(b h) w c -> b c h w", b=x.shape[0], h=x.shape[2])

    else:  # axis == "col"
        seq  = rearrange(s,    "b c h w -> (b w) h c")
        mask = rearrange(valid, "b 1 h w -> (b w) h 1")
        seq = seq * mask

        zf = mamba_f(seq)
        zb = torch.flip(mamba_b(torch.flip(seq, dims=[1])), dims=[1])
        z  = proj(torch.cat([zf, zb], dim=-1))

        z  = z * mask
        z  = rearrange(z, "(b w) h c -> b c h w", b=x.shape[0], w=x.shape[3])

    return z
```

**Pitfalls**

* Do **not** flatten the whole 2D map into one sequence.
* Do **not** use BatchNorm inside the token sequence path; use LayerNorm before Mamba.
* Always zero invalid tokens before and after scan.
* Use a projection after concatenating forward/backward outputs.

---

### Module C — `DepthAwareGate`

**Input**

* `u_local`: `[B,C,H,W]`
* `z_ctx`: `[B,C,H,W]`
* `valid`: `[B,1,H,W]`
* `gradmag`: `[B,1,H,W]`

**Output**

* `y`: `[B,C,H,W]`

**Functionality**

* Control how much global context is injected
* Reduce oversmoothing across depth discontinuities

**Implementation steps**

1. Concatenate `[u_local, z_ctx, valid, gradmag]`
2. Apply `1x1 Conv -> sigmoid`
3. Fuse with `y = u_local + gate * z_ctx`

**Pseudocode**

```python
def depth_aware_gate(u_local, z_ctx, valid, gradmag, gate_conv):
    gate_in = torch.cat([u_local, z_ctx, valid, gradmag], dim=1)
    gate = torch.sigmoid(gate_conv(gate_in))
    return u_local + gate * z_ctx
```

**Pitfalls**

* Initialize the last gate bias negative, e.g. `-2.0`, so the model starts conservative.
* Compute `gradmag` at the **same resolution** as the current stage.

---

### Module D — `SELite`

**Input**

* `y`: `[B,C,H,W]`

**Output**

* `y_se`: `[B,C,H,W]`

**Functionality**

* Same SE concept as baseline, but lightweight

**Implementation steps**

1. Global average pool to `[B,C,1,1]`
2. `1x1 Conv -> ReLU -> 1x1 Conv -> Hardsigmoid`
3. Multiply weights back into `y`

**Pseudocode**

```python
def se_lite(y, conv1, conv2):
    w = F.adaptive_avg_pool2d(y, 1)
    w = F.relu(conv1(w), inplace=True)
    w = F.hardsigmoid(conv2(w))
    return y * w
```

**Pitfalls**

* Keep channel count unchanged.
* Use conv layers, not dense layers, to match the lightweight baseline style. The baseline Conv-SE-NeXt uses depthwise separable conv and an SE-style recalibration with 1×1 convolutions for efficiency.  

---

### Module E — `RangeMambaSEBlock`

**Input**

* `x`: `[B,C,H,W]`
* `depth`, `intensity`, `valid`: `[B,1,H,W]`
* `use_col_mamba`: bool

**Output**

* `tilde_px_n`: `[B,C,H,W]`

**Functionality**

* Drop-in replacement for the pixel block in stage 3 or 4
* Preserve spatial size and channel count

**Implementation steps**

1. Local stem: `DWConv -> BN -> Hardswish -> PWConv -> BN`
2. Build geometry context
3. Apply row-wise `CircularBiMamba1D`
4. If `use_col_mamba=True`, apply column Mamba on the row output
5. Fuse with `DepthAwareGate`
6. Apply `SELite`
7. Add residual skip from input

**Pseudocode**

```python
class RangeMambaSEBlock(nn.Module):
    def __init__(self, c, use_col_mamba=False):
        ...
    def forward(self, x, depth, intensity, valid):
        # 1) local conv stem
        u = self.bn1(self.dwconv(x))
        u = F.hardswish(u)
        u = self.bn2(self.pwconv(u))

        # 2) geometry
        geom, gradmag, valid_s = build_geometry_context(
            depth, intensity, valid, out_hw=u.shape[-2:], out_channels=u.shape[1]
        )

        # 3) row Mamba
        z = circular_bimamba_1d(
            u, geom, valid_s, axis="row",
            mamba_f=self.row_f, mamba_b=self.row_b, proj=self.row_proj
        )

        # 4) optional column Mamba
        if self.use_col_mamba:
            z_col = circular_bimamba_1d(
                z, geom, valid_s, axis="col",
                mamba_f=self.col_f, mamba_b=self.col_b, proj=self.col_proj
            )
            z = self.mix(torch.cat([z, z_col], dim=1))

        # 5) gated fusion
        y = depth_aware_gate(u, z, valid_s, gradmag, self.gate_conv)

        # 6) SE recalibration
        y = se_lite(y, self.se1, self.se2)

        # 7) residual
        return y + x
```

**Default config**

* `C = 128`
* `stage3.use_col_mamba = False`
* `stage4.use_col_mamba = True`
* start with `d_state = 16` or `32`
* start with `expand = 2`

**Pitfalls**

* Output shape must exactly match the replaced Conv-SE-NeXt block.
* Do not insert extra downsampling inside this block.
* Do not change point branch dimensions.

---

### Module F — `StageWrapper` integration

**Input**

* `px_prev`: pixel features for stage (n)
* `pt_prev`: point features for stage (n)
* `bar_px_prev`: mapped pixel features from previous point stage
* `stage_aux`: stage-resolution depth/intensity/valid
* `proj_indices_stage_n`: point-to-pixel mapping indices for this stage

**Output**

* `px_n`
* `pt_n`
* `bar_px_n`

**Functionality**

* Preserve the exact HARP-NeXt fusion contract
* Only change the pixel feature extractor inside the stage

**Implementation steps**

1. `tilde_px_n = pixel_block_n(px_prev, aux)`
2. `fuse_px_n = Conv(cat(tilde_px_n, upsample(bar_px_prev)))`
3. `attn_n = sigmoid(attn_head(fuse_px_n))`
4. `px_n = tilde_px_n + attn_n * fuse_px_n`
5. `px2pt_n = Px2Pt(tilde_px_n, proj_indices_stage_n)`
6. `pt_n = point_refine(cat(px2pt_n, pt_prev))`
7. `bar_px_n = Pt2Px(pt_n, proj_indices_stage_n)`

**Pseudocode**

```python
def stage_forward(px_prev, pt_prev, bar_px_prev, aux, proj_idx, pixel_block,
                  fuse_conv, attn_head, point_refine, px2pt, pt2px):
    tilde_px = pixel_block(px_prev, aux["depth"], aux["intensity"], aux["valid"])

    fuse_px = fuse_conv(torch.cat([
        tilde_px,
        F.interpolate(bar_px_prev, size=tilde_px.shape[-2:], mode="bilinear", align_corners=False)
    ], dim=1))

    attn = torch.sigmoid(attn_head(fuse_px))
    px_n = tilde_px + attn * fuse_px

    pt_from_px = px2pt(tilde_px, proj_idx)      # IMPORTANT: use tilde_px, not px_n
    pt_n = point_refine(torch.cat([pt_from_px, pt_prev], dim=-1))
    bar_px_n = pt2px(pt_n, proj_idx)

    return px_n, pt_n, bar_px_n
```

**Pitfalls**

* `bar_px_prev` must be the mapped point feature map from the previous stage, not the previous pixel feature map.
* Keep the point refiner identical to baseline in the first version.
* Precompute or reuse stage-specific projection indices; do not recompute projection by floating-point rounding at every stage.

## 1.3 Backbone-level implementation order

Implement in this order:

1. Add config flags:

   * `use_range_mamba_stage3 = True`
   * `use_range_mamba_stage4 = True`
   * `use_col_mamba_stage3 = False`
   * `use_col_mamba_stage4 = True`
2. Keep stages 1–2 as baseline Conv-SE-NeXt.
3. Swap stage 3 and 4 pixel blocks only.
4. Reproduce baseline training with no loss changes.
5. Confirm tensor shapes are unchanged.
6. Run ablations:

   * stage 4 only
   * stage 3+4
   * row only
   * row+col
   * with/without gate

Critical pitfall: the baseline paper already shows adding more stages or more blocks is not the right path, so resist the temptation to stack more RangeMamba blocks.  

---

# IDEA 2 — Optional training method: soft pixel supervision

The baseline assigns each pixel a **hard pseudo-label** by majority vote over all points projected into that pixel, then uses point-level CE plus pixel-level CE/Lovasz/boundary losses. This is the clean place to improve supervision without changing the network interface. 

## 2.1 What to implement

Replace the hard pixel target with a **soft class histogram** at each pixel:

* build class counts from projected points
* normalize to probabilities
* compute a soft cross-entropy or KL loss
* optionally downweight ambiguous pixels by entropy

Keep:

* point-level CE
* pixel-level boundary loss
* Lovasz on hard argmax mask only for stable baselines, or disable Lovasz on ambiguous pixels

## 2.2 File/module breakdown

Suggested files:

* `losses/soft_pixel_labels.py`
* `losses/soft_pixel_ce.py`
* `losses/pixel_entropy_weight.py`

### Module A — `SoftPixelLabelBuilder`

**Input**

* `point_labels`: `[sumN]`
* `proj_idx`: per-point pixel index for the original range image
* `num_classes`
* `ignore_index`

**Output**

* `soft_targets`: `[B,num_classes,H,W]`
* `pixel_weight`: `[B,1,H,W]`
* `hard_targets`: `[B,H,W]` for compatibility

**Functionality**

* Build a normalized per-pixel class histogram
* Estimate ambiguity from entropy

**Implementation steps**

1. Initialize zero histogram per pixel and class.
2. For each labeled point, scatter-add one-hot class to its pixel bin.
3. Normalize along class dimension.
4. Compute entropy.
5. Set `pixel_weight = 1 - entropy / log(num_classes)`.
6. Keep `hard_targets = argmax(hist)` for Lovasz/boundary compatibility.

**Pseudocode**

```python
def build_soft_pixel_labels(point_labels, proj_idx, batch_ids, H, W, num_classes, ignore_index):
    hist = torch.zeros(batch_size, num_classes, H, W, device=point_labels.device)

    valid_pts = point_labels != ignore_index
    cls_onehot = F.one_hot(point_labels[valid_pts], num_classes=num_classes).float()
    b, u, v = batch_ids[valid_pts], proj_idx[valid_pts, 0], proj_idx[valid_pts, 1]

    hist.index_put_((b, slice(None), v, u), cls_onehot.T, accumulate=True)  # or custom scatter
    count = hist.sum(dim=1, keepdim=True).clamp_min(1.0)
    soft = hist / count

    entropy = -(soft.clamp_min(1e-8).log() * soft).sum(dim=1, keepdim=True)
    weight = 1.0 - entropy / math.log(num_classes)

    hard = soft.argmax(dim=1)
    return soft, weight, hard
```

**Pitfalls**

* Do not collapse batch items together.
* Ignore unlabeled points before histogramming.
* Do not use standard `CrossEntropyLoss` with soft targets.

### Module B — `SoftPixelCELoss`

**Input**

* `pixel_logits`: `[B,num_classes,H,W]`
* `soft_targets`: `[B,num_classes,H,W]`
* `pixel_weight`: `[B,1,H,W]`

**Output**

* scalar loss

**Functionality**

* Soft CE or KL with per-pixel weighting

**Pseudocode**

```python
def soft_pixel_ce(pixel_logits, soft_targets, pixel_weight):
    logp = F.log_softmax(pixel_logits, dim=1)
    ce = -(soft_targets * logp).sum(dim=1, keepdim=True)
    return (ce * pixel_weight).sum() / pixel_weight.sum().clamp_min(1.0)
```

**Pitfalls**

* `CrossEntropyLoss` with class indices is wrong here.
* Clamp probabilities before `log`.
* Keep loss scale similar to the original pixel CE.

## 2.3 Loss integration

Baseline loss:
[
L = L^{pt}*{ce} + \lambda (L^{px}*{ce} + \beta L^{px}*{ls} + L^{px}*{bd})
]
with hard majority pseudo-labels. Keep the same overall structure first; replace only `L_px_ce` with the soft-target version, and optionally apply boundary/Lovasz only on confident or hard targets. 

**Recommended first version**

* `L_pt_ce`: unchanged
* `L_px_soft_ce`: new
* `L_px_bd`: unchanged
* `L_px_lovasz`: use `hard_targets` and optionally multiply by a confidence mask

**Pitfalls**

* Do not remove point-level supervision.
* Do not apply Lovasz directly to dense soft histograms unless you intentionally re-derive the loss.

---