# HARP-NeXt Model Architecture

This doc summarizes the model architecture as implemented in the codebase, with references to the key files that define each component.

## High-Level Flow

1. **Network build** (`core/network.py`)
   - Constructs `HARPNeXt` using three main configs: `voxel_encoder`, `backbone`, `decode_head`.
   - Optionally adds multiple auxiliary heads for deep supervision.

2. **Segmentor** (`core/harpnext_core/segmentor/harpnext.py`)
   - Orchestrates the pipeline:
     - `FeaturesEncoder` (voxel/point feature extraction)
     - `HARPNeXtBackbone` (multi-stage fusion backbone)
     - `HARPNeXtHead` (point-wise segmentation head)
     - `AuxHead` list (optional auxiliary outputs)

```
batch_inputs_dict
  └─ voxels (features + coors)
        ├─ FeaturesEncoder
        ├─ HARPNeXtBackbone
        ├─ HARPNeXtHead
        └─ AuxHead(s)
```

## Component Details

### 1) Feature Encoder (Point/Voxel Features)
**File:** `core/harpnext_core/encoder/features_encoder.py`

- Inputs: `voxel_dict['voxels']`, `voxel_dict['coors']`.
- Optional feature augmentation:
  - Distance (`with_distance`)
  - Cluster center offset (`with_cluster_center`)
  - Pre-norm (`with_pre_norm`)
- MLP stack (`ffe_layers`) applied per point.
- Voxel aggregation via `torch_scatter.scatter_max`.
- Optional compression layer (`feat_compression`).

**Outputs added to `voxel_dict`:**
- `voxel_feats`: aggregated voxel features.
- `voxel_coors`: unique voxel coordinates.
- `point_feats`: per-layer point features.

### 2) HARPNeXt Backbone (Hybrid Point-Cluster-Pixel)
**File:** `core/harpnext_core/backbone/harpnext_backbone.py`

**Key ideas:**
- Uses a lightweight ConvSENeXt block (depthwise + pointwise conv + SE + residual).
- Performs repeated **point ⇄ cluster ⇄ pixel** transformations through
  `EfficientTransformationPipeline` (ETP).
- Fuses point features and pixel features at each stage with attention.

**Stages:**
- **Stem**
  - Pixel stem: 2D convs over dense range grid.
  - Point stem: MLP over fused point features.
  - Fusion stem: 2D conv to combine stem pixel features with point-derived pixels.
- **Residual stages**
  - A sequence of ConvSENeXt blocks.
  - Each stage:
    - Pixel → point fusion
    - (except last) Point → cluster → pixel fusion
    - Attention fusion with previous pixel features
- **Multi-scale fusion**
  - All stage outputs upsampled to stem resolution.
  - Concatenated and passed through `fuse_layers` (pixel) and `point_fuse_layers` (point).

**Outputs added to `voxel_dict`:**
- `voxel_feats`: list where index `0` is the fused pixel feature map.
- `point_feats_backbone`: list where index `0` is fused point features.

### 3) Main Decode Head
**File:** `core/harpnext_core/decode_heads/harpnext_head.py`

- Maps pixel features back to points using point coordinates.
- Applies MLP stack (`middle_channels`).
- Residual adds:
  - First MLP output adds `point_feats_backbone[0]`.
  - Later MLPs add earlier point features from the encoder.
- Final classifier is a per-point `Linear` layer.

**Output:**
- `seg_logit` for each point (classes per point).

### 4) Auxiliary Heads (Optional)
**File:** `core/harpnext_core/decode_heads/aux_head.py`

- Simple 2D conv head on intermediate `voxel_feats` levels.
- Used for deep supervision during training.
- Configured in `core/network.py` with different `indices`.

## Config-Driven Parameters
**Example (SemanticKITTI):** `configs/net/harpnext-semantickitti.yaml`

- `model.voxel_encoder`: point feature MLP sizes, compression, extra features.
- `model.backbone`: output grid size (`output_shape`), depth, channels, strides.
- `model.decode_head`: MLP widths and classifier settings.
- `model.auxiliary_heads`: channels and kernel sizes for deep supervision.

The backbone `output_shape` matches the range projection size expected by the dataset configuration.

## Data Interface Assumptions

The model expects `batch_inputs_dict` with:
- `voxels`: point features (e.g., XYZ + intensity).
- `coors`: integer coordinates (batch, y, x) aligning points to the range image grid.

These are produced by the dataset processor / preprocessing pipeline (see `datasets/pc_processors` and `core/harpnext_core/preprocessing`).

## File Map

- Build + wiring: `core/network.py`
- Segmentor: `core/harpnext_core/segmentor/harpnext.py`
- Feature encoder: `core/harpnext_core/encoder/features_encoder.py`
- Backbone: `core/harpnext_core/backbone/harpnext_backbone.py`
- Main head: `core/harpnext_core/decode_heads/harpnext_head.py`
- Auxiliary head: `core/harpnext_core/decode_heads/aux_head.py`
