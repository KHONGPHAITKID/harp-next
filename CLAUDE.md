# HARP-NeXt — Claude Code Context

## Project Overview

HARP-NeXt is a research implementation of a High-Speed and Accurate Range-Point Fusion Network for 3D LiDAR Semantic Segmentation (IROS 2025). It fuses range-image (2D) and point-cloud (3D) representations through a multi-scale backbone using Conv-SE-NeXt blocks. It supports two benchmarks: **nuScenes** and **SemanticKITTI**.

- Paper: https://arxiv.org/abs/2510.06876
- License: Apache 2.0

---

## Directory Layout

```
main.py                         # Single entry point for all train/eval/test runs
configs/
  main/main-config.yaml         # Training hyperparameters: optimizer, scheduler, batch size, MLflow, augmentation flags
  net/harpnext-nuscenes.yaml    # nuScenes model config: architecture, range projection, preprocessing mode
  net/harpnext-semantickitti.yaml  # SemanticKITTI model config
  net/harpnext-*-tinyvim.yaml   # TinyViM block variants of the above
core/
  network.py                    # Builds the full model from netconfig
  harpnext_core/
    segmentor/harpnext.py       # Orchestrates encoder → backbone → head pipeline
    encoder/features_encoder.py # Point/voxel feature extraction (scatter_max aggregation)
    backbone/harpnext_backbone.py # Multi-stage range-point fusion backbone
    decode_heads/harpnext_head.py # Point-wise segmentation head
    decode_heads/aux_head.py    # Auxiliary heads for deep supervision
    preprocessing/              # Range projection and voxelization
trainer/
  manager.py                    # Training loop, checkpointing, validation, MLflow logging
  scheduler.py                  # WarmupCosine LR schedule
datasets/
  nuscenes/                     # nuScenes dataset loader
  semantickitti/                # SemanticKITTI dataset loader
  pc_processors/                # Point cloud preprocessing pipeline
utils/
  loss/                         # Lovász softmax, boundary loss, cross-entropy
  metrics/                      # Semantic segmentation mIoU tracking
  transformations/              # Train-time augmentations (flip, rotate, scale, point sample)
logs/                           # Output: checkpoints (ckpt_best.pth, ckpt_last.pth), TensorBoard
pretrained/                     # Downloaded pretrained weights (not committed)
experiments/                    # Utility scripts (e.g., range image visualization)
tmp/                            # Local temp dir (redirected from /tmp to avoid disk exhaustion)
```

---

## Key Commands

All commands are run from the project root with `conda activate harpnext`.

### Train

**SemanticKITTI:**
```bash
python main.py \
  --net harpnext \
  --dataset semantic_kitti \
  --path_dataset /path/to/SemanticKITTI \
  --mainconfig ./configs/main/main-config.yaml \
  --netconfig ./configs/net/harpnext-semantickitti.yaml \
  --log_path ./logs/<run-name> \
  --gpu 0 --seed 0 --fp16
```

**nuScenes:**
```bash
python main.py \
  --net harpnext \
  --dataset nuscenes \
  --path_dataset /path/to/nuscenes \
  --mainconfig ./configs/main/main-config.yaml \
  --netconfig ./configs/net/harpnext-nuscenes.yaml \
  --log_path ./logs/<run-name> \
  --gpu 0 --seed 0 --fp16
```

### Evaluate (validation split only)

Add `--eval --restart` to any training command. `--restart` is required to load the best checkpoint.

```bash
python main.py \
  --net harpnext --dataset semantic_kitti \
  --path_dataset /path/to/SemanticKITTI \
  --mainconfig ./configs/main/main-config.yaml \
  --netconfig ./configs/net/harpnext-semantickitti.yaml \
  --log_path ./logs/<run-name> \
  --gpu 0 --seed 0 --fp16 \
  --restart --eval
```

### Resume training from checkpoint

Add `--restart` to a training command. It loads `ckpt_best.pth` (or `ckpt_last.pth`) from `--log_path`.

### Test set dump (SemanticKITTI only)

Writes `.label` files under `--test_output/sequences/<seq>/predictions/`.

```bash
python main.py \
  --net harpnext --dataset semantic_kitti \
  --path_dataset /path/to/SemanticKITTI \
  --mainconfig ./configs/main/main-config.yaml \
  --netconfig ./configs/net/harpnext-semantickitti.yaml \
  --log_path ./logs/<run-name> \
  --checkpoint ./logs/<run-name> \
  --gpu 0 --fp16 \
  --test --test_output ./logs/<run-name>/test_pred
```

---

## Config System

All model and training behavior is YAML-driven. **Never hardcode values** — always add new knobs to the appropriate config file.

### `configs/main/main-config.yaml`
Controls training-level settings shared across runs:
- `dataloader.batch_size` / `num_workers`
- `optim`: AdamW lr, weight_decay, betas, eps
- `scheduler`: max_epoch, epoch_warmup, min_lr, `save_frequency` (0 = disabled), `val_frequency` (0 = disabled)
- `augmentations.instance_cutmix`: enable/disable Instance CutMix (SemanticKITTI training only)
- `mlflow`: enable, tracking_uri, experiment_name, run_name

### `configs/net/harpnext-<dataset>.yaml`
Controls dataset- and model-specific settings:
- `model.voxel_encoder`: MLP sizes, compression, extra features (distance, cluster center)
- `model.backbone`: output_shape (range grid H×W), channels, depth, strides
- `model.decode_head`: MLP widths
- `model.auxiliary_heads`: deep supervision channel config
- `range_proj`: fov_up, fov_down, range_H, range_W
- `preproc.gpu`: set `true` to run preprocessing on GPU (requires batch_size=1 during eval)
- `augmentations`: per-dataset augmentation config
- `classif.ignore_class`: ignore index for loss computation

---

## Important Gotchas

### 1. SemanticKITTI label shift
Training labels are shifted by **-1** internally (learning labels 1–19 become 0–18). When exporting test predictions, always shift back by **+1** before applying `learning_map_inv`. This is handled in `run_semantickitti_test()` in `main.py:331`. If you add new label-handling code, follow the same convention.

### 2. GPU preprocessing requires batch_size=1 during eval
When `preproc.gpu: true` in the netconfig, the preprocessing is stateful and only safe with `batch_size: 1`. Always set this in `main-config.yaml` before running eval with GPU preprocessing. CPU mode (`preproc.gpu: false`) has no such restriction.

### 3. `instance_cutmix` must be False for validation
In `configs/main/main-config.yaml`, set `augmentations.instance_cutmix: false` before any eval run. It is only meaningful during SemanticKITTI training. Leaving it `true` during eval causes unnecessary instance extraction overhead.

### 4. `--restart` is required for eval to load weights
Running `--eval` without `--restart` will evaluate a randomly initialized model. Always pair them: `--eval --restart`.

### 5. Temp files go to `./tmp/`, not `/tmp/`
`main.py:_set_local_tmpdir()` redirects `TMPDIR`, `TMP`, `TEMP`, and `tempfile.tempdir` to `./tmp/` at startup. This prevents `/tmp` disk exhaustion on the GPU server. The `./tmp/` directory is created automatically; do not change this behavior.

### 6. Checkpoint loading strips `module.` prefix
`load_checkpoint_for_inference()` in `main.py` handles checkpoints saved under DDP (`module.*` keys) by stripping the prefix automatically. No manual intervention needed when switching between single-GPU and multi-GPU checkpoints.

---

## Environment & Workflow

- **Platform:** Remote GPU server, accessed via SSH
- **Python env:** `conda activate harpnext` (Python 3.12, PyTorch 2.5.0, CUDA 11.8)
- **No formal test suite.** Verify changes by running a short eval:
  ```bash
  python main.py --eval --restart --gpu 0 --fp16 \
    --mainconfig ./configs/main/main-config.yaml \
    --netconfig ./configs/net/harpnext-semantickitti.yaml \
    --log_path ./logs/<run-name>
  ```
- **MLflow** is configured in `main-config.yaml`. To disable tracking locally, set `mlflow.enable: false`.
- **TensorBoard** logs are written to `--log_path` during training (not during eval-only runs).
- **Checkpoints** saved as `ckpt_best.pth` (best val mIoU) and `ckpt_last.pth` in `--log_path`. Periodic saves can be enabled with `scheduler.save_frequency > 0`.
