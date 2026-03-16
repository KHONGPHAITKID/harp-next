# Test Utilities

This folder contains small utilities for validating and profiling HARP-NeXt modules. All commands assume you are running from the repo root.

## Setup

Use `PYTHONPATH=.` so the repo modules are importable:

```bash
PYTHONPATH=. python3 <script>
```

## Files

### `test/test_convmonarch_block.py`

Quick sanity check for `ConvMonarchBlock` forward shapes.

Run:

```bash
PYTHONPATH=. python3 test/test_convmonarch_block.py
```

### `test/benchmark_blocks.py`

Benchmarks runtime (ms/iter) of `ConvSENeXt` vs `ConvMonarchBlock` on a synthetic input.

Run:

```bash
PYTHONPATH=. python3 test/benchmark_blocks.py
```

Notes:
- Uses `H=64, W=512, C=128, B=2` by default.
- Dilation is set to `3` to keep spatial size stable with `kernel=7`.

### `test/profile_flops.py`

Profiles FLOPs for blocks and full backbones (ConvSENeXt vs ConvMonarch) using `torch.profiler`.

Run:

```bash
PYTHONPATH=. python3 test/profile_flops.py
```

Optional flags:
- `--batch_size`, `--channels`, `--height`, `--width`, `--dilation`
- `--netconfig-conv`, `--netconfig-mon`

Notes:
- FLOPs are approximate; profiler may miss fused or sparse ops.

### `test/compare_attention.py`

Compares **normal full attention** vs **Monarch attention** for time and GFLOPs.

Run (safe size):

```bash
PYTHONPATH=. python3 test/compare_attention.py --height 16 --width 128
```

Run at full SemanticKITTI size (may OOM):

```bash
PYTHONPATH=. python3 test/compare_attention.py --height 64 --width 512 --force
```

### `test/profile_model_stats.py`

Reports **params, runtime (ms), FLOPs, MACs, and peak GPU memory** for one forward pass of a model config.

Run with nuScenes preset (default):

```bash
PYTHONPATH=. python3 test/profile_model_stats.py --batch_size 1
```

Run with SemanticKITTI preset:

```bash
PYTHONPATH=. python3 test/profile_model_stats.py --dataset semantic_kitti --batch_size 1
```

Override config path explicitly:

```bash
PYTHONPATH=. python3 test/profile_model_stats.py --netconfig configs/net/harpnext-semantickitti-convmonarch.yaml --batch_size 1
```

Run with **normal full attention** (monkey-patched at runtime):

```bash
PYTHONPATH=. python3 test/profile_model_stats.py --netconfig configs/net/harpnext-semantickitti-convmonarch.yaml --batch_size 1 --attention normal
```

Notes:
- MACs are reported as `FLOPs / 2`.
- FLOPs are approximate; profiler may miss fused or sparse ops.
- Peak GPU memory requires CUDA; otherwise it prints `N/A`.

## Training Timing Profiler
This folder contains `profile_training_timing.py`, a standalone timing profiler for the HARP-NeXt training pipeline. It runs a fixed number of iterations and reports per-stage latency (ms) for:
- Data loading
- Preprocessing
- Host-to-device transfer
- Forward pass
- Loss computation
- Backward pass (optional)
- Metrics computation

The script does not train a model; it only measures performance.

### Basic Usage
```
python test/profile_training_timing.py \
--path_dataset /path/to/SemanticKITTI \
--mainconfig configs/main/main-config.yaml \
--netconfig configs/net/harpnext-semantickitti.yaml \
--gpu 0 \
--batch_size 2 \
--iters 100 \
--warmup 5 \
--mode train \
--fp16
```

### Common Options
- `--mode` (`train`|`val`): Choose training or validation timing.
- `--iters`: Number of measured iterations (after warmup).
- `--warmup`: Number of warmup iterations.
- `--batch_size`: Batch size used by the profiler.
- `--fp16`: Enable autocast for forward/loss/backward.
- `--no_backward`: Skip backward pass to isolate forward/loss.
- `--no_aug`: Disable training augmentations.
- `--no_cutmix`: Disable instance CutMix.
- `--workers`: DataLoader workers.
- `--persistent_workers`: Keep worker processes alive.
- `--prefetch_factor`: Prefetch factor for DataLoader.

### Example: Isolate Data + Forward
```
python test/profile_training_timing.py \
--path_dataset ../dataset/SemanticKitti/data_odometry_velodyne \
--mainconfig configs/main/main-config.yaml \
--netconfig configs/net/harpnext-semantickitti.yaml \
--gpu 0 \
--batch_size 2 \
--iters 100 \
--warmup 5 \
--mode train \
--no_backward \
--no_aug \
--no_cutmix
```

### Output
The script prints a timing summary with mean, p50, p90, and max (ms) per stage, plus the average total iteration time.
