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

Run with Monarch attention (default):

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
