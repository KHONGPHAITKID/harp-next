# CLAUDE.md Design

**Date:** 2026-02-21
**Topic:** Creating CLAUDE.md for HARP-NeXt

## Context

HARP-NeXt is a research ML codebase for 3D LiDAR semantic segmentation. The goal is to create a CLAUDE.md that gives Claude Code everything it needs to work autonomously on this project — primarily for debugging, adding new features/models, and navigating the codebase.

## Approach

Comprehensive Option B (~150–200 lines), with 6 sections.

## Sections

### 1. Project Overview
Brief description: HARP-NeXt, 3D LiDAR semantic segmentation, range-point fusion network, two benchmarks (nuScenes, SemanticKITTI).

### 2. Directory Layout
- `main.py` — entry point
- `configs/main/` — training hyperparameters
- `configs/net/` — model/dataset-specific configs
- `core/` — model architecture
- `trainer/` — training manager and LR scheduler
- `datasets/` — data loading
- `utils/` — losses, metrics, augmentations
- `logs/` — checkpoints and TensorBoard output
- `pretrained/` — pretrained weights
- `experiments/` — utility scripts (e.g., range image plotting)

### 3. Key Commands
Train, eval, test-set dump, resume from checkpoint — for both nuScenes and SemanticKITTI.

### 4. Config System
- `main-config.yaml`: optimizer, scheduler, batch size, MLflow, `instance_cutmix`, `save_frequency`, `val_frequency`
- `harpnext-*.yaml`: model architecture, range projection, preprocessing mode
- Rule: always expose new knobs via YAML; no hardcoding.

### 5. Important Gotchas
1. SemanticKITTI label shift (-1 internally, +1 on export)
2. GPU preprocessing requires batch_size=1 during eval
3. `instance_cutmix` must be False for eval
4. `--restart` required for eval to load checkpoint
5. Temp files go to `./tmp/` not `/tmp/`

### 6. Environment & Workflow
- Remote GPU server via SSH
- conda env `harpnext` (Python 3.12, PyTorch 2.5.0, CUDA 11.8)
- No formal test suite; verify with `--eval --restart`
- MLflow config in `main-config.yaml`; disable with `enable: false`

## User Approvals
- Section 1: Approved
- Section 2: Approved
- Section 3: Approved
- Section 4: Approved
- Section 5: Approved
- Section 6: Approved
