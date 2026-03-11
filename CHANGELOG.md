# Changelog

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

### HARPNeXtAttentionBlock implementation

- Added `HARPNeXtAttentionBlock` transformer encoder block for stages 3-4 of the hybrid backbone
- Provides global context via MHSA while matching the ConvSENeXt `[B, C, H, W]` interface
- Includes lazy 2D positional embedding, configurable drop-path, and FFN with GELU

Files touched:
- `core/harpnext_core/backbone/attention_block.py` (new)

### Knowledge base for coding agents

- Created `docs/knowledge/` with 8 structured Markdown files covering every component of the codebase
- Files: index, preprocessing/CAP, feature encoder, backbone, decode heads, data pipeline, training, configs
- Each file includes paper equations, function signatures, tensor shapes, data flow, invariants, and gotchas
- Purpose: agent reference for accurate feature development without hallucination

Files touched:
- `docs/knowledge/00-index.md` through `07-configs.md` (new)
- `docs/plans/2026-03-11-knowledge-base-design.md` (new)

## 2026-03-06

### Task 1: Add `center_scores` param to `do_range_projection()` for CAP support

- Added optional `center_scores` parameter to `LaserScan.do_range_projection()`; when provided, the sort key becomes `depth / (scores + 0.01)` so high-centerness points win pixel conflicts instead of nearest points
- Created TDD test suite under `tests/preprocessing/test_laserscan_cap.py` verifying both baseline-unchanged and CAP-selection behaviors

Files touched:
- `core/harpnext_core/preprocessing/laserscan.py`
- `tests/__init__.py` (new)
- `tests/preprocessing/__init__.py` (new)
- `tests/preprocessing/test_laserscan_cap.py` (new)

### CAP full implementation (Tasks 2 & 3)

- Added `SemLaserScan._compute_cap_scores()`: computes per-point centerness scores via bbox midpoint + unit-covariance Gaussian, normalized per-instance to [0,1]; stuff/small instances get 0.0
- Wired CAP into `SemLaserScan.set_label()`: calls `_compute_cap_scores()` → `do_range_projection(center_scores=...)` → `do_range_label_projection()` so central instance points win pixel collisions whenever ground-truth labels are available
- Fixed pre-existing `proj_range_mask` bug: `> 0` → `>= 0` so point at index 0 is included
- 7 unit/integration tests cover all edge cases (stuff, normalized, small instance, degenerate, integration)

Files touched:
- `core/harpnext_core/preprocessing/laserscan.py`
- `tests/preprocessing/test_laserscan_cap.py`
