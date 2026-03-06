# Changelog

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
