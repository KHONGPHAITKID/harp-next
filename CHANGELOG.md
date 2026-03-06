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

Follow-ups / TODOs:
- Task 2: Add `_compute_cap_scores()` to `SemLaserScan`
- Task 3: Wire CAP into `set_label()`
