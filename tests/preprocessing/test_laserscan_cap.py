import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from core.harpnext_core.preprocessing.laserscan import LaserScan


def _make_scan():
    scan = LaserScan(dataset="semantic_kitti", project_range=True,
                     range_H=64, range_W=1024, fov_up=3.0, fov_down=-25.0)
    rng = np.random.default_rng(0)
    points = rng.uniform(-10, 10, (500, 3)).astype(np.float32)
    points[:, 0] = np.abs(points[:, 0]) + 1.0  # ensure positive x so points are in front
    remissions = rng.uniform(0, 1, 500).astype(np.float32)
    return scan, points, remissions


def test_depth_projection_unchanged():
    """With no center_scores, result must match the current depth-only baseline."""
    scan_a, points, remissions = _make_scan()
    scan_b, _, _ = _make_scan()

    scan_a.set_points(points, remissions)          # calls do_range_projection() internally
    scan_b.set_points(points, remissions)
    scan_b.do_range_projection(center_scores=None) # explicit None must behave the same

    np.testing.assert_array_equal(scan_a.proj_range_idx, scan_b.proj_range_idx)


def test_cap_projection_changes_selection():
    """With high center_scores on a subset, those points should be preferred over closer ones."""
    scan, points, remissions = _make_scan()
    scan.set_points(points, remissions)

    # Give every point a score of 0 (stuff) except one point at index 42
    center_scores = np.zeros(len(points), dtype=np.float32)
    center_scores[42] = 1.0  # maximum centerness

    # Find which pixel point 42 maps to
    scan.do_range_projection(center_scores=center_scores)
    py = scan.proj_range_y[42]
    px = scan.proj_range_x[42]

    # Point 42 must win its pixel
    assert scan.proj_range_idx[py, px] == 42, (
        f"Expected point 42 to win pixel ({py},{px}), "
        f"got index {scan.proj_range_idx[py, px]}"
    )
