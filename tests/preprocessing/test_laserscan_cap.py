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
    """With high center_scores on a subset, those points should be preferred over closer ones.
    Two points are placed along the +x axis so they project to the same pixel.
    The farther point (depth=8) gets score=1.0; the closer point (depth=2) gets score=0.
    CAP should keep the farther central point.
    """
    scan = LaserScan(dataset="semantic_kitti", project_range=True,
                     range_H=64, range_W=1024, fov_up=3.0, fov_down=-25.0)
    # Both points along +x at y=0, z=0 → same yaw=0, pitch=0 → same pixel
    points = np.array([
        [8.0, 0.0, 0.0],  # index 0: farther, central → will get score=1.0
        [2.0, 0.0, 0.0],  # index 1: closer, non-central → score=0.0
    ], dtype=np.float32)
    remissions = np.zeros(2, dtype=np.float32)
    scan.set_points(points, remissions)

    py0, px0 = scan.proj_range_y[0], scan.proj_range_x[0]
    py1, px1 = scan.proj_range_y[1], scan.proj_range_x[1]
    assert (py0 == py1) and (px0 == px1), (
        "Test setup error: the two points must map to the same pixel"
    )

    # Depth-only: point 1 (depth=2, closer) wins
    assert scan.proj_range_idx[py0, px0] == 1, "Depth-only baseline: closer point should win"

    # Apply CAP scores: point 0 gets 1.0 (central), point 1 gets 0.0
    center_scores = np.array([1.0, 0.0], dtype=np.float32)
    scan.do_range_projection(center_scores=center_scores)

    # CAP: point 0 (higher centerness, score=depth/1.01) wins over point 1 (score=depth/0.01)
    assert scan.proj_range_idx[py0, px0] == 0, (
        f"CAP should keep the high-centerness point (0), got {scan.proj_range_idx[py0, px0]}"
    )
