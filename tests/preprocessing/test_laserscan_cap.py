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


import yaml
from core.harpnext_core.preprocessing.laserscan import SemLaserScan


def _make_sem_scan():
    with open("./datasets/semantickitti/semantic-kitti.yaml") as f:
        cfg = yaml.safe_load(f)
    color_dict = cfg["color_map"]
    scan = SemLaserScan(
        dataset="semantic_kitti",
        sem_color_dict=color_dict,
        project_range=True,
        range_H=64, range_W=1024,
        fov_up=3.0, fov_down=-25.0,
    )
    return scan


def test_compute_cap_scores_stuff_is_zero():
    """Points with inst_id == 0 must have score 0.0."""
    scan = _make_sem_scan()
    n = 100
    points = np.ones((n, 3), dtype=np.float32)
    remissions = np.zeros(n, dtype=np.float32)
    scan.set_points(points, remissions)
    scan.inst_label = np.zeros(n, dtype=np.uint32)  # all stuff
    scores = scan._compute_cap_scores()
    np.testing.assert_array_equal(scores, 0.0)


def test_compute_cap_scores_instance_normalized():
    """Instance points must have scores in [0, 1], with the center point scoring highest."""
    scan = _make_sem_scan()
    # 5 points: center + 4 outliers, all in instance 1
    center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    outliers = np.array([[5.0, 0, 0], [-5, 0, 0], [0, 5, 0], [0, -5, 0]], dtype=np.float32)
    points = np.vstack([center[None], outliers])
    remissions = np.zeros(5, dtype=np.float32)
    scan.set_points(points, remissions)
    scan.inst_label = np.ones(5, dtype=np.uint32)  # all same instance
    scores = scan._compute_cap_scores()
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0
    # The center point (index 0) should have the highest score
    assert scores[0] == scores.max()


def test_compute_cap_scores_small_instance_skipped():
    """Instance with fewer than 3 points must stay at score 0.0 (no crash)."""
    scan = _make_sem_scan()
    points = np.array([[1.0, 0, 0], [2.0, 0, 0]], dtype=np.float32)
    remissions = np.zeros(2, dtype=np.float32)
    scan.set_points(points, remissions)
    scan.inst_label = np.array([7, 7], dtype=np.uint32)  # instance with only 2 pts
    scores = scan._compute_cap_scores()
    np.testing.assert_array_equal(scores, 0.0)


def test_compute_cap_scores_degenerate_instance():
    """Instance where all points are at identical positions (g_max == g_min)
    must produce all-zero scores without crashing."""
    scan = _make_sem_scan()
    # 5 points all at the same location → bbox center == all points → g is constant
    points = np.tile(np.array([[3.0, 1.0, 0.5]], dtype=np.float32), (5, 1))
    remissions = np.zeros(5, dtype=np.float32)
    scan.set_points(points, remissions)
    scan.inst_label = np.ones(5, dtype=np.uint32)  # all in instance 1
    scores = scan._compute_cap_scores()
    np.testing.assert_array_equal(scores, 0.0)
