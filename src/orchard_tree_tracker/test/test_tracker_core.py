#!/usr/bin/env python3

import math
import sys
import unittest
from pathlib import Path

import numpy as np


def _load_tracker_module():
    # Load the node script as a module without requiring ROS runtime.
    import importlib.util
    import sys

    ws_dir = Path(__file__).resolve().parents[3]
    script = ws_dir / "src" / "orchard_tree_tracker" / "scripts" / "fruit_tree_tracker_node.py"
    spec = importlib.util.spec_from_file_location("fruit_tree_tracker_node", str(script))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import module from: {script}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


MOD = _load_tracker_module()


class TestTrackerCore(unittest.TestCase):
    def test_roi_crop_filters(self):
        roi = MOD.RoiParams(x_min=0.0, x_max=1.0, y_min=-1.0, y_max=1.0, z_min=0.0, z_max=2.0)
        pts = np.array(
            [
                [0.5, 0.0, 1.0],  # in
                [1.5, 0.0, 1.0],  # x out
                [0.5, 2.0, 1.0],  # y out
                [0.5, 0.0, -1.0],  # z out
            ],
            dtype=np.float32,
        )
        out = MOD._roi_crop(pts, roi)
        self.assertEqual(out.shape, (1, 3))
        self.assertTrue(np.allclose(out[0], np.array([0.5, 0.0, 1.0], dtype=np.float32)))

    def test_voxel_downsample_reduces_points(self):
        pts = np.array(
            [
                [0.01, 0.01, 0.01],
                [0.02, 0.02, 0.01],  # same voxel for size=0.1
                [0.20, 0.20, 0.00],  # different voxel
            ],
            dtype=np.float32,
        )
        out = MOD._voxel_downsample(pts, voxel_size=0.1)
        self.assertEqual(out.shape[1], 3)
        self.assertEqual(out.shape[0], 2)

    def test_grid_instances_two_clusters(self):
        roi = MOD.RoiParams(x_min=0.0, x_max=10.0, y_min=-4.0, y_max=4.0, z_min=-1.0, z_max=3.0)
        grid = MOD.GridParams(cell_size=0.2, count_threshold=3)

        rng = np.random.default_rng(0)
        c1 = MOD._simulate_tree_points(rng, center_xy=(2.0, 0.0), n_points=300, crown_radius=0.25, height=1.6)
        c2 = MOD._simulate_tree_points(rng, center_xy=(7.0, 1.5), n_points=300, crown_radius=0.25, height=1.6)
        pts = np.concatenate([c1, c2], axis=0)

        obs = MOD._grid_instances(pts, roi, grid)
        self.assertGreaterEqual(len(obs), 2)
        # Pick two largest clusters and verify they are near the expected centers.
        obs_sorted = sorted(obs, key=lambda o: int(o.point_count), reverse=True)[:2]
        def near(x, y):
            for o in obs_sorted:
                if abs(float(o.cx) - float(x)) < 0.4 and abs(float(o.cy) - float(y)) < 0.4:
                    return True
            return False

        self.assertTrue(near(2.0, 0.0))
        self.assertTrue(near(7.0, 1.5))

    def test_fit_outputs_non_negative(self):
        rng = np.random.default_rng(1)
        pts = MOD._simulate_tree_points(rng, center_xy=(3.0, -1.0), n_points=500, crown_radius=0.3, height=1.2)
        cx, cy, height, crown, z_med = MOD._fit_from_points(pts)
        self.assertTrue(np.isfinite([cx, cy, height, crown, z_med]).all())
        self.assertGreaterEqual(height, 0.0)
        self.assertGreaterEqual(crown, 0.0)

    def test_confidence_is_bounded_and_monotone_wrt_missed(self):
        max_missed = 10
        c0 = MOD._confidence(point_count=80, track_age=10, missed=0, max_missed=max_missed)
        c1 = MOD._confidence(point_count=80, track_age=10, missed=1, max_missed=max_missed)
        c9 = MOD._confidence(point_count=80, track_age=10, missed=9, max_missed=max_missed)
        self.assertTrue(0.0 <= c0 <= 1.0)
        self.assertTrue(0.0 <= c1 <= 1.0)
        self.assertTrue(0.0 <= c9 <= 1.0)
        self.assertGreaterEqual(c0, c1)
        self.assertGreaterEqual(c1, c9)

    def test_tracker_filters_nan_inf(self):
        params = MOD.TrackerParams()
        tr = MOD.FruitTreeTracker(params)
        pts = np.array(
            [
                [1.0, 0.0, 1.0],
                [np.nan, 0.0, 1.0],
                [2.0, np.inf, 1.0],
            ],
            dtype=np.float32,
        )
        det = tr.process_points(pts)
        self.assertIsInstance(det, list)
        for d in det:
            self.assertTrue(math.isfinite(float(d.cx)))
            self.assertTrue(math.isfinite(float(d.cy)))

    def test_track_deletes_after_max_missed(self):
        roi = MOD.RoiParams(x_min=0.0, x_max=10.0, y_min=-4.0, y_max=4.0, z_min=-1.0, z_max=3.0)
        params = MOD.TrackerParams(
            roi=roi,
            voxel_size=0.03,
            grid=MOD.GridParams(cell_size=0.2, count_threshold=3),
            mot=MOD.MotParams(gate_distance=0.5, max_missed=2),
            fit=MOD.FitParams(window_size=3, ema_alpha=0.5),
        )
        tracker = MOD.FruitTreeTracker(params)
        rng = np.random.default_rng(2)

        # Frame 0: one tree present.
        pts0 = MOD._simulate_tree_points(rng, center_xy=(5.0, 0.0), n_points=300, crown_radius=0.2, height=1.0)
        det0 = [d for d in tracker.process_points(pts0) if d.point_count > 0]
        self.assertEqual(len(det0), 1)
        tid = int(det0[0].tree_id)

        # Frames 1..3: no points. Track should be deleted after missed > max_missed.
        empty = np.empty((0, 3), dtype=np.float32)
        tracker.process_points(empty)  # missed=1
        tracker.process_points(empty)  # missed=2
        tracker.process_points(empty)  # missed=3 -> deleted
        self.assertNotIn(tid, tracker.tracks)

    def test_test_mode_multiple_seeds_low_mismatch(self):
        # This is a "stress-like" regression check: over multiple seeds, ID mismatch should stay very low.
        # If it starts flapping, it's usually due to gating/assignment logic changes.
        args = type("Args", (), {})()
        args.frames = 120
        args.trees = 6
        args.points_per_tree = 450
        args.crown_radius = 0.30
        args.height = 1.6
        args.drift = 0.02
        args.drop_prob = 0.03
        args.min_center_dist = 1.0
        args.verbose = False
        args.roi_x_min = 0.0
        args.roi_x_max = 10.0
        args.roi_y_min = -4.0
        args.roi_y_max = 4.0
        args.roi_z_min = -0.5
        args.roi_z_max = 2.5
        args.voxel_size = 0.03
        args.cell_size = 0.10
        args.grid_T = 5
        args.gate = 0.30
        args.max_missed = 10
        args.K = 20
        args.alpha = 0.4

        # Run a handful of seeds; require a conservative mismatch ceiling.
        mismatch_rates = []
        for seed in range(10):
            args.seed = seed
            # Capture stdout so the test output stays clean.
            old_stdout = sys.stdout
            try:
                from io import StringIO

                sys.stdout = StringIO()
                MOD.run_test_mode(args)
                text = sys.stdout.getvalue()
            finally:
                sys.stdout = old_stdout
            # Parse "mismatches=X/Y (Z%)"
            for line in text.splitlines():
                if line.startswith("[test_mode] mismatches="):
                    pct = float(line.split("(")[1].split("%")[0])
                    mismatch_rates.append(pct)
                    break
        self.assertTrue(mismatch_rates)
        self.assertLessEqual(max(mismatch_rates), 2.0)


if __name__ == "__main__":
    unittest.main()
