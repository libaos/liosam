import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


def _pkg_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_nn_node_module():
    nn_node_path = _pkg_dir() / "scripts" / "nn_scancontext_node.py"
    spec = importlib.util.spec_from_file_location("nn_scancontext_node", nn_node_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestNNScanContextOffline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(_pkg_dir()))
        cls.nn_node = _load_nn_node_module()

    def test_scan_context_empty_returns_zeros(self) -> None:
        from orchard_nn_scancontext.scan_context import ScanContext

        sc = ScanContext(num_sectors=60, num_rings=20)
        desc = sc.generate_scan_context(np.empty((0, 3), dtype=np.float32))
        self.assertEqual(desc.shape, (20, 60))
        self.assertEqual(desc.dtype, np.float32)
        self.assertTrue(np.all(desc == 0.0))

    def test_scan_context_normalized_to_unit_range(self) -> None:
        from orchard_nn_scancontext.scan_context import ScanContext

        sc = ScanContext(num_sectors=60, num_rings=20, min_range=0.1, max_range=80.0, height_lower_bound=-1.0, height_upper_bound=9.0)
        points = np.asarray(
            [
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 2.0],  # same cell, higher z -> max
                [2.0, 0.0, 0.5],
            ],
            dtype=np.float32,
        )
        desc = sc.generate_scan_context(points)
        self.assertEqual(desc.shape, (20, 60))
        self.assertGreater(desc.max(), 0.0)
        self.assertLessEqual(float(desc.max()), 1.0)
        self.assertGreaterEqual(float(desc.min()), 0.0)

    def test_downsample_xyz_caps_max_points(self) -> None:
        xyz = np.random.RandomState(0).randn(1200, 3).astype(np.float32)
        xyz_ds = self.nn_node._downsample_xyz(xyz, max_points=600)
        self.assertLessEqual(xyz_ds.shape[0], 600)
        self.assertEqual(xyz_ds.shape[1], 3)
        np.testing.assert_allclose(xyz_ds[0], xyz[0])

    def test_default_model_path_exists(self) -> None:
        p = self.nn_node._default_model_path()
        self.assertTrue(Path(p).is_file(), msg=f"missing default model: {p}")

    def test_pointcloud_to_route_id_confidence(self) -> None:
        from orchard_nn_scancontext.scan_context import ScanContext

        import sensor_msgs.point_cloud2 as pc2
        from std_msgs.msg import Header

        points = []
        rng = np.random.RandomState(1)
        for _ in range(2000):
            x = float(rng.uniform(1.0, 20.0))
            y = float(rng.uniform(-5.0, 5.0))
            z = float(rng.uniform(-0.5, 3.0))
            points.append((x, y, z))
        cloud = pc2.create_cloud_xyz32(Header(), points)

        xyz = self.nn_node._cloud_to_xyz(cloud)
        self.assertEqual(xyz.shape[1], 3)
        self.assertGreater(xyz.shape[0], 0)

        xyz = self.nn_node._downsample_xyz(xyz, max_points=60000)
        sc = ScanContext(num_sectors=60, num_rings=20, min_range=0.1, max_range=80.0, height_lower_bound=-1.0, height_upper_bound=9.0)
        desc = sc.generate_scan_context(xyz)

        import torch

        model = self.nn_node._load_model(
            model_type="simple2dcnn",
            num_classes=20,
            device=torch.device("cpu"),
            checkpoint_path=Path(self.nn_node._default_model_path()),
        )
        x = torch.from_numpy(desc).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)
        route_id = int(pred.item())
        route_conf = float(conf.item())

        self.assertGreaterEqual(route_id, 0)
        self.assertLess(route_id, 20)
        self.assertGreaterEqual(route_conf, 0.0)
        self.assertLessEqual(route_conf, 1.0)


if __name__ == "__main__":
    unittest.main()

