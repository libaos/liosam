#!/usr/bin/env python3

import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


def _load_tracker_module():
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

try:
    import sys

    ros_py = Path("/opt/ros/noetic/lib/python3/dist-packages")
    if ros_py.is_dir() and str(ros_py) not in sys.path:
        sys.path.insert(0, str(ros_py))
except Exception:
    pass


try:
    from sensor_msgs.msg import PointField
except Exception:  # pragma: no cover
    PointField = None


@dataclass
class _Field:
    name: str
    offset: int
    datatype: int
    count: int = 1


@dataclass
class _Msg:
    fields: List[_Field]
    point_step: int
    row_step: int
    width: int
    height: int
    is_bigendian: bool
    data: bytes


class TestPointCloud2Parsing(unittest.TestCase):
    def setUp(self) -> None:
        if PointField is None:
            self.skipTest("sensor_msgs not available (source ROS env to enable PointCloud2 parsing tests)")

    def _make_msg(self, *, dtype: np.dtype, values: np.ndarray, width: int, height: int, row_pad: int = 0, is_bigendian: bool) -> _Msg:
        fields = []
        for name in dtype.names or []:
            off = int(dtype.fields[name][1])
            base = dtype.fields[name][0]
            if base == np.dtype(">f4") or base == np.dtype("<f4"):
                dt = PointField.FLOAT32
            elif base == np.dtype(">f8") or base == np.dtype("<f8"):
                dt = PointField.FLOAT64
            elif base == np.dtype(">u4") or base == np.dtype("<u4"):
                dt = PointField.UINT32
            elif base == np.dtype(">i4") or base == np.dtype("<i4"):
                dt = PointField.INT32
            else:
                raise ValueError(f"unsupported base dtype for field {name}: {base}")
            fields.append(_Field(name=str(name), offset=off, datatype=int(dt), count=1))

        point_step = int(dtype.itemsize)
        row_step = int(width) * point_step + int(row_pad)

        # Fill as a (height*width,) structured array, then lay out per-row with optional padding.
        flat = np.asarray(values, dtype=dtype).reshape((int(height) * int(width),))
        rows = []
        for r in range(int(height)):
            chunk = flat[r * int(width) : (r + 1) * int(width)].tobytes()
            rows.append(chunk)
            if row_pad:
                rows.append(b"\x00" * int(row_pad))
        data = b"".join(rows)
        return _Msg(
            fields=fields,
            point_step=point_step,
            row_step=row_step,
            width=int(width),
            height=int(height),
            is_bigendian=bool(is_bigendian),
            data=data,
        )

    def test_parses_tree_label_zero(self):
        endian = "<"
        dtype = np.dtype(
            {"names": ["x", "y", "z", "label"], "formats": [endian + "f4", endian + "f4", endian + "f4", endian + "u4"], "offsets": [0, 4, 8, 12], "itemsize": 16}
        )
        values = np.zeros((5,), dtype=dtype)
        values["x"] = [0.0, 1.0, np.nan, 3.0, 4.0]
        values["y"] = [0.0, 0.0, 0.0, 0.0, 0.0]
        values["z"] = [1.0, 1.0, 1.0, 1.0, np.inf]
        values["label"] = [0, 1, 0, 0, 0]
        msg = self._make_msg(dtype=dtype, values=values, width=5, height=1, row_pad=0, is_bigendian=False)

        dummy = type("Dummy", (), {})()
        dummy.label_field = "label"
        dummy._dtype_cache_key = None
        dummy._dtype_cache = None
        dummy._dtype_cache_fields = None

        pts = MOD.RosTreeTrackerNode._cloud_to_tree_points(dummy, msg)
        # Only points with label==0 AND finite xyz survive.
        self.assertEqual(pts.shape[1], 3)
        self.assertEqual(pts.shape[0], 2)  # indices 0 and 3
        self.assertTrue(np.all(np.isfinite(pts)))

    def test_parses_big_endian(self):
        endian = ">"
        dtype = np.dtype(
            {"names": ["x", "y", "z", "label"], "formats": [endian + "f4", endian + "f4", endian + "f4", endian + "u4"], "offsets": [0, 4, 8, 12], "itemsize": 16}
        )
        values = np.zeros((3,), dtype=dtype)
        values["x"] = [1.0, 2.0, 3.0]
        values["y"] = [0.0, 0.0, 0.0]
        values["z"] = [0.5, 0.5, 0.5]
        values["label"] = [0, 0, 0]
        msg = self._make_msg(dtype=dtype, values=values, width=3, height=1, row_pad=0, is_bigendian=True)

        dummy = type("Dummy", (), {})()
        dummy.label_field = "label"
        dummy._dtype_cache_key = None
        dummy._dtype_cache = None
        dummy._dtype_cache_fields = None

        pts = MOD.RosTreeTrackerNode._cloud_to_tree_points(dummy, msg)
        self.assertEqual(pts.shape, (3, 3))
        self.assertTrue(np.allclose(pts[:, 0], np.array([1.0, 2.0, 3.0], dtype=np.float32)))

    def test_parses_multi_row_with_padding(self):
        endian = "<"
        dtype = np.dtype(
            {"names": ["x", "y", "z", "label"], "formats": [endian + "f4", endian + "f4", endian + "f4", endian + "u4"], "offsets": [0, 4, 8, 12], "itemsize": 16}
        )
        width = 3
        height = 2
        values = np.zeros((width * height,), dtype=dtype)
        values["x"] = [0.0, 1.0, 2.0, 10.0, 11.0, 12.0]
        values["y"] = 0.0
        values["z"] = 1.0
        values["label"] = [0, 1, 0, 0, 0, 1]
        msg = self._make_msg(dtype=dtype, values=values, width=width, height=height, row_pad=8, is_bigendian=False)

        dummy = type("Dummy", (), {})()
        dummy.label_field = "label"
        dummy._dtype_cache_key = None
        dummy._dtype_cache = None
        dummy._dtype_cache_fields = None

        pts = MOD.RosTreeTrackerNode._cloud_to_tree_points(dummy, msg)
        self.assertEqual(pts.shape[0], 4)  # 4 points with label==0
        self.assertTrue(np.all(np.isfinite(pts)))
        self.assertTrue(np.allclose(sorted(pts[:, 0].tolist()), [0.0, 2.0, 10.0, 11.0]))

    def test_missing_label_field_raises(self):
        endian = "<"
        dtype = np.dtype(
            {"names": ["x", "y", "z"], "formats": [endian + "f4", endian + "f4", endian + "f4"], "offsets": [0, 4, 8], "itemsize": 12}
        )
        values = np.zeros((2,), dtype=dtype)
        values["x"] = [0.0, 1.0]
        values["y"] = [0.0, 0.0]
        values["z"] = [1.0, 1.0]
        msg = self._make_msg(dtype=dtype, values=values, width=2, height=1, row_pad=0, is_bigendian=False)

        dummy = type("Dummy", (), {})()
        dummy.label_field = "label"
        dummy._dtype_cache_key = None
        dummy._dtype_cache = None
        dummy._dtype_cache_fields = None

        with self.assertRaises(KeyError):
            MOD.RosTreeTrackerNode._cloud_to_tree_points(dummy, msg)


if __name__ == "__main__":
    unittest.main()
