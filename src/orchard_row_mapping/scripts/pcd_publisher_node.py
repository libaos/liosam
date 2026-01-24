#!/usr/bin/env python3
"""Publish a local PCD file as a latched PointCloud2 topic for RViz."""

from __future__ import annotations

import struct
from pathlib import Path
import threading
from typing import List, Optional, Tuple

import numpy as np

if not hasattr(threading.Thread, "isAlive"):
    setattr(threading.Thread, "isAlive", threading.Thread.is_alive)

import rospy
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header


def _lzf_decompress(data: bytes, expected_size: int) -> bytes:
    """Decompress LZF data as used by PCL binary_compressed PCD files."""
    in_len = len(data)
    in_idx = 0
    out = bytearray()
    while in_idx < in_len:
        ctrl = data[in_idx]
        in_idx += 1
        if ctrl < 32:
            length = ctrl + 1
            if in_idx + length > in_len:
                raise RuntimeError("Invalid LZF data: literal run exceeds input length")
            out.extend(data[in_idx : in_idx + length])
            in_idx += length
            continue

        length = ctrl >> 5
        ref_offset = (ctrl & 0x1F) << 8
        if length == 7:
            if in_idx >= in_len:
                raise RuntimeError("Invalid LZF data: missing length byte")
            length += data[in_idx]
            in_idx += 1
        if in_idx >= in_len:
            raise RuntimeError("Invalid LZF data: missing offset byte")
        ref_offset += data[in_idx]
        in_idx += 1
        ref_offset += 1

        ref = len(out) - ref_offset
        if ref < 0:
            raise RuntimeError("Invalid LZF data: back reference before output start")
        for _ in range(length + 2):
            out.append(out[ref])
            ref += 1

    if expected_size >= 0 and len(out) != expected_size:
        raise RuntimeError(f"Invalid LZF data: expected {expected_size} bytes, got {len(out)} bytes")
    return bytes(out)


def _pcd_numpy_dtype(size: int, typ: str) -> np.dtype:
    typ = typ.upper()
    if typ == "F":
        if size == 4:
            return np.float32
        if size == 8:
            return np.float64
    if typ == "I":
        if size == 1:
            return np.int8
        if size == 2:
            return np.int16
        if size == 4:
            return np.int32
        if size == 8:
            return np.int64
    if typ == "U":
        if size == 1:
            return np.uint8
        if size == 2:
            return np.uint16
        if size == 4:
            return np.uint32
        if size == 8:
            return np.uint64
    raise RuntimeError(f"Unsupported PCD field type: TYPE={typ} SIZE={size}")


def _load_pcd_xyz(pcd_path: Path) -> np.ndarray:
    with pcd_path.open("rb") as handle:
        header_lines: List[str] = []
        data_mode: Optional[str] = None
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError(f"Invalid PCD header: {pcd_path}")
            decoded = line.decode("utf-8", errors="ignore").strip()
            header_lines.append(decoded)
            if decoded.startswith("DATA"):
                parts = decoded.split()
                data_mode = parts[1].lower() if len(parts) >= 2 else None
                break

        header: dict[str, str] = {}
        for line in header_lines:
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                header[parts[0].upper()] = parts[1]

        fields = header.get("FIELDS", "").split()
        sizes = [int(v) for v in header.get("SIZE", "").split()]
        types = header.get("TYPE", "").split()
        counts = [int(v) for v in header.get("COUNT", "").split()] if "COUNT" in header else [1] * len(fields)
        points_count = int(header.get("POINTS", header.get("WIDTH", "0"))) or 0

        if data_mode is None:
            raise RuntimeError(f"Missing DATA line in PCD: {pcd_path}")

        if data_mode == "ascii":
            data = np.loadtxt(handle, dtype=np.float32)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            name_to_index = {name: idx for idx, name in enumerate(fields)}
            if not all(name in name_to_index for name in ("x", "y", "z")):
                raise RuntimeError(f"PCD missing xyz fields: {pcd_path} ({fields})")
            return data[:, [name_to_index["x"], name_to_index["y"], name_to_index["z"]]].astype(np.float32)

        if data_mode not in ("binary", "binary_compressed"):
            raise RuntimeError(f"Unsupported PCD DATA mode: {data_mode} ({pcd_path})")

        if not fields or not sizes or not types:
            raise RuntimeError(f"Incomplete PCD header (FIELDS/SIZE/TYPE): {pcd_path}")
        if len(fields) != len(sizes) or len(fields) != len(types) or len(fields) != len(counts):
            raise RuntimeError(f"PCD header length mismatch: {pcd_path}")

        dtype_fields: List[Tuple[str, np.dtype]] = []
        for name, size, typ, count in zip(fields, sizes, types, counts):
            if count != 1:
                for i in range(count):
                    dtype_fields.append((f"{name}_{i}", _pcd_numpy_dtype(size, typ)))
            else:
                dtype_fields.append((name, _pcd_numpy_dtype(size, typ)))
        dtype = np.dtype(dtype_fields)

        point_step = int(dtype.itemsize)
        expected_bytes = int(points_count) * point_step

        if data_mode == "binary":
            raw = handle.read(expected_bytes)
            if len(raw) < expected_bytes:
                raise RuntimeError(f"PCD data too short: expected {expected_bytes} bytes, got {len(raw)} ({pcd_path})")
            arr = np.frombuffer(raw, dtype=dtype, count=points_count)
        else:
            header_bytes = handle.read(8)
            if len(header_bytes) < 8:
                raise RuntimeError(f"PCD compressed header too short: {pcd_path}")
            compressed_size, uncompressed_size = struct.unpack("<II", header_bytes)
            compressed = handle.read(int(compressed_size))
            if len(compressed) < int(compressed_size):
                raise RuntimeError(
                    f"PCD compressed data too short: expected {compressed_size} bytes, got {len(compressed)} ({pcd_path})"
                )
            decompressed = _lzf_decompress(compressed, int(uncompressed_size))
            if int(uncompressed_size) < expected_bytes:
                raise RuntimeError(
                    f"PCD decompressed data too short: expected >= {expected_bytes} bytes, got {uncompressed_size} ({pcd_path})"
                )

            in_bytes = np.frombuffer(decompressed, dtype=np.uint8, count=expected_bytes)
            out_bytes = np.empty((points_count, point_step), dtype=np.uint8)

            in_offset = 0
            out_offset = 0
            for size, count in zip(sizes, counts):
                field_step = int(size) * int(count)
                block_len = field_step * int(points_count)
                field_block = in_bytes[in_offset : in_offset + block_len]
                if field_block.size != block_len:
                    raise RuntimeError(f"PCD compressed payload truncated for field block ({pcd_path})")
                out_bytes[:, out_offset : out_offset + field_step] = field_block.reshape(points_count, field_step)
                in_offset += block_len
                out_offset += field_step

            arr = np.frombuffer(out_bytes.tobytes(), dtype=dtype, count=points_count)

        if not all(name in arr.dtype.names for name in ("x", "y", "z")):
            raise RuntimeError(f"PCD missing xyz fields: {pcd_path} ({arr.dtype.names})")
        xyz = np.vstack([arr["x"], arr["y"], arr["z"]]).T
        return xyz.astype(np.float32)


class PcdPublisherNode:
    def __init__(self) -> None:
        pcd_param = str(rospy.get_param("~pcd_path", "")).strip()
        if not pcd_param:
            raise RuntimeError("~pcd_path is required")
        self.pcd_path = Path(pcd_param).expanduser()
        if not self.pcd_path.is_file():
            raise RuntimeError(f"PCD not found: {self.pcd_path}")

        self.topic = str(rospy.get_param("~topic", "/orchard_tree_map_builder/tree_map")).strip()
        self.frame_id = str(rospy.get_param("~frame_id", "map")).strip() or "map"
        self.publish_max_points = int(rospy.get_param("~publish_max_points", 0))
        self.sample_seed = int(rospy.get_param("~sample_seed", 0))

        rospy.loginfo("[orchard_row_mapping] Loading PCD for publishing: %s", self.pcd_path)
        points = _load_pcd_xyz(self.pcd_path)
        if points.size == 0:
            raise RuntimeError(f"Empty PCD: {self.pcd_path}")

        if self.publish_max_points > 0 and points.shape[0] > self.publish_max_points:
            rng = np.random.default_rng(int(self.sample_seed))
            idx = rng.choice(points.shape[0], int(self.publish_max_points), replace=False)
            points = points[idx]

        header = Header()
        header.stamp = rospy.Time(0)
        header.frame_id = self.frame_id
        cloud_msg = pc2.create_cloud_xyz32(header, points.tolist())

        self._pub = rospy.Publisher(self.topic, PointCloud2, queue_size=1, latch=True)
        self._pub.publish(cloud_msg)
        rospy.loginfo(
            "[orchard_row_mapping] Published PCD: topic=%s frame=%s points=%d",
            self.topic,
            self.frame_id,
            points.shape[0],
        )


def main() -> None:
    rospy.init_node("pcd_publisher_node")
    PcdPublisherNode()
    rospy.spin()


if __name__ == "__main__":
    main()
