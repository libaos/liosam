#!/usr/bin/env python3
"""Fuse consecutive tree-point frames into denser map-frame PCD chunks (offline).

Why:
  Single-frame tree points can look sparse. This tool groups frames into
  N-frame chunks (e.g. 5 frames per output) and saves one denser PCD per chunk.

Input:
  A folder produced by `segment_bag_to_tree_pcd.py` (or similar):
    <in-dir>/frames.csv
    <in-dir>/pcd/tree_000000.pcd ...

Output:
  <out-dir>/pcd/chunk_000000.pcd
  <out-dir>/chunks.csv
  <out-dir>/frames.csv
  <out-dir>/run_meta.json

Notes:
  - If --bag is provided, we use TF from that rosbag to transform each frame
    from base-frame (default: base_link_est) into map-frame (default: map).
  - We do NOT export huge sliding-window sequences here; it is chunked output
    ("5帧一张") to keep file count and disk usage reasonable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _load_tree_circles_impl() -> Any:
    impl_path = Path(__file__).resolve().parents[1] / "scripts" / "orchard_tree_circles_node.py"
    if not impl_path.is_file():
        raise RuntimeError(f"Cannot find implementation: {impl_path}")

    import importlib.util

    spec = importlib.util.spec_from_file_location("orchard_tree_circles_node", str(impl_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for: {impl_path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses expect the module to be present in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_frames_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"frames.csv not found: {path}")
    out: List[Dict[str, str]] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            out.append({str(k): str(v) for k, v in row.items() if k is not None})
    if not out:
        raise RuntimeError(f"frames.csv is empty: {path}")
    return out


def _resolve_pcd_path(in_dir: Path, pcd_path_str: str, fallback_name: str) -> Optional[Path]:
    pcd_path_str = (pcd_path_str or "").strip()
    if pcd_path_str:
        candidate = Path(pcd_path_str).expanduser()
        pcd_path = candidate.resolve() if candidate.is_absolute() else (in_dir / candidate).resolve()
        if pcd_path.is_file():
            return pcd_path
        fallback = (in_dir / "pcd" / Path(pcd_path_str).name).resolve()
        if fallback.is_file():
            return fallback
    fallback2 = (in_dir / "pcd" / fallback_name).resolve()
    if fallback2.is_file():
        return fallback2
    return None


def _write_pcd_xyz(path: Path, xyz: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(xyz, dtype=np.float32).reshape((-1, 3))
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {int(pts.shape[0])}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {int(pts.shape[0])}\n"
        "DATA binary\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        if pts.size:
            handle.write(pts.astype(np.float32, copy=False).tobytes())


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, zz = qx * x2, qy * y2, qz * z2
    xy, xz, yz = qx * y2, qx * z2, qy * z2
    wx, wy, wz = qw * x2, qw * y2, qw * z2
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def _transform_matrix(translation: Tuple[float, float, float], rotation: Tuple[float, float, float, float]) -> np.ndarray:
    tx, ty, tz = translation
    qx, qy, qz, qw = rotation
    rot = _quat_to_rot(qx, qy, qz, qw)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    mat[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return mat


def _invert_transform(mat: np.ndarray) -> np.ndarray:
    rot = mat[:3, :3]
    trans = mat[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = rot.T
    inv[:3, 3] = -rot.T @ trans
    return inv


def _update_tf_buffer(buffer: Dict[Tuple[str, str], np.ndarray], tf_msg: Any) -> None:
    for tr in getattr(tf_msg, "transforms", []) or []:
        parent = str(tr.header.frame_id)
        child = str(tr.child_frame_id)
        t = tr.transform.translation
        q = tr.transform.rotation
        buffer[(parent, child)] = _transform_matrix((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))


def _lookup_transform(buffer: Dict[Tuple[str, str], np.ndarray], target: str, source: str) -> Optional[np.ndarray]:
    if target == source:
        return np.eye(4, dtype=np.float64)
    adjacency: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    for (parent, child), mat in buffer.items():
        adjacency.setdefault(child, []).append((parent, mat))
        adjacency.setdefault(parent, []).append((child, _invert_transform(mat)))
    visited = set()
    queue: List[Tuple[str, np.ndarray]] = [(source, np.eye(4, dtype=np.float64))]
    visited.add(source)
    while queue:
        frame, mat = queue.pop(0)
        if frame == target:
            return mat
        for nxt, edge in adjacency.get(frame, []):
            if nxt in visited:
                continue
            visited.add(nxt)
            queue.append((nxt, edge @ mat))
    return None


def _apply_transform(points_xyz: np.ndarray, mat: np.ndarray) -> np.ndarray:
    if points_xyz.size == 0:
        return points_xyz.astype(np.float32, copy=False).reshape((-1, 3))
    pts = points_xyz.astype(np.float64, copy=False).reshape((-1, 3))
    rot = mat[:3, :3]
    trans = mat[:3, 3]
    out = pts @ rot.T + trans.reshape((1, 3))
    return out.astype(np.float32, copy=False)


def _voxel_downsample(points_xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    points_xyz = np.asarray(points_xyz, dtype=np.float32).reshape((-1, 3))
    voxel_size = float(voxel_size)
    if points_xyz.size == 0 or not (voxel_size > 0.0):
        return points_xyz
    keys = np.floor(points_xyz.astype(np.float64) / float(voxel_size)).astype(np.int32)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx.astype(np.int64, copy=False))
    return points_xyz[unique_idx]


def _filter_points(
    points_xyz: np.ndarray,
    *,
    z_min: float,
    z_max: float,
    x_min: float,
    x_max: float,
    y_abs_max: float,
) -> np.ndarray:
    points_xyz = np.asarray(points_xyz, dtype=np.float32).reshape((-1, 3))
    if points_xyz.size == 0:
        return points_xyz
    mask = np.isfinite(points_xyz.astype(np.float64)).all(axis=1)
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]
    mask &= z >= float(z_min)
    mask &= z <= float(z_max)
    mask &= x >= float(x_min)
    mask &= x <= float(x_max)
    mask &= np.abs(y) <= float(y_abs_max)
    return points_xyz[mask]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True, type=str, help="Folder with frames.csv + pcd/tree_*.pcd")
    parser.add_argument("--out-dir", default="", type=str, help="Output directory (can be Chinese)")
    parser.add_argument("--chunk-frames", type=int, default=5, help="How many consecutive frames per output PCD")
    parser.add_argument("--stride", type=int, default=5, help="Stride in frames between outputs (5 => non-overlap)")
    parser.add_argument("--max-chunks", type=int, default=0)

    parser.add_argument("--bag", type=str, default="", help="ROS bag providing TF (recommended). If empty, no TF is applied.")
    parser.add_argument("--tf-topic", type=str, default="/tf")
    parser.add_argument("--map-frame", type=str, default="map")
    parser.add_argument("--base-frame", type=str, default="base_link_est")
    parser.add_argument("--missing-tf-policy", choices=["skip", "hold", "first"], default="first")

    parser.add_argument("--z-min", type=float, default=-1.0)
    parser.add_argument("--z-max", type=float, default=4.0)
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=40.0)
    parser.add_argument("--y-abs-max", type=float, default=12.0)

    parser.add_argument("--voxel-size", type=float, default=0.0, help="Optional voxel size in map-frame (0 disables)")
    args = parser.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    frames_path = (in_dir / "frames.csv").resolve()
    frames = _read_frames_csv(frames_path)

    ws_dir = Path(__file__).resolve().parents[3]
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if str(args.out_dir).strip()
        else (ws_dir / "output" / f"每5帧合成树点地图PCD_{time.strftime('%Y%m%d_%H%M%S')}")
    )
    out_pcd_dir = out_dir / "pcd"
    out_pcd_dir.mkdir(parents=True, exist_ok=True)

    chunk_frames = int(max(1, int(args.chunk_frames)))
    stride = int(max(1, int(args.stride)))
    max_chunks = int(max(0, int(args.max_chunks)))

    tf_enabled = bool(str(args.bag).strip())
    tf_buffer: Dict[Tuple[str, str], np.ndarray] = {}
    tf_bag: Any = None
    tf_iter: Any = None
    tf_next: Any = None
    last_map_T_base: Optional[np.ndarray] = None
    first_map_T_base: Optional[np.ndarray] = None

    def _tf_setup() -> None:
        nonlocal tf_bag, tf_iter, tf_next
        if not tf_enabled:
            return
        try:
            import rosbag  # type: ignore
        except Exception as exc:
            raise RuntimeError("--bag requires ROS1 python 'rosbag' to be available") from exc

        bag_path = Path(str(args.bag)).expanduser().resolve()
        if not bag_path.is_file():
            raise FileNotFoundError(f"--bag not found: {bag_path}")
        tf_bag = rosbag.Bag(str(bag_path))
        tf_iter = tf_bag.read_messages(topics=[str(args.tf_topic)])
        try:
            tf_next = next(tf_iter)
        except StopIteration:
            tf_next = None

    def _tf_teardown() -> None:
        nonlocal tf_bag
        if tf_bag is not None:
            try:
                tf_bag.close()
            except Exception:
                pass
            tf_bag = None

    def _advance_tf_to(t_sec: float) -> None:
        nonlocal tf_next
        if not tf_enabled:
            return
        while tf_next is not None and float(tf_next[2].to_sec()) <= float(t_sec):
            _update_tf_buffer(tf_buffer, tf_next[1])
            try:
                tf_next = next(tf_iter)
            except StopIteration:
                tf_next = None

    def _map_T_base_at(t_sec: float) -> Optional[np.ndarray]:
        nonlocal first_map_T_base, last_map_T_base, tf_next
        if not tf_enabled:
            return np.eye(4, dtype=np.float64)

        _advance_tf_to(float(t_sec))
        mat = _lookup_transform(tf_buffer, str(args.map_frame), str(args.base_frame))
        if mat is not None:
            last_map_T_base = mat
            if first_map_T_base is None:
                first_map_T_base = mat
            return mat

        policy = str(args.missing_tf_policy)
        if policy == "hold":
            return last_map_T_base
        if policy == "first":
            if first_map_T_base is not None:
                return first_map_T_base
            while tf_next is not None:
                _update_tf_buffer(tf_buffer, tf_next[1])
                try:
                    tf_next = next(tf_iter)
                except StopIteration:
                    tf_next = None
                mat2 = _lookup_transform(tf_buffer, str(args.map_frame), str(args.base_frame))
                if mat2 is not None:
                    first_map_T_base = mat2
                    last_map_T_base = mat2
                    return mat2
        return None

    impl = _load_tree_circles_impl()

    run_meta = {
        "in_dir": str(in_dir),
        "frames_csv": str(frames_path),
        "out_frames_csv": str((out_dir / "frames.csv").resolve()),
        "chunk_frames": int(chunk_frames),
        "stride": int(stride),
        "max_chunks": int(max_chunks),
        "filter": {"z_min": float(args.z_min), "z_max": float(args.z_max), "x_min": float(args.x_min), "x_max": float(args.x_max), "y_abs_max": float(args.y_abs_max)},
        "voxel_size": float(args.voxel_size),
        "tf_align": {
            "enabled": bool(tf_enabled),
            "bag": str(Path(str(args.bag)).expanduser().resolve()) if tf_enabled else "",
            "tf_topic": str(args.tf_topic),
            "map_frame": str(args.map_frame),
            "base_frame": str(args.base_frame),
            "missing_tf_policy": str(args.missing_tf_policy),
        },
        "note": "Each output PCD contains points from N consecutive frames (chunked, not sliding).",
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    chunks_csv = out_dir / "chunks.csv"
    out_frames_csv = out_dir / "frames.csv"
    try:
        _tf_setup()
        with chunks_csv.open("w", newline="") as handle_chunks, out_frames_csv.open("w", newline="") as handle_frames:
            chunks_writer = csv.writer(handle_chunks)
            frames_writer = csv.writer(handle_frames)
            frames_writer.writerow(["index", "t_sec", "start_frame", "end_frame", "frames", "points", "pcd_path"])
            chunks_writer.writerow(
                [
                    "chunk_index",
                    "start_frame",
                    "end_frame",
                    "start_t_sec",
                    "end_t_sec",
                    "frames",
                    "points_in",
                    "points_out",
                    "pcd_path",
                ]
            )

            total_frames = int(len(frames))
            chunk_index = 0
            frame_cursor = 0
            while frame_cursor < total_frames:
                if max_chunks > 0 and chunk_index >= max_chunks:
                    break

                frames_used = 0
                points_in = 0
                chunk_points: List[np.ndarray] = []
                start_frame_idx: Optional[int] = None
                end_frame_idx: Optional[int] = None
                start_t_sec: Optional[float] = None
                end_t_sec: Optional[float] = None

                j = frame_cursor
                while j < total_frames and frames_used < chunk_frames:
                    row = frames[j]
                    idx_str = (row.get("index") or row.get("frame") or row.get("idx") or "").strip()
                    try:
                        frame_idx = int(idx_str) if idx_str else int(j)
                    except Exception:
                        frame_idx = int(j)

                    t_sec_text = (row.get("t_sec") or row.get("t") or "").strip()
                    try:
                        t_sec = float(t_sec_text)
                    except Exception:
                        t_sec = float("nan")

                    pcd_path = _resolve_pcd_path(in_dir, (row.get("pcd_path") or row.get("pcd") or "").strip(), f"tree_{frame_idx:06d}.pcd")
                    if pcd_path is None:
                        j += 1
                        continue

                    pts_base = impl._load_pcd_xyz(pcd_path).astype(np.float32)
                    if pts_base.size:
                        pts_base = _filter_points(
                            pts_base,
                            z_min=float(args.z_min),
                            z_max=float(args.z_max),
                            x_min=float(args.x_min),
                            x_max=float(args.x_max),
                            y_abs_max=float(args.y_abs_max),
                        )

                    map_T_base = _map_T_base_at(float(t_sec)) if math.isfinite(float(t_sec)) else _map_T_base_at(0.0)
                    if map_T_base is None:
                        if str(args.missing_tf_policy) == "skip":
                            j += 1
                            continue
                        map_T_base = np.eye(4, dtype=np.float64)

                    pts_map = _apply_transform(pts_base, map_T_base)
                    if pts_map.size:
                        chunk_points.append(pts_map)
                        points_in += int(pts_map.shape[0])

                    if start_frame_idx is None:
                        start_frame_idx = int(frame_idx)
                        start_t_sec = float(t_sec) if math.isfinite(float(t_sec)) else None
                    end_frame_idx = int(frame_idx)
                    end_t_sec = float(t_sec) if math.isfinite(float(t_sec)) else end_t_sec

                    frames_used += 1
                    j += 1

                if frames_used == 0:
                    break

                if chunk_points:
                    pts_out = np.vstack(chunk_points).astype(np.float32, copy=False)
                else:
                    pts_out = np.empty((0, 3), dtype=np.float32)

                pts_out = _voxel_downsample(pts_out, float(args.voxel_size))
                points_out = int(pts_out.shape[0])

                out_pcd = (out_pcd_dir / f"chunk_{chunk_index:06d}.pcd").resolve()
                _write_pcd_xyz(out_pcd, pts_out)

                chunks_writer.writerow(
                    [
                        int(chunk_index),
                        int(start_frame_idx) if start_frame_idx is not None else "",
                        int(end_frame_idx) if end_frame_idx is not None else "",
                        f"{start_t_sec:.6f}" if start_t_sec is not None else "",
                        f"{end_t_sec:.6f}" if end_t_sec is not None else "",
                        int(frames_used),
                        int(points_in),
                        int(points_out),
                        str(out_pcd),
                    ]
                )
                t_ref = end_t_sec if end_t_sec is not None else start_t_sec
                frames_writer.writerow(
                    [
                        int(chunk_index),
                        f"{t_ref:.6f}" if t_ref is not None else "",
                        int(start_frame_idx) if start_frame_idx is not None else "",
                        int(end_frame_idx) if end_frame_idx is not None else "",
                        int(frames_used),
                        int(points_out),
                        str(out_pcd),
                    ]
                )

                chunk_index += 1
                frame_cursor += int(stride)
                if chunk_index % 50 == 0:
                    print(f"[OK] wrote {chunk_index} chunks")
    finally:
        _tf_teardown()

    print(f"[OK] Done. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
