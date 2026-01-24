#!/usr/bin/env python3
"""Fuse consecutive raw pointcloud frames into denser map-frame PCD chunks (offline).

Goal:
  "5帧一张"：把连续 N 帧 raw 点云对齐到 map 后合成 1 个更稠密的 PCD。

Input:
  A directory with:
    <in-dir>/frames.csv
    <in-dir>/pcd/raw_000000.pcd ...

Output:
  <out-dir>/pcd/chunk_000000.pcd
  <out-dir>/chunks.csv
  <out-dir>/frames.csv
  <out-dir>/run_meta.json

Notes:
  - Requires --bag to provide TF to transform points into map-frame.
  - No ROI filtering is applied (keeps all points, only removes NaNs).
  - Works with float32-only PCD fields (e.g. x y z intensity).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


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


def _parse_pcd_header(handle) -> Tuple[List[str], List[int], List[str], List[int], str, int]:
    header: Dict[str, str] = {}
    data_mode: Optional[str] = None
    while True:
        line = handle.readline()
        if not line:
            raise RuntimeError("Invalid PCD header (missing DATA)")
        decoded = line.decode("utf-8", errors="ignore").strip()
        if decoded and not decoded.startswith("#"):
            parts = decoded.split(maxsplit=1)
            if len(parts) == 2:
                header[parts[0].upper()] = parts[1].strip()
        if decoded.upper().startswith("DATA"):
            parts = decoded.split()
            data_mode = parts[1].lower() if len(parts) >= 2 else "ascii"
            break

    if data_mode is None:
        raise RuntimeError("Missing DATA line in PCD header")

    fields = header.get("FIELDS", "").split()
    sizes = [int(x) for x in header.get("SIZE", "").split()] if header.get("SIZE") else []
    types = header.get("TYPE", "").split()
    counts = [int(x) for x in header.get("COUNT", "").split()] if header.get("COUNT") else [1] * len(fields)
    if not fields:
        raise RuntimeError("PCD missing FIELDS")
    if len(sizes) != len(fields) or len(types) != len(fields) or len(counts) != len(fields):
        raise RuntimeError("PCD header mismatch (FIELDS/SIZE/TYPE/COUNT)")

    points = int(header.get("POINTS", "0") or "0")
    if points <= 0:
        width = int(header.get("WIDTH", "0") or "0")
        height = int(header.get("HEIGHT", "1") or "1")
        points = int(width) * int(height)
    points = max(0, int(points))
    return list(fields), sizes, list(types), counts, str(data_mode), points


def _read_pcd_float_matrix(path: Path) -> Tuple[List[str], List[int], List[str], List[int], np.ndarray]:
    with path.open("rb") as handle:
        fields, sizes, types, counts, data_mode, points = _parse_pcd_header(handle)
        if any(int(c) != 1 for c in counts):
            raise RuntimeError(f"Unsupported PCD COUNT!=1 in {path} (counts={counts})")
        if any(int(s) != 4 for s in sizes) or any(str(t).upper() != "F" for t in types):
            raise RuntimeError(f"Only float32 fields supported: {path} (SIZE={sizes}, TYPE={types})")

        dim = int(len(fields))
        if points <= 0:
            return fields, sizes, types, counts, np.empty((0, dim), dtype=np.float32)

        if str(data_mode) == "ascii":
            mat = np.loadtxt(handle, dtype=np.float32)
            if mat.ndim == 1:
                mat = mat.reshape(1, -1)
            if int(mat.shape[1]) < dim:
                raise RuntimeError(f"PCD ascii columns < fields: {path} ({mat.shape[1]} < {dim})")
            return fields, sizes, types, counts, mat[:, :dim].astype(np.float32, copy=False)

        if str(data_mode) != "binary":
            raise RuntimeError(f"Unsupported PCD DATA mode: {data_mode} ({path})")

        expected = int(points) * int(dim) * 4
        raw = handle.read(expected)
        if len(raw) < expected:
            raise RuntimeError(f"PCD data too short: {path} (expected {expected} bytes, got {len(raw)})")
        mat = np.frombuffer(raw, dtype=np.float32, count=int(points) * int(dim)).reshape((-1, dim))
        return fields, sizes, types, counts, mat.astype(np.float32, copy=False)


def _write_pcd_float_matrix(
    path: Path,
    *,
    fields: Sequence[str],
    sizes: Sequence[int],
    types: Sequence[str],
    counts: Sequence[int],
    mat: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim != 2 or int(mat.shape[1]) != int(len(fields)):
        raise ValueError("matrix shape does not match fields")

    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS {' '.join(str(f) for f in fields)}\n"
        f"SIZE {' '.join(str(int(s)) for s in sizes)}\n"
        f"TYPE {' '.join(str(t) for t in types)}\n"
        f"COUNT {' '.join(str(int(c)) for c in counts)}\n"
        f"WIDTH {int(mat.shape[0])}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {int(mat.shape[0])}\n"
        "DATA binary\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        if mat.size:
            handle.write(mat.astype(np.float32, copy=False).tobytes())


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


def _apply_transform_xyz(xyz: np.ndarray, mat: np.ndarray) -> np.ndarray:
    if xyz.size == 0:
        return xyz.astype(np.float32, copy=False).reshape((-1, 3))
    pts = xyz.astype(np.float64, copy=False).reshape((-1, 3))
    rot = mat[:3, :3]
    trans = mat[:3, 3]
    out = pts @ rot.T + trans.reshape((1, 3))
    return out.astype(np.float32, copy=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True, type=str, help="Folder with frames.csv + pcd/raw_*.pcd")
    parser.add_argument("--out-dir", default="", type=str, help="Output directory (can be Chinese)")
    parser.add_argument("--chunk-frames", type=int, default=5, help="How many consecutive frames per output PCD")
    parser.add_argument("--stride", type=int, default=5, help="Stride in frames between outputs (5 => non-overlap)")
    parser.add_argument("--max-chunks", type=int, default=0)

    parser.add_argument("--bag", required=True, type=str, help="ROS bag providing TF")
    parser.add_argument("--tf-topic", type=str, default="/tf")
    parser.add_argument("--map-frame", type=str, default="map")
    parser.add_argument("--base-frame", type=str, default="base_link_est")
    parser.add_argument("--missing-tf-policy", choices=["skip", "hold", "first"], default="first")

    args = parser.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    frames_path = (in_dir / "frames.csv").resolve()
    frames = _read_frames_csv(frames_path)

    ws_dir = Path(__file__).resolve().parents[3]
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if str(args.out_dir).strip()
        else (ws_dir / "output" / f"每5帧合成raw地图PCD_{time.strftime('%Y%m%d_%H%M%S')}")
    )
    out_pcd_dir = out_dir / "pcd"
    out_pcd_dir.mkdir(parents=True, exist_ok=True)

    chunk_frames = int(max(1, int(args.chunk_frames)))
    stride = int(max(1, int(args.stride)))
    max_chunks = int(max(0, int(args.max_chunks)))

    try:
        import rosbag  # type: ignore
    except Exception as exc:
        raise RuntimeError("--bag requires ROS1 python 'rosbag' to be available") from exc

    bag_path = Path(str(args.bag)).expanduser().resolve()
    if not bag_path.is_file():
        raise FileNotFoundError(f"--bag not found: {bag_path}")

    tf_buffer: Dict[Tuple[str, str], np.ndarray] = {}
    tf_next: Any = None
    last_map_T_base: Optional[np.ndarray] = None
    first_map_T_base: Optional[np.ndarray] = None

    def _advance_tf_to(t_sec: float) -> None:
        nonlocal tf_next
        while tf_next is not None and float(tf_next[2].to_sec()) <= float(t_sec):
            _update_tf_buffer(tf_buffer, tf_next[1])
            try:
                tf_next = next(tf_iter)
            except StopIteration:
                tf_next = None

    def _map_T_base_at(t_sec: float) -> Optional[np.ndarray]:
        nonlocal first_map_T_base, last_map_T_base, tf_next
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

    run_meta = {
        "in_dir": str(in_dir),
        "frames_csv": str(frames_path),
        "out_frames_csv": str((out_dir / "frames.csv").resolve()),
        "chunk_frames": int(chunk_frames),
        "stride": int(stride),
        "max_chunks": int(max_chunks),
        "tf_align": {
            "enabled": True,
            "bag": str(bag_path),
            "tf_topic": str(args.tf_topic),
            "map_frame": str(args.map_frame),
            "base_frame": str(args.base_frame),
            "missing_tf_policy": str(args.missing_tf_policy),
        },
        "note": "No ROI filter. Each output PCD contains points from N consecutive frames (chunked, not sliding).",
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    chunks_csv = out_dir / "chunks.csv"
    out_frames_csv = out_dir / "frames.csv"
    with rosbag.Bag(str(bag_path)) as bag:
        tf_iter = bag.read_messages(topics=[str(args.tf_topic)])
        try:
            tf_next = next(tf_iter)
        except StopIteration:
            tf_next = None

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
            schema: Optional[Tuple[List[str], List[int], List[str], List[int]]] = None
            xyz_idx: Optional[Tuple[int, int, int]] = None

            while frame_cursor < total_frames:
                if max_chunks > 0 and chunk_index >= max_chunks:
                    break

                frames_used = 0
                points_in = 0
                mats: List[np.ndarray] = []
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

                    pcd_path = _resolve_pcd_path(in_dir, (row.get("pcd_path") or row.get("pcd") or "").strip(), f"raw_{frame_idx:06d}.pcd")
                    if pcd_path is None:
                        j += 1
                        continue

                    fields, sizes, types, counts, mat = _read_pcd_float_matrix(pcd_path)
                    if schema is None:
                        schema = (fields, sizes, types, counts)
                        name_to_idx = {str(name): int(k) for k, name in enumerate(fields)}
                        if not all(key in name_to_idx for key in ("x", "y", "z")):
                            raise RuntimeError(f"PCD missing xyz fields: {pcd_path} (fields={fields})")
                        xyz_idx = (name_to_idx["x"], name_to_idx["y"], name_to_idx["z"])
                    else:
                        if (fields, sizes, types, counts) != schema:
                            raise RuntimeError(f"PCD schema mismatch: {pcd_path}")

                    if xyz_idx is None:
                        raise RuntimeError("Internal error: xyz_idx not set")

                    if mat.size:
                        xyz = np.stack([mat[:, xyz_idx[0]], mat[:, xyz_idx[1]], mat[:, xyz_idx[2]]], axis=1).astype(np.float32, copy=False)
                        finite = np.isfinite(xyz.astype(np.float64)).all(axis=1)
                        if not np.all(finite):
                            mat = mat[finite]
                            xyz = xyz[finite]
                        points_in += int(mat.shape[0])

                        map_T_base = _map_T_base_at(float(t_sec) if np.isfinite(float(t_sec)) else 0.0)
                        if map_T_base is None:
                            if str(args.missing_tf_policy) == "skip":
                                j += 1
                                continue
                            map_T_base = np.eye(4, dtype=np.float64)

                        xyz_map = _apply_transform_xyz(xyz, map_T_base)
                        mat = mat.copy()
                        mat[:, xyz_idx[0]] = xyz_map[:, 0]
                        mat[:, xyz_idx[1]] = xyz_map[:, 1]
                        mat[:, xyz_idx[2]] = xyz_map[:, 2]

                    mats.append(mat)

                    if start_frame_idx is None:
                        start_frame_idx = int(frame_idx)
                        start_t_sec = float(t_sec) if np.isfinite(float(t_sec)) else None
                    end_frame_idx = int(frame_idx)
                    end_t_sec = float(t_sec) if np.isfinite(float(t_sec)) else end_t_sec

                    frames_used += 1
                    j += 1

                if frames_used == 0:
                    break

                mat_out = np.vstack(mats).astype(np.float32, copy=False) if mats else np.empty((0, 0), dtype=np.float32)
                points_out = int(mat_out.shape[0]) if mat_out.ndim == 2 else 0

                out_pcd = (out_pcd_dir / f"chunk_{chunk_index:06d}.pcd").resolve()
                if schema is None:
                    raise RuntimeError("No valid input frames were read; cannot write output")
                _write_pcd_float_matrix(out_pcd, fields=schema[0], sizes=schema[1], types=schema[2], counts=schema[3], mat=mat_out)

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

    print(f"[OK] Done. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
