#!/usr/bin/env python3
"""Accumulate per-frame PCDs into sliding-window PCDs (offline).

This is meant for "one frame is too sparse" cases: we take existing exported
PCD sequences (raw/tree/colored) and create a new sequence where each output
frame contains points from the last N frames (sliding window).

Input folder format (examples in this workspace):
  - raw frames:     FIELDS x y z intensity
  - tree frames:    FIELDS x y z
  - colored frames: FIELDS x y z intensity rgb label

Input:
  <in-dir>/frames.csv
  <in-dir>/pcd/*.pcd

Output:
  <out-dir>/pcd/<same-basename>.pcd
  <out-dir>/frames.csv
  <out-dir>/run_meta.json

Note:
  Accumulation is *sliding window*: output still has one PCD per frame index.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

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


def _resolve_pcd_path(in_dir: Path, pcd_path_str: str) -> Optional[Path]:
    pcd_path_str = (pcd_path_str or "").strip()
    if not pcd_path_str:
        return None
    candidate = Path(pcd_path_str).expanduser()
    pcd_path = candidate.resolve() if candidate.is_absolute() else (in_dir / candidate).resolve()
    if pcd_path.is_file():
        return pcd_path
    fallback = (in_dir / "pcd" / Path(pcd_path_str).name).resolve()
    if fallback.is_file():
        return fallback
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
            raise RuntimeError(f"Only float32 fields supported for accumulation: {path} (SIZE={sizes}, TYPE={types})")

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True, type=str, help="Input folder with frames.csv + pcd/*.pcd")
    parser.add_argument("--out-dir", default="", type=str, help="Output folder (can be Chinese)")
    parser.add_argument("--accumulate-frames", type=int, default=5, help="Sliding window size N")
    parser.add_argument("--resume", action="store_true", help="Skip already-exported frames (keeps window state).")
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    frames_path = in_dir / "frames.csv"
    frames = _read_frames_csv(frames_path)

    ws_dir = Path(__file__).resolve().parents[3]
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if str(args.out_dir).strip()
        else (ws_dir / "output" / f"累积点云帧_{time.strftime('%Y%m%d_%H%M%S')}")
    )
    out_pcd_dir = out_dir / "pcd"
    out_pcd_dir.mkdir(parents=True, exist_ok=True)

    accum_n = int(max(1, args.accumulate_frames))

    run_meta = {
        "in_dir": str(in_dir),
        "frames_csv": str(frames_path),
        "accumulate_frames": int(accum_n),
        "every": int(max(1, args.every)),
        "max_frames": int(max(0, args.max_frames)),
        "note": "Sliding-window accumulation: frame k output contains points from frames (k-N+1 ... k) that were available.",
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_out = out_dir / "frames.csv"
    with csv_out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "t_sec", "points_in", "points_out", "pcd_path"])

        every = max(1, int(args.every))
        max_frames = int(args.max_frames)
        max_frames = max_frames if max_frames > 0 else 0

        schema: Optional[Tuple[List[str], List[int], List[str], List[int]]] = None
        window: Deque[np.ndarray] = deque()

        processed = 0
        for i, row in enumerate(frames):
            if i % every != 0:
                continue

            idx_str = (row.get("index") or row.get("frame") or row.get("idx") or "").strip()
            try:
                frame_idx = int(idx_str) if idx_str else int(i)
            except Exception:
                frame_idx = int(i)

            t_sec = (row.get("t_sec") or row.get("t") or "").strip()
            pcd_in = _resolve_pcd_path(in_dir, (row.get("pcd_path") or row.get("pcd") or "").strip())
            if pcd_in is None or not pcd_in.is_file():
                continue

            out_pcd = (out_pcd_dir / pcd_in.name).resolve()

            # Resume: if output exists, record it but still advance the window state.
            if args.resume and out_pcd.is_file():
                f, s, t, c, mat = _read_pcd_float_matrix(pcd_in)
                if schema is None:
                    schema = (f, s, t, c)
                else:
                    if (f, s, t, c) != schema:
                        raise RuntimeError(f"PCD schema mismatch: {pcd_in}")
                window.append(mat)
                while len(window) > accum_n:
                    window.popleft()

                mat_out = np.vstack(list(window)) if window else mat
                writer.writerow([int(frame_idx), t_sec, int(mat.shape[0]), int(mat_out.shape[0]), str(out_pcd)])
                processed += 1
                if processed % 200 == 0:
                    print(f"[OK] recorded {processed} existing frames")
                if max_frames > 0 and processed >= max_frames:
                    break
                continue

            f, s, t, c, mat = _read_pcd_float_matrix(pcd_in)
            if schema is None:
                schema = (f, s, t, c)
            else:
                if (f, s, t, c) != schema:
                    raise RuntimeError(f"PCD schema mismatch: {pcd_in}")

            window.append(mat)
            while len(window) > accum_n:
                window.popleft()

            mat_out = np.vstack(list(window)) if window else mat
            _write_pcd_float_matrix(out_pcd, fields=f, sizes=s, types=t, counts=c, mat=mat_out)

            writer.writerow([int(frame_idx), t_sec, int(mat.shape[0]), int(mat_out.shape[0]), str(out_pcd)])

            processed += 1
            if processed % 50 == 0:
                print(f"[OK] processed {processed} frames")
            if max_frames > 0 and processed >= max_frames:
                break

    print(f"[OK] Done. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

