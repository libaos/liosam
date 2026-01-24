#!/usr/bin/env python3
"""Compare multiple BEV export runs and generate montage PNGs for frames with large differences.

Input: one or more BEV run directories produced by `clustered_frames_to_circles_bev.py`:
  <run>/circles/circles_000123.json
  <run>/png/bev_000123.png

We compute per-frame difference metrics across runs:
  - clusters_range: max(clusters) - min(clusters)
  - geo_max: maximum pairwise symmetric NN-mean distance between circle centers

Then we select top-K frames and write:
  - stats_all_frames.csv
  - top_frames.csv
  - montage PNGs + per-frame symlinked originals
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Run:
    name: str
    run_dir: Path


def _safe_float(v: object, default: float = float("nan")) -> float:
    try:
        x = float(v)  # type: ignore[arg-type]
    except Exception:
        return float(default)
    return x if math.isfinite(x) else float(default)


def _load_circle_centers(path: Path) -> np.ndarray:
    obj = json.loads(path.read_text(encoding="utf-8"))
    circles = obj.get("circles") or []
    pts: List[Tuple[float, float]] = []
    for c in circles:
        x = _safe_float((c or {}).get("x"), float("nan"))
        y = _safe_float((c or {}).get("y"), float("nan"))
        if math.isfinite(x) and math.isfinite(y):
            pts.append((float(x), float(y)))
    if not pts:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32).reshape((-1, 2))


def _sym_nn_mean_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape((-1, 2))
    b = np.asarray(b, dtype=np.float32).reshape((-1, 2))
    if a.size == 0 and b.size == 0:
        return 0.0
    if a.size == 0 or b.size == 0:
        return float("inf")
    d = a[:, None, :] - b[None, :, :]
    dist = np.sqrt(np.sum(d.astype(np.float64) ** 2, axis=2))
    a_to_b = float(np.mean(np.min(dist, axis=1)))
    b_to_a = float(np.mean(np.min(dist, axis=0)))
    return 0.5 * (a_to_b + b_to_a)


def _symlink_force(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    rel = os.path.relpath(src.resolve(), start=dst.parent.resolve())
    dst.symlink_to(rel)


def _iter_frame_indices(run: Run) -> List[int]:
    circles_dir = run.run_dir / "circles"
    if not circles_dir.is_dir():
        return []
    out: List[int] = []
    for p in sorted(circles_dir.glob("circles_*.json")):
        stem = p.stem
        try:
            idx = int(stem.split("_")[-1])
        except Exception:
            continue
        out.append(int(idx))
    return sorted(set(out))


def _read_png(path: Path) -> np.ndarray:
    import cv2  # type: ignore

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read PNG: {path}")
    return img


def _resize_like(img: np.ndarray, *, width: int, height: int) -> np.ndarray:
    import cv2  # type: ignore

    h, w = img.shape[:2]
    if int(w) == int(width) and int(h) == int(height):
        return img
    return cv2.resize(img, (int(width), int(height)), interpolation=cv2.INTER_AREA)


def _put_text(img: np.ndarray, text: str, *, x: int, y: int) -> None:
    import cv2  # type: ignore

    cv2.putText(
        img,
        text,
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (20, 20, 20),
        2,
        lineType=cv2.LINE_AA,
    )


def _make_montage_2x2(
    imgs: Sequence[np.ndarray],
    labels: Sequence[str],
    *,
    header: str,
) -> np.ndarray:
    import cv2  # type: ignore

    if len(imgs) != 4 or len(labels) != 4:
        raise ValueError("This helper expects exactly 4 images/labels (2x2 montage).")

    h0, w0 = imgs[0].shape[:2]
    resized = [_resize_like(im, width=w0, height=h0) for im in imgs]
    for im, label in zip(resized, labels):
        _put_text(im, label, x=20, y=40)

    top = cv2.hconcat([resized[0], resized[1]])
    bot = cv2.hconcat([resized[2], resized[3]])
    grid = cv2.vconcat([top, bot])

    header_h = 70
    canvas = np.full((int(header_h) + grid.shape[0], grid.shape[1], 3), 255, dtype=np.uint8)
    canvas[header_h : header_h + grid.shape[0], :, :] = grid
    _put_text(canvas, header, x=20, y=45)
    return canvas


def _make_montage_1x2(
    imgs: Sequence[np.ndarray],
    labels: Sequence[str],
    *,
    header: str,
) -> np.ndarray:
    import cv2  # type: ignore

    if len(imgs) != 2 or len(labels) != 2:
        raise ValueError("This helper expects exactly 2 images/labels (1x2 montage).")

    h0, w0 = imgs[0].shape[:2]
    resized = [_resize_like(im, width=w0, height=h0) for im in imgs]
    for im, label in zip(resized, labels):
        _put_text(im, label, x=20, y=40)

    grid = cv2.hconcat([resized[0], resized[1]])

    header_h = 70
    canvas = np.full((int(header_h) + grid.shape[0], grid.shape[1], 3), 255, dtype=np.uint8)
    canvas[header_h : header_h + grid.shape[0], :, :] = grid
    _put_text(canvas, header, x=20, y=45)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="BEV run directory (repeatable).")
    parser.add_argument("--name", action="append", default=[], help="Display name (repeatable; must match --run count if set).")
    parser.add_argument("--out-dir", required=True, help="Output directory (Chinese path OK).")
    parser.add_argument("--top-k", type=int, default=60, help="How many frames to export montages for.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap on how many common frames to scan (0=all).")
    args = parser.parse_args()

    run_dirs = [Path(s).expanduser().resolve() for s in (args.run or [])]
    if not run_dirs:
        raise RuntimeError("--run is required")

    names = [str(s).strip() for s in (args.name or []) if str(s).strip()]
    if names and len(names) != len(run_dirs):
        raise ValueError("--name count must match --run count (or omit --name)")
    if not names:
        names = [p.name for p in run_dirs]

    runs = [Run(name=n, run_dir=d) for n, d in zip(names, run_dirs)]
    if len(runs) not in (2, 4):
        raise ValueError("This script currently supports 2 runs (1x2 montage) or 4 runs (2x2 montage).")

    for r in runs:
        if not (r.run_dir / "circles").is_dir() or not (r.run_dir / "png").is_dir():
            raise FileNotFoundError(f"Expected circles/png under: {r.run_dir}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    montage_dir = out_dir / "montage"
    selected_dir = out_dir / "selected"
    montage_dir.mkdir(parents=True, exist_ok=True)
    selected_dir.mkdir(parents=True, exist_ok=True)

    # Common frame indices across runs
    common: Optional[set[int]] = None
    for r in runs:
        idxs = set(_iter_frame_indices(r))
        common = idxs if common is None else (common & idxs)
    indices = sorted(common or [])
    if not indices:
        raise RuntimeError("No common frames found across runs.")

    if int(args.max_frames) > 0:
        indices = indices[: int(args.max_frames)]

    # Compute stats
    rows: List[Dict[str, object]] = []
    for idx in indices:
        centers_by_run: Dict[str, np.ndarray] = {}
        clusters_by_run: Dict[str, int] = {}
        for r in runs:
            circles_path = r.run_dir / "circles" / f"circles_{idx:06d}.json"
            centers = _load_circle_centers(circles_path)
            centers_by_run[r.name] = centers
            clusters_by_run[r.name] = int(centers.shape[0])

        clusters = list(clusters_by_run.values())
        clusters_range = int(max(clusters) - min(clusters)) if clusters else 0

        # Pairwise geo distances
        pairwise: Dict[Tuple[str, str], float] = {}
        geo_vals: List[float] = []
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                a = runs[i].name
                b = runs[j].name
                d = _sym_nn_mean_distance(centers_by_run[a], centers_by_run[b])
                pairwise[(a, b)] = float(d)
                if math.isfinite(float(d)):
                    geo_vals.append(float(d))
        geo_max = float(max(geo_vals)) if geo_vals else float("nan")
        geo_mean = float(sum(geo_vals) / float(len(geo_vals))) if geo_vals else float("nan")

        row: Dict[str, object] = {
            "index": int(idx),
            "clusters_range": int(clusters_range),
            "geo_max": float(geo_max),
            "geo_mean": float(geo_mean),
        }
        for r in runs:
            row[f"clusters_{r.name}"] = int(clusters_by_run[r.name])
        for (a, b), d in pairwise.items():
            row[f"geo_{a}__{b}"] = float(d)
        rows.append(row)

    # Sort: prioritize cluster-count disagreements, then geometry
    def sort_key(r: Dict[str, object]) -> Tuple[int, float, int]:
        cr = int(r.get("clusters_range", 0) or 0)
        gm = float(r.get("geo_max", float("nan")))
        gm_key = gm if math.isfinite(gm) else float("inf")
        return (-cr, -gm_key, int(r.get("index", 0) or 0))

    rows_sorted = sorted(rows, key=sort_key)
    top_k = max(0, int(args.top_k))
    selected = rows_sorted[:top_k] if top_k > 0 else []

    # Write CSVs
    all_csv = out_dir / "stats_all_frames.csv"
    top_csv = out_dir / "top_frames.csv"
    fieldnames = sorted({k for row in rows_sorted for k in row.keys()}, key=str)
    with all_csv.open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fieldnames)
        w.writeheader()
        for row in rows_sorted:
            w.writerow(row)
    with top_csv.open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fieldnames)
        w.writeheader()
        for row in selected:
            w.writerow(row)

    # Summaries
    ranges = [int(r.get("clusters_range", 0) or 0) for r in rows_sorted]
    geo_max_vals = [float(r.get("geo_max", float("nan"))) for r in rows_sorted]
    geo_max_finite = [v for v in geo_max_vals if math.isfinite(v)]
    summary = {
        "runs": [{"name": r.name, "run_dir": str(r.run_dir)} for r in runs],
        "frames_common": int(len(rows_sorted)),
        "clusters_range": {
            "max": int(max(ranges) if ranges else 0),
            "mean": float(statistics.mean(ranges) if ranges else 0.0),
            "median": float(statistics.median(ranges) if ranges else 0.0),
        },
        "geo_max": {
            "max": float(max(geo_max_finite) if geo_max_finite else float("nan")),
            "mean": float(statistics.mean(geo_max_finite) if geo_max_finite else float("nan")),
            "median": float(statistics.median(geo_max_finite) if geo_max_finite else float("nan")),
        },
        "top_k": int(top_k),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Generate montage PNGs for selected frames
    for rank, row in enumerate(selected, start=1):
        idx = int(row.get("index", 0) or 0)
        clusters_range = int(row.get("clusters_range", 0) or 0)
        geo_max = float(row.get("geo_max", float("nan")))

        imgs: List[np.ndarray] = []
        labels: List[str] = []
        for r in runs:
            png_path = r.run_dir / "png" / f"bev_{idx:06d}.png"
            imgs.append(_read_png(png_path))
            labels.append(f"{r.name}  n={int(row.get(f'clusters_{r.name}', 0) or 0)}")

        header = f"rank {rank:03d}  idx {idx:06d}  clusters_range={clusters_range}  geo_max={geo_max:.2f}m"
        if len(runs) == 2:
            montage = _make_montage_1x2(imgs, labels, header=header)
        else:
            montage = _make_montage_2x2(imgs, labels, header=header)

        out_png = montage_dir / f"rank_{rank:03d}_idx_{idx:06d}.png"
        import cv2  # type: ignore

        cv2.imwrite(str(out_png), montage)

        # Per-frame folder with symlinks to originals + montage
        frame_dir = selected_dir / f"rank_{rank:03d}_idx_{idx:06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        _symlink_force(out_png, frame_dir / "00_对比拼图.png")
        for k, r in enumerate(runs, start=1):
            src = r.run_dir / "png" / f"bev_{idx:06d}.png"
            safe = str(r.name).replace("/", "_")
            _symlink_force(src, frame_dir / f"{k:02d}_{safe}.png")

    print(f"[OK] Wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
