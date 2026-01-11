#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RowLine:
    v_center: float
    u_min: float
    u_max: float
    z: float


def _unit2(vec: Tuple[float, float]) -> Tuple[float, float]:
    norm = math.hypot(vec[0], vec[1])
    if norm < 1.0e-9:
        return (1.0, 0.0)
    return (vec[0] / norm, vec[1] / norm)


def _rotate90(vec: Tuple[float, float]) -> Tuple[float, float]:
    return (-vec[1], vec[0])


def _parse_indices(spec: Optional[str], max_count: int) -> Optional[List[int]]:
    if not spec:
        return None
    indices: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a_str, b_str = part.split("-", 1)
            a = int(a_str)
            b = int(b_str)
            if a <= b:
                indices.extend(range(a, b + 1))
            else:
                indices.extend(range(a, b - 1, -1))
        else:
            indices.append(int(part))
    indices = sorted(set(i for i in indices if 0 <= i < max_count))
    return indices


def _load_row_model(model_path: Path) -> Tuple[Tuple[float, float], Tuple[float, float], List[RowLine]]:
    raw = json.loads(model_path.read_text(encoding="utf-8"))
    direction_xy = _unit2((float(raw["direction_xy"][0]), float(raw["direction_xy"][1])))
    perp_xy = _rotate90(direction_xy)
    rows = [
        RowLine(
            v_center=float(r["v_center"]),
            u_min=float(r["u_min"]),
            u_max=float(r["u_max"]),
            z=float(r.get("z", 0.0)),
        )
        for r in raw["rows"]
    ]
    rows.sort(key=lambda r: r.v_center)
    return direction_xy, perp_xy, rows


def _row_endpoints(
    direction_xy: Tuple[float, float],
    perp_xy: Tuple[float, float],
    row: RowLine,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    x0 = direction_xy[0] * row.u_min + perp_xy[0] * row.v_center
    y0 = direction_xy[1] * row.u_min + perp_xy[1] * row.v_center
    x1 = direction_xy[0] * row.u_max + perp_xy[0] * row.v_center
    y1 = direction_xy[1] * row.u_max + perp_xy[1] * row.v_center
    return (x0, y0, row.z), (x1, y1, row.z)


def _centerline_from_pair(left: RowLine, right: RowLine) -> Optional[RowLine]:
    u_min = max(left.u_min, right.u_min)
    u_max = min(left.u_max, right.u_max)
    if u_max <= u_min:
        return None
    return RowLine(
        v_center=0.5 * (left.v_center + right.v_center),
        u_min=u_min,
        u_max=u_max,
        z=0.5 * (left.z + right.z),
    )


def _write_csv(
    out_path: Path,
    rows: Iterable[Tuple[int, str, RowLine, Tuple[float, float, float], Tuple[float, float, float]]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "kind",
                "x0",
                "y0",
                "z0",
                "x1",
                "y1",
                "z1",
                "u_min",
                "u_max",
                "v_center",
            ]
        )
        for idx, kind, row, p0, p1 in rows:
            writer.writerow(
                [
                    idx,
                    kind,
                    f"{p0[0]:.6f}",
                    f"{p0[1]:.6f}",
                    f"{p0[2]:.6f}",
                    f"{p1[0]:.6f}",
                    f"{p1[1]:.6f}",
                    f"{p1[2]:.6f}",
                    f"{row.u_min:.6f}",
                    f"{row.u_max:.6f}",
                    f"{row.v_center:.6f}",
                ]
            )


def _write_json(
    out_path: Path,
    rows: Iterable[Tuple[int, str, RowLine, Tuple[float, float, float], Tuple[float, float, float]]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for idx, kind, row, p0, p1 in rows:
        payload.append(
            {
                "index": int(idx),
                "kind": str(kind),
                "p0": [float(p0[0]), float(p0[1]), float(p0[2])],
                "p1": [float(p1[0]), float(p1[1]), float(p1[2])],
                "u_min": float(row.u_min),
                "u_max": float(row.u_max),
                "v_center": float(row.v_center),
            }
        )
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sample_line_points(
    p0: Tuple[float, float, float],
    p1: Tuple[float, float, float],
    step_m: float,
) -> List[Tuple[float, float, float]]:
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    dz = p1[2] - p0[2]
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length < 1.0e-9:
        return [p0]
    step_m = max(1.0e-4, float(step_m))
    count = max(2, int(math.ceil(length / step_m)) + 1)
    pts = []
    for i in range(count):
        t = float(i) / float(count - 1)
        pts.append((p0[0] + dx * t, p0[1] + dy * t, p0[2] + dz * t))
    return pts


def _write_pcd(
    out_path: Path,
    rows: Iterable[Tuple[int, str, RowLine, Tuple[float, float, float], Tuple[float, float, float]]],
    step_m: float,
) -> None:
    points: List[Tuple[float, float, float, float]] = []
    for idx, kind, _row, p0, p1 in rows:
        kind_offset = 0.0 if kind == "row" else 1000.0
        intensity = float(idx) + kind_offset
        for x, y, z in _sample_line_points(p0, p1, step_m=step_m):
            points.append((float(x), float(y), float(z), intensity))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS x y z intensity",
        "SIZE 4 4 4 4",
        "TYPE F F F F",
        "COUNT 1 1 1 1",
        f"WIDTH {len(points)}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {len(points)}",
        "DATA ascii",
    ]
    lines = ["\n".join(header)]
    for x, y, z, intensity in points:
        lines.append(f"{x:.6f} {y:.6f} {z:.6f} {intensity:.1f}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export orchard row model lines (tree-row boundaries or lane centerlines) to a CSV file.\n\n"
            "Examples:\n"
            "  tools/export_row_model_lines.py maps/row_model_pca_major.json --mode rows --out maps/rows.csv\n"
            "  tools/export_row_model_lines.py maps/row_model_pca_major.json --mode centerlines --out maps/center.csv\n"
            "  tools/export_row_model_lines.py maps/row_model_pca_major.json --mode rows --indices 0-4 --out maps/sub.csv\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "model",
        help="Input row model json (from row_model_file), e.g. maps/row_model_pca_major.json",
    )
    parser.add_argument(
        "--mode",
        choices=("rows", "centerlines", "both"),
        default="rows",
        help="What to export: tree-row boundaries (rows) or lane centerlines (centerlines). Default: rows",
    )
    parser.add_argument(
        "--out",
        default="maps/exported_lines.csv",
        help="Output file path (default: maps/exported_lines.csv)",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "json", "pcd"),
        default="csv",
        help="Output format (default: csv). PCD exports sampled points with intensity=line id.",
    )
    parser.add_argument(
        "--pcd-step",
        type=float,
        default=0.2,
        help="PCD sampling step in meters (default: 0.2). Only used when --format pcd.",
    )
    parser.add_argument(
        "--indices",
        default="",
        help="Optional indices filter, e.g. '0-4,7,9' (interpreted per mode).",
    )

    args = parser.parse_args(argv)
    model_path = Path(args.model).expanduser()
    out_path = Path(args.out).expanduser()

    if not model_path.is_file():
        raise FileNotFoundError(f"Row model not found: {model_path}")

    direction_xy, perp_xy, rows = _load_row_model(model_path)
    out_rows: List[Tuple[int, str, RowLine, Tuple[float, float, float], Tuple[float, float, float]]] = []

    if args.mode in ("rows", "both"):
        filter_idx = _parse_indices(args.indices, len(rows)) if args.indices else None
        for i, row in enumerate(rows):
            if filter_idx is not None and i not in filter_idx:
                continue
            p0, p1 = _row_endpoints(direction_xy, perp_xy, row)
            out_rows.append((i, "row", row, p0, p1))

    if args.mode in ("centerlines", "both") and len(rows) >= 2:
        centers: List[Tuple[int, RowLine]] = []
        for lane_idx, (left, right) in enumerate(zip(rows[:-1], rows[1:])):
            center = _centerline_from_pair(left, right)
            if center is None:
                continue
            centers.append((lane_idx, center))
        filter_idx = _parse_indices(args.indices, len(centers)) if args.indices else None
        for lane_idx, center in centers:
            if filter_idx is not None and lane_idx not in filter_idx:
                continue
            p0, p1 = _row_endpoints(direction_xy, perp_xy, center)
            out_rows.append((lane_idx, "centerline", center, p0, p1))

    if args.format == "csv":
        _write_csv(out_path, out_rows)
    elif args.format == "json":
        _write_json(out_path, out_rows)
    else:
        _write_pcd(out_path, out_rows, step_m=float(args.pcd_step))
    print(
        f"[OK] Exported {len(out_rows)} line(s) to: {os.path.abspath(str(out_path))}\n"
        f"     Model: {os.path.abspath(str(model_path))} (rows={len(rows)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
