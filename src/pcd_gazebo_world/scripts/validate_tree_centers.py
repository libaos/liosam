#!/usr/bin/env python3
"""Cross-validate (and optionally align) tree centers between PCD and rosbag.

Typical workflow:
1) From PCD: `pcd_to_orchard_world.py` -> maps/*_circles.json
2) From rosbag (same run): `rosbag_registered_cloud_to_orchard_world.py` -> rosbags/runs/*_circles.json
3) Validate: keep only PCD centers that are supported by the rosbag centers.

This is useful when the PCD-based clustering yields extra/misaligned trees, or when you want
to focus the Gazebo world on trees that are actually observed along the trajectory.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _load_circles(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    circles = data.get("circles", None)
    if not isinstance(circles, list) or not circles:
        raise RuntimeError(f"No circles found in: {path}")
    out: List[Dict[str, Any]] = []
    for c in circles:
        if not isinstance(c, dict):
            continue
        if "x" not in c or "y" not in c:
            continue
        out.append(dict(c))
    if not out:
        raise RuntimeError(f"No usable circles found in: {path}")
    return out


def _centers_xy(circles: Sequence[Dict[str, Any]]) -> np.ndarray:
    return np.asarray([(float(c["x"]), float(c["y"])) for c in circles], dtype=np.float64).reshape(-1, 2)


def _kabsch_2d(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Solve rigid transform Q ~= R*P + t (2D)."""
    P = np.asarray(P, dtype=np.float64).reshape(-1, 2)
    Q = np.asarray(Q, dtype=np.float64).reshape(-1, 2)
    if P.shape[0] < 3 or Q.shape[0] < 3:
        return np.eye(2, dtype=np.float64), np.zeros((2,), dtype=np.float64)
    Pc = P - np.mean(P, axis=0, keepdims=True)
    Qc = Q - np.mean(Q, axis=0, keepdims=True)
    H = Pc.T @ Qc
    U, _S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if float(np.linalg.det(R)) < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = np.mean(Q, axis=0) - (R @ np.mean(P, axis=0))
    return R.astype(np.float64), t.astype(np.float64)


def _apply_RT(xy: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    return (xy @ R.T) + t.reshape(1, 2)


def _nearest_neighbor_indices(src_xy: np.ndarray, dst_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """For each src point, find nearest dst index + distance."""
    src_xy = np.asarray(src_xy, dtype=np.float64).reshape(-1, 2)
    dst_xy = np.asarray(dst_xy, dtype=np.float64).reshape(-1, 2)
    d2 = np.sum((src_xy[:, None, :] - dst_xy[None, :, :]) ** 2, axis=2)
    idx = np.argmin(d2, axis=1).astype(np.int64)
    dist = np.sqrt(d2[np.arange(src_xy.shape[0]), idx])
    return idx, dist


def _fmt(x: float) -> str:
    return f"{float(x):.6f}"


def _write_world(
    out_path: Path,
    world_name: str,
    tree_model_uri: str,
    tree_poses_xy: Sequence[Tuple[float, float]],
    ground_mode: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree_poses_xy = list(tree_poses_xy)

    lines: List[str] = []
    lines.append('<?xml version="1.0"?>')
    lines.append('<sdf version="1.6">')
    lines.append(f'  <world name="{world_name}">')
    lines.append("    <include>")
    lines.append("      <uri>model://sun</uri>")
    lines.append("    </include>")
    lines.append("")

    ground_mode = (ground_mode or "plane").strip().lower()
    if ground_mode == "terrain":
        lines.append("    <include>")
        lines.append("      <uri>model://pcd_terrain</uri>")
        lines.append("      <pose>0 0 0 0 0 0</pose>")
        lines.append("    </include>")
    else:
        lines.append("    <include>")
        lines.append("      <uri>model://ground_plane</uri>")
        lines.append("    </include>")

    lines.append("")
    lines.append(f"    <!-- Trees: {len(tree_poses_xy)} includes of {tree_model_uri} -->")

    for i, (x, y) in enumerate(tree_poses_xy):
        lines.append("    <include>")
        lines.append(f"      <uri>{tree_model_uri}</uri>")
        lines.append(f"      <name>tree_{i:04d}</name>")
        lines.append(f"      <pose>{_fmt(x)} {_fmt(y)} 0 0 0 0</pose>")
        lines.append("    </include>")

    lines.append("  </world>")
    lines.append("</sdf>")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ws_dir = Path(__file__).resolve().parents[3]
    default_circles = ws_dir / "maps" / "map4_bin_tree_label0_circles.json"
    default_validator = ws_dir / "rosbags" / "runs" / "tree_centers_from_cloud_registered_full_0p7_1p3.json"
    default_out_circles = ws_dir / "maps" / "map4_bin_tree_label0_circles_validated.json"
    default_out_world = ws_dir / "src" / "pcd_gazebo_world" / "worlds" / "orchard_from_pcd_validated.world"

    parser = argparse.ArgumentParser(description="PCD vs rosbag tree centers cross-validation")
    parser.add_argument("--circles", type=str, default=str(default_circles), help="输入 circles json（通常来自 PCD）")
    parser.add_argument("--validator", type=str, default=str(default_validator), help="用于认证的 circles json（通常来自 rosbag）")
    parser.add_argument("--threshold", type=float, default=1.0, help="通过认证的最近邻阈值（m）")
    parser.add_argument("--align", type=int, default=0, help="1=先对 validator 做 2D 刚体对齐（robust NN+Kabsch）")
    parser.add_argument("--align-gate", type=float, default=2.0, help="对齐时的最近邻 gating 阈值（m）")

    parser.add_argument("--out-circles", type=str, default=str(default_out_circles), help="输出验证后的 circles json")
    parser.add_argument("--out-world", type=str, default=str(default_out_world), help="输出 Gazebo .world（空则不输出）")
    parser.add_argument("--world-name", type=str, default="orchard_world")
    parser.add_argument("--ground", choices=["plane", "terrain"], default="plane")
    parser.add_argument("--tree-model-uri", type=str, default="model://tree_trunk")

    args = parser.parse_args()

    circles_path = Path(args.circles).expanduser().resolve()
    validator_path = Path(args.validator).expanduser().resolve()
    out_circles = Path(args.out_circles).expanduser().resolve()
    out_world = Path(args.out_world).expanduser().resolve() if str(args.out_world).strip() else None

    if not circles_path.is_file():
        raise SystemExit(f"circles not found: {circles_path}")
    if not validator_path.is_file():
        raise SystemExit(f"validator not found: {validator_path}")

    circles = _load_circles(circles_path)
    validator = _load_circles(validator_path)
    A = _centers_xy(circles)  # source (PCD)
    B = _centers_xy(validator)  # validator (rosbag)

    R = np.eye(2, dtype=np.float64)
    t = np.zeros((2,), dtype=np.float64)
    used_align = False
    if bool(int(args.align)):
        nn_idx, nn_dist = _nearest_neighbor_indices(B, A)
        mask = nn_dist < float(args.align_gate)
        if int(np.sum(mask)) >= 3:
            R, t = _kabsch_2d(B[mask], A[nn_idx[mask]])
            B = _apply_RT(B, R, t)
            used_align = True

    _, dist = _nearest_neighbor_indices(A, B)
    keep = dist < float(args.threshold)
    circles_out = [circles[i] for i in np.flatnonzero(keep).tolist()]

    out_circles.parent.mkdir(parents=True, exist_ok=True)
    out_circles.write_text(
        json.dumps(
            {
                "mode": "validated_by_nn",
                "circles": [c if isinstance(c, dict) else asdict(c) for c in circles_out],
                "source_circles": str(circles_path),
                "validator_circles": str(validator_path),
                "threshold_m": float(args.threshold),
                "align": bool(int(args.align)),
                "align_used": bool(used_align),
                "align_gate_m": float(args.align_gate),
                "validator_R_rowmajor": [float(x) for x in R.reshape(-1).tolist()],
                "validator_t_xy": [float(x) for x in t.reshape(-1).tolist()],
                "stats": {
                    "source_total": int(A.shape[0]),
                    "validator_total": int(B.shape[0]),
                    "kept": int(len(circles_out)),
                    "nn_mean_m": float(np.mean(dist)) if dist.size else float("nan"),
                    "nn_median_m": float(np.median(dist)) if dist.size else float("nan"),
                    "nn_max_m": float(np.max(dist)) if dist.size else float("nan"),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    if out_world is not None:
        poses_xy = [(float(c["x"]), float(c["y"])) for c in circles_out]
        _write_world(
            out_path=out_world,
            world_name=str(args.world_name),
            tree_model_uri=str(args.tree_model_uri),
            tree_poses_xy=poses_xy,
            ground_mode=str(args.ground),
        )

    print(f"[OK] source circles:    {int(A.shape[0])} ({circles_path.name})")
    print(f"[OK] validator circles: {int(B.shape[0])} ({validator_path.name})")
    print(f"[OK] kept:             {int(len(circles_out))} / {int(A.shape[0])}  (thr={float(args.threshold):.2f} m)")
    if used_align:
        print(f"[OK] align (validator->source) R={R.tolist()} t={t.tolist()}")
    print(f"[OK] wrote: {out_circles}")
    if out_world is not None:
        print(f"[OK] wrote: {out_world}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

