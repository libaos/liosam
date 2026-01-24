#!/usr/bin/env python3
"""Render a BEV (top-down) plot from a PCD and overlay extracted tree centers.

This is the quick way to verify whether the tree-center extraction matches the PCD layout.
Typical inputs:
- PCD: `maps/map4_bin_tree_label0.pcd` (tree-only)
- Circles #1: `maps/map4_bin_tree_label0_circles.json`
- Circles #2 (optional): `maps/map4_bin_tree_label0_circles_validated_by_bag.json`
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _load_tree_circles_impl():
    impl_path = Path(__file__).resolve().parents[2] / "orchard_row_mapping" / "scripts" / "orchard_tree_circles_node.py"
    if not impl_path.is_file():
        raise RuntimeError(f"Cannot find implementation: {impl_path}")

    import importlib.util

    spec = importlib.util.spec_from_file_location("orchard_tree_circles_node", str(impl_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for: {impl_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_circles_xyr(circles_json: Path, default_radius: float = 0.15) -> np.ndarray:
    data = json.loads(circles_json.read_text(encoding="utf-8"))
    circles = data.get("circles", [])
    out = []
    for c in circles:
        if "x" not in c or "y" not in c:
            continue
        r = float(c.get("radius", default_radius))
        out.append((float(c["x"]), float(c["y"]), float(r)))
    if not out:
        raise RuntimeError(f"No circles found in: {circles_json}")
    return np.asarray(out, dtype=np.float32)


def _roi_from_xy(xy: np.ndarray, margin: float) -> Tuple[float, float, float, float]:
    xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    x_min = float(np.min(xy[:, 0])) - float(margin)
    x_max = float(np.max(xy[:, 0])) + float(margin)
    y_min = float(np.min(xy[:, 1])) - float(margin)
    y_max = float(np.max(xy[:, 1])) + float(margin)
    return x_min, x_max, y_min, y_max


def _roi_union(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    ax0, ax1, ay0, ay1 = a
    bx0, bx1, by0, by1 = b
    return min(ax0, bx0), max(ax1, bx1), min(ay0, by0), max(ay1, by1)


def main() -> int:
    ws_dir = Path(__file__).resolve().parents[3]
    default_pcd = ws_dir / "maps" / "map4_bin_tree_label0.pcd"
    default_circles = ws_dir / "maps" / "map4_bin_tree_label0_circles.json"
    default_circles2 = ws_dir / "maps" / "map4_bin_tree_label0_circles_validated_by_bag.json"
    default_out = ws_dir / "maps" / "runs" / "bev_map4_bin_tree_label0.png"

    parser = argparse.ArgumentParser(description="Render BEV from PCD and overlay tree centers")
    parser.add_argument("--pcd", type=str, default=str(default_pcd), help="输入 PCD（树点）")
    parser.add_argument("--circles", type=str, default=str(default_circles), help="circles json（中心点）")
    parser.add_argument("--circles2", type=str, default=str(default_circles2), help="可选第二份 circles json（空则不画）")
    parser.add_argument("--z-min", type=float, default=0.7, help="PCD z 过滤下界（用于看树干带）")
    parser.add_argument("--z-max", type=float, default=1.3, help="PCD z 过滤上界")
    parser.add_argument("--max-points", type=int, default=120000, help="PCD 最大绘制点数（0=不采样）")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--margin", type=float, default=5.0, help="视野边距（m）")
    parser.add_argument("--draw-radius", type=int, default=1, help="1=按 circles 的 radius 画圆（用于看粗细/覆盖范围）")
    parser.add_argument("--radius-scale", type=float, default=1.0, help="radius 整体缩放（仅绘图用）")
    parser.add_argument("--out", type=str, default=str(default_out), help="输出 PNG")

    args = parser.parse_args()

    pcd_path = Path(args.pcd).expanduser().resolve()
    circles_path = Path(args.circles).expanduser().resolve()
    circles2_path = Path(str(args.circles2)).expanduser().resolve() if str(args.circles2).strip() else None
    out_path = Path(args.out).expanduser().resolve()

    if not pcd_path.is_file():
        raise SystemExit(f"PCD not found: {pcd_path}")
    if not circles_path.is_file():
        raise SystemExit(f"circles json not found: {circles_path}")
    if circles2_path is not None and not circles2_path.is_file():
        circles2_path = None

    impl = _load_tree_circles_impl()

    points = impl._load_pcd_xyz(pcd_path).astype(np.float32)
    points = impl._filter_points(
        points,
        z_min=float(args.z_min),
        z_max=float(args.z_max),
        x_min=-1.0e9,
        x_max=1.0e9,
        y_abs_max=1.0e9,
    )
    points = impl._sample_points(points, int(args.max_points), seed=int(args.sample_seed))
    pts_xy = points[:, :2].astype(np.float32)

    circles1_xyr = _load_circles_xyr(circles_path)
    centers_xy = circles1_xyr[:, :2]
    roi = _roi_from_xy(centers_xy, margin=float(args.margin))

    centers2_xy: Optional[np.ndarray] = None
    circles2_xyr: Optional[np.ndarray] = None
    if circles2_path is not None:
        circles2_xyr = _load_circles_xyr(circles2_path)
        centers2_xy = circles2_xyr[:, :2]
        roi = _roi_union(roi, _roi_from_xy(centers2_xy, margin=float(args.margin)))

    x_min, x_max, y_min, y_max = roi
    mask = (pts_xy[:, 0] >= x_min) & (pts_xy[:, 0] <= x_max) & (pts_xy[:, 1] >= y_min) & (pts_xy[:, 1] <= y_max)
    pts_xy = pts_xy[mask]

    def _render_bev_cv2() -> None:
        try:
            import cv2  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"OpenCV (cv2) is required when matplotlib is unavailable: {exc}") from exc

        width = 1600
        height = 1120
        margin_px = 60

        dx = max(1.0e-6, float(x_max) - float(x_min))
        dy = max(1.0e-6, float(y_max) - float(y_min))
        scale = min((float(width - 2 * margin_px) / dx), (float(height - 2 * margin_px) / dy))

        def _xy_to_px(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            px = margin_px + (x.astype(np.float64, copy=False) - float(x_min)) * float(scale)
            py = margin_px + (float(y_max) - y.astype(np.float64, copy=False)) * float(scale)
            return px.astype(np.int32), py.astype(np.int32)

        img = np.full((int(height), int(width), 3), 255, dtype=np.uint8)

        plot_x0 = int(margin_px)
        plot_y0 = int(margin_px)
        plot_x1 = int(width) - int(margin_px)
        plot_y1 = int(height) - int(margin_px)

        def _choose_grid_step(span: float) -> float:
            span = float(max(span, 1.0e-6))
            target = span / 8.0
            candidates = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
            return min(candidates, key=lambda s: abs(float(s) - float(target)))

        grid_step_x = _choose_grid_step(float(x_max) - float(x_min))
        grid_step_y = _choose_grid_step(float(y_max) - float(y_min))
        grid_color = (230, 230, 230)
        axis_color = (80, 80, 80)

        # Grid lines + tick labels.
        def _frange_start(vmin: float, step: float) -> float:
            return math.ceil(vmin / step) * step

        # Vertical grid (x)
        x0 = _frange_start(float(x_min), grid_step_x)
        x_vals = np.arange(x0, float(x_max) + 0.5 * grid_step_x, grid_step_x, dtype=np.float64)
        for xv in x_vals.tolist():
            px, _ = _xy_to_px(np.asarray([xv], dtype=np.float64), np.asarray([y_min], dtype=np.float64))
            xpx = int(px[0])
            if plot_x0 <= xpx <= plot_x1:
                cv2.line(img, (xpx, plot_y0), (xpx, plot_y1), grid_color, 1, lineType=cv2.LINE_AA)
                cv2.putText(
                    img,
                    f"{xv:g}",
                    (xpx - 12, plot_y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    axis_color,
                    1,
                    cv2.LINE_AA,
                )

        # Horizontal grid (y)
        y0 = _frange_start(float(y_min), grid_step_y)
        y_vals = np.arange(y0, float(y_max) + 0.5 * grid_step_y, grid_step_y, dtype=np.float64)
        for yv in y_vals.tolist():
            _, py = _xy_to_px(np.asarray([x_min], dtype=np.float64), np.asarray([yv], dtype=np.float64))
            ypx = int(py[0])
            if plot_y0 <= ypx <= plot_y1:
                cv2.line(img, (plot_x0, ypx), (plot_x1, ypx), grid_color, 1, lineType=cv2.LINE_AA)
                cv2.putText(
                    img,
                    f"{yv:g}",
                    (plot_x0 - 45, ypx + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    axis_color,
                    1,
                    cv2.LINE_AA,
                )

        # Border box.
        cv2.rectangle(img, (plot_x0, plot_y0), (plot_x1, plot_y1), axis_color, 1, lineType=cv2.LINE_AA)

        if pts_xy.size:
            px, py = _xy_to_px(pts_xy[:, 0], pts_xy[:, 1])
            inside = (px >= 0) & (px < int(width)) & (py >= 0) & (py < int(height))
            px = px[inside]
            py = py[inside]
            img[py, px] = (180, 180, 180)

        # Approximate matplotlib tab colors (BGR for OpenCV).
        green = (44, 160, 44)  # tab:green -> RGB(44,160,44)
        blue = (180, 119, 31)  # tab:blue  -> RGB(31,119,180)

        scale_r = float(args.radius_scale)
        for x, y, r in circles1_xyr.tolist():
            cx, cy = _xy_to_px(np.asarray([x], dtype=np.float32), np.asarray([y], dtype=np.float32))
            cx_i = int(cx[0])
            cy_i = int(cy[0])
            if not (0 <= cx_i < int(width) and 0 <= cy_i < int(height)):
                continue
            cv2.circle(img, (cx_i, cy_i), 6, green, 2, lineType=cv2.LINE_AA)
            if bool(int(args.draw_radius)):
                rr = max(float(r) * scale_r, 1.0e-3)
                rp = int(round(rr * float(scale)))
                if rp > 0:
                    cv2.circle(img, (cx_i, cy_i), rp, green, 2, lineType=cv2.LINE_AA)

        if circles2_xyr is not None:
            for x, y, r in circles2_xyr.tolist():
                cx, cy = _xy_to_px(np.asarray([x], dtype=np.float32), np.asarray([y], dtype=np.float32))
                cx_i = int(cx[0])
                cy_i = int(cy[0])
                if not (0 <= cx_i < int(width) and 0 <= cy_i < int(height)):
                    continue
                cv2.circle(img, (cx_i, cy_i), 6, blue, 2, lineType=cv2.LINE_AA)
                if bool(int(args.draw_radius)):
                    rr = max(float(r) * scale_r, 1.0e-3)
                    rp = int(round(rr * float(scale)))
                    if rp > 0:
                        cv2.circle(img, (cx_i, cy_i), rp, blue, 2, lineType=cv2.LINE_AA)

        # Title + quick legend
        title = f"BEV from PCD ({pcd_path.name})"
        cv2.putText(img, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 10, 10), 2, cv2.LINE_AA)
        cv2.putText(
            img,
            f"pcd points (z={args.z_min:.1f}..{args.z_max:.1f})",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (120, 120, 120),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"centers #1 ({circles_path.name})",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            green,
            2,
            cv2.LINE_AA,
        )
        if circles2_path is not None and circles2_xyr is not None:
            cv2.putText(
                img,
                f"centers #2 ({circles2_path.name})",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                blue,
                2,
                cv2.LINE_AA,
            )

        # Axis labels
        cv2.putText(img, "x [m]", (plot_x1 - 80, plot_y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, axis_color, 2, cv2.LINE_AA)
        cv2.putText(img, "y [m]", (plot_x0 - 55, plot_y0 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, axis_color, 2, cv2.LINE_AA)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)

    try:
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.patches import Circle  # type: ignore

        use_matplotlib = True
    except Exception as exc:
        print(f"[WARN] matplotlib not available ({exc}); using OpenCV renderer.")
        use_matplotlib = False

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not use_matplotlib:
        _render_bev_cv2()
        print(f"[OK] wrote: {out_path}")
        print(f"[OK] pcd pts (ROI): {int(pts_xy.shape[0])}")
        print(f"[OK] centers #1: {int(centers_xy.shape[0])}")
        if centers2_xy is not None:
            print(f"[OK] centers #2: {int(centers2_xy.shape[0])}")
        return 0

    fig, ax = plt.subplots(figsize=(10, 7), dpi=160)
    if pts_xy.size:
        ax.scatter(
            pts_xy[:, 0],
            pts_xy[:, 1],
            s=2,
            c="0.7",
            alpha=0.6,
            linewidths=0,
            label=f"pcd points (z={args.z_min:.1f}..{args.z_max:.1f})",
        )

    ax.scatter(
        centers_xy[:, 0],
        centers_xy[:, 1],
        s=30,
        facecolors="none",
        edgecolors="tab:green",
        linewidths=1.2,
        label=f"centers #1 ({circles_path.name})",
    )

    if bool(int(args.draw_radius)):
        scale = float(args.radius_scale)
        for x, y, r in circles1_xyr.tolist():
            rr = max(float(r) * scale, 1.0e-3)
            ax.add_patch(Circle((float(x), float(y)), rr, fill=False, ec="tab:green", lw=0.8, alpha=0.9))

    if centers2_xy is not None and centers2_xy.size:
        ax.scatter(
            centers2_xy[:, 0],
            centers2_xy[:, 1],
            s=28,
            facecolors="none",
            edgecolors="tab:blue",
            linewidths=1.2,
            label=f"centers #2 ({circles2_path.name})",
        )
        if bool(int(args.draw_radius)) and circles2_xyr is not None:
            scale = float(args.radius_scale)
            for x, y, r in circles2_xyr.tolist():
                rr = max(float(r) * scale, 1.0e-3)
                ax.add_patch(Circle((float(x), float(y)), rr, fill=False, ec="tab:blue", lw=0.8, alpha=0.9))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    ax.set_title(f"BEV from PCD ({pcd_path.name})")

    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)

    print(f"[OK] wrote: {out_path}")
    print(f"[OK] pcd pts (ROI): {int(pts_xy.shape[0])}")
    print(f"[OK] centers #1: {int(centers_xy.shape[0])}")
    if centers2_xy is not None:
        print(f"[OK] centers #2: {int(centers2_xy.shape[0])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
