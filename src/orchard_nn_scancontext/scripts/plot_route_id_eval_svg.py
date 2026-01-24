#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Tuple


def _read_rows(csv_path: Path) -> List[Tuple[int, float, int, int, float]]:
    rows: List[Tuple[int, float, int, int, float]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"idx", "t_sec", "gt", "pred", "conf"}
        if set(reader.fieldnames or []) < required:
            raise ValueError(f"unexpected CSV header: {reader.fieldnames} (need {sorted(required)})")
        for r in reader:
            rows.append(
                (
                    int(r["idx"]),
                    float(r["t_sec"]),
                    int(r["gt"]),
                    int(r["pred"]),
                    float(r["conf"]),
                )
            )
    return rows


def _polyline(points: List[Tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Plot route_id eval (pred vs gt) into a standalone SVG (no matplotlib).")
    p.add_argument("--csv", required=True, help="predictions.csv produced by eval_route_id_on_bag.py")
    p.add_argument("--out", required=True, help="Output .svg path")
    p.add_argument("--title", default="", help="Optional title")
    p.add_argument("--width", type=int, default=1200)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--k", type=int, default=0, help="Num classes (0=auto)")
    args = p.parse_args(argv)

    csv_path = Path(args.csv).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    rows = _read_rows(csv_path)
    if not rows:
        raise RuntimeError(f"empty CSV: {csv_path}")

    k = int(args.k)
    if k <= 0:
        k = 1 + max(max(gt, pred) for _, _, gt, pred, _ in rows if gt >= 0 or pred >= 0)
        k = max(k, 1)

    w = int(args.width)
    h = int(args.height)
    margin_l = 60
    margin_r = 20
    margin_t = 30 if str(args.title).strip() else 12
    margin_b = 28
    plot_w = max(10, w - margin_l - margin_r)
    plot_h = max(10, h - margin_t - margin_b)

    n = len(rows)
    x0 = float(margin_l)
    y0 = float(margin_t)

    def x_map(i: int) -> float:
        if n <= 1:
            return x0
        return x0 + (float(i) / float(n - 1)) * float(plot_w)

    def y_map(seg: int) -> float:
        if k <= 1:
            return y0 + plot_h / 2.0
        seg = max(0, min(k - 1, int(seg)))
        return y0 + (1.0 - (float(seg) / float(k - 1))) * float(plot_h)

    gt_pts: List[Tuple[float, float]] = []
    pred_pts: List[Tuple[float, float]] = []
    bad_pts: List[Tuple[float, float]] = []

    for i, (_idx, _t, gt, pred, _conf) in enumerate(rows):
        x = x_map(i)
        gt_y = y_map(gt if gt >= 0 else 0)
        pred_y = y_map(pred if pred >= 0 else 0)
        gt_pts.append((x, gt_y))
        pred_pts.append((x, pred_y))
        if pred != gt:
            bad_pts.append((x, pred_y))

    title = str(args.title).strip()
    svg: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{plot_w:.2f}" height="{plot_h:.2f}" fill="none" stroke="#111" stroke-width="1"/>',
    ]

    if title:
        svg.append(f'<text x="{w/2:.1f}" y="{margin_t-10:.1f}" text-anchor="middle" font-size="14" fill="#111">{title}</text>')

    for seg in range(0, k, max(1, k // 10)):
        y = y_map(seg)
        svg.append(f'<line x1="{x0:.2f}" y1="{y:.2f}" x2="{(x0+plot_w):.2f}" y2="{y:.2f}" stroke="#eee" stroke-width="1"/>')
        svg.append(f'<text x="{(x0-8):.2f}" y="{(y+4):.2f}" text-anchor="end" font-size="10" fill="#555">{seg}</text>')

    svg.append(f'<polyline points="{_polyline(gt_pts)}" fill="none" stroke="#666" stroke-width="2" opacity="0.9"/>')
    svg.append(f'<polyline points="{_polyline(pred_pts)}" fill="none" stroke="#d62728" stroke-width="2" opacity="0.85"/>')
    for x, y in bad_pts[:2000]:
        svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.8" fill="#ff7f0e" opacity="0.55"/>')

    svg.append(f'<text x="{x0:.2f}" y="{(y0+plot_h+20):.2f}" font-size="10" fill="#333">gray=GT, red=pred, orange=pred!=gt (N={n})</text>')
    svg.append("</svg>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(f"[OK] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

