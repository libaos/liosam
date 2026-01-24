#!/usr/bin/env python3
"""Evaluate BEV circle export runs and write a paper-friendly experiment log.

This script scans BEV export folders produced by:
  - src/orchard_row_mapping/tools/clustered_frames_to_circles_bev.py
  - src/orchard_row_mapping/tools/tree_frames_to_circles_bev.py

It computes quantitative metrics (cluster count stats, big-circle rate, etc.) and
writes CSV/Markdown summaries plus a small "paper trace" folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _safe_float(value: object, default: float) -> float:
    try:
        v = float(value)  # type: ignore[arg-type]
    except Exception:
        return float(default)
    return v if math.isfinite(v) else float(default)


def _safe_int(value: object, default: int) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return int(default)


def _percentile(values: Sequence[float], q: float) -> float:
    xs = [float(v) for v in values if math.isfinite(float(v))]
    if not xs:
        return float("nan")
    xs.sort()
    q = float(max(0.0, min(float(q), 1.0)))
    if len(xs) == 1:
        return float(xs[0])
    pos = q * float(len(xs) - 1)
    i0 = int(math.floor(pos))
    i1 = int(math.ceil(pos))
    if i0 == i1:
        return float(xs[i0])
    w = pos - float(i0)
    return float((1.0 - w) * float(xs[i0]) + w * float(xs[i1]))


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_circle_json_paths(circles_dir: Path) -> List[Path]:
    if not circles_dir.is_dir():
        return []
    files = sorted(circles_dir.glob("circles_*.json"))
    return files if files else sorted(circles_dir.glob("*.json"))


def _read_clusters_from_frames_csv(path: Path) -> List[int]:
    clusters: List[int] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if "clusters" not in row:
                continue
            try:
                clusters.append(int(str(row.get("clusters", "")).strip()))
            except Exception:
                pass
    return clusters


def _format_pct(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value * 100.0:.1f}%"


def _rel(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path)


def _symlink_force(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    rel = os.path.relpath(src.resolve(), start=dst.parent.resolve())
    dst.symlink_to(rel)


def _get_upstream_run_meta(run_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    in_dir = str(run_meta.get("in_dir") or "").strip()
    if not in_dir:
        return None
    meta_path = Path(in_dir).expanduser().resolve() / "run_meta.json"
    if not meta_path.is_file():
        return None
    try:
        return _read_json(meta_path)
    except Exception:
        return None


@dataclass(frozen=True)
class RunMetrics:
    group: str
    run_name: str
    run_dir: Path
    algo_name: str
    circle_z_mode: str
    radius_max: float
    frames: int
    clusters_mean: float
    clusters_median: float
    clusters_std: float
    clusters_min: int
    clusters_p10: float
    clusters_p90: float
    clusters_max: int
    total_circles: int
    radius_mean: float
    radius_median: float
    radius_p10: float
    radius_p90: float
    radius_max_hit_rate: float
    frames_any_radius_max: float
    in_dir: str
    upstream_cluster_z_mode: str
    upstream_dbscan_eps: str
    upstream_dbscan_min_samples: str
    upstream_dbscan_min_cluster_size: str


def _collect_run_metrics(run_dir: Path, *, group: str) -> Optional[RunMetrics]:
    frames_csv = run_dir / "frames.csv"
    run_meta_path = run_dir / "run_meta.json"
    circles_dir = run_dir / "circles"

    if not frames_csv.is_file() or not run_meta_path.is_file() or not circles_dir.is_dir():
        return None

    run_meta = _read_json(run_meta_path)
    algo_name = str(run_meta.get("algo_name") or run_dir.name).strip()
    circle_z_mode = str((run_meta.get("circle_z") or {}).get("mode") or "").strip()
    radius_max = _safe_float((run_meta.get("radius") or {}).get("radius_max"), 1.2)

    clusters = _read_clusters_from_frames_csv(frames_csv)
    if not clusters:
        return None

    clusters_f = [float(v) for v in clusters]
    clusters_mean = float(sum(clusters_f) / float(len(clusters_f)))
    clusters_median = float(statistics.median(clusters))
    clusters_std = float(statistics.pstdev(clusters_f)) if len(clusters_f) > 1 else 0.0
    clusters_min = int(min(clusters))
    clusters_max = int(max(clusters))
    clusters_p10 = float(_percentile(clusters_f, 0.10))
    clusters_p90 = float(_percentile(clusters_f, 0.90))

    radii: List[float] = []
    maxhit = 0
    frames_any = 0
    for circle_json in _iter_circle_json_paths(circles_dir):
        obj = _read_json(circle_json)
        rs = [c.get("radius") for c in (obj.get("circles") or [])]
        rs_f = [float(r) for r in rs if r is not None and math.isfinite(float(r))]
        radii.extend(rs_f)
        mh = sum(1 for r in rs_f if float(r) >= float(radius_max) - 1.0e-6)
        maxhit += int(mh)
        if mh > 0:
            frames_any += 1

    radius_mean = float(sum(radii) / float(len(radii))) if radii else float("nan")
    radius_median = float(statistics.median(radii)) if radii else float("nan")
    radius_p10 = float(_percentile(radii, 0.10)) if radii else float("nan")
    radius_p90 = float(_percentile(radii, 0.90)) if radii else float("nan")
    radius_max_hit_rate = float(maxhit) / float(len(radii)) if radii else 0.0
    frames_any_radius_max = float(frames_any) / float(len(clusters)) if clusters else 0.0

    upstream = _get_upstream_run_meta(run_meta) or {}
    upstream_cluster_z_mode = str((upstream.get("cluster_z") or {}).get("mode") or "").strip()
    upstream_dbscan = upstream.get("dbscan") or {}
    upstream_dbscan_eps = str(upstream_dbscan.get("eps") or "").strip()
    upstream_dbscan_min_samples = str(upstream_dbscan.get("min_samples") or "").strip()
    upstream_dbscan_min_cluster_size = str(upstream_dbscan.get("min_cluster_size") or "").strip()

    return RunMetrics(
        group=str(group),
        run_name=str(run_dir.name),
        run_dir=run_dir,
        algo_name=algo_name,
        circle_z_mode=circle_z_mode,
        radius_max=float(radius_max),
        frames=int(len(clusters)),
        clusters_mean=clusters_mean,
        clusters_median=clusters_median,
        clusters_std=clusters_std,
        clusters_min=clusters_min,
        clusters_p10=clusters_p10,
        clusters_p90=clusters_p90,
        clusters_max=clusters_max,
        total_circles=int(len(radii)),
        radius_mean=radius_mean,
        radius_median=radius_median,
        radius_p10=radius_p10,
        radius_p90=radius_p90,
        radius_max_hit_rate=radius_max_hit_rate,
        frames_any_radius_max=frames_any_radius_max,
        in_dir=str(run_meta.get("in_dir") or "").strip(),
        upstream_cluster_z_mode=upstream_cluster_z_mode,
        upstream_dbscan_eps=upstream_dbscan_eps,
        upstream_dbscan_min_samples=upstream_dbscan_min_samples,
        upstream_dbscan_min_cluster_size=upstream_dbscan_min_cluster_size,
    )


def _write_csv(path: Path, rows: List[RunMetrics], *, root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "group",
        "run_name",
        "run_dir",
        "algo_name",
        "circle_z_mode",
        "upstream_cluster_z_mode",
        "upstream_dbscan_eps",
        "upstream_dbscan_min_samples",
        "upstream_dbscan_min_cluster_size",
        "frames",
        "clusters_mean",
        "clusters_median",
        "clusters_std",
        "clusters_min",
        "clusters_p10",
        "clusters_p90",
        "clusters_max",
        "total_circles",
        "radius_mean",
        "radius_median",
        "radius_p10",
        "radius_p90",
        "radius_max",
        "radius_max_hit_rate",
        "frames_any_radius_max",
        "in_dir",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "group": r.group,
                    "run_name": r.run_name,
                    "run_dir": _rel(r.run_dir, root),
                    "algo_name": r.algo_name,
                    "circle_z_mode": r.circle_z_mode,
                    "upstream_cluster_z_mode": r.upstream_cluster_z_mode,
                    "upstream_dbscan_eps": r.upstream_dbscan_eps,
                    "upstream_dbscan_min_samples": r.upstream_dbscan_min_samples,
                    "upstream_dbscan_min_cluster_size": r.upstream_dbscan_min_cluster_size,
                    "frames": r.frames,
                    "clusters_mean": f"{r.clusters_mean:.6f}",
                    "clusters_median": f"{r.clusters_median:.6f}",
                    "clusters_std": f"{r.clusters_std:.6f}",
                    "clusters_min": r.clusters_min,
                    "clusters_p10": f"{r.clusters_p10:.6f}",
                    "clusters_p90": f"{r.clusters_p90:.6f}",
                    "clusters_max": r.clusters_max,
                    "total_circles": r.total_circles,
                    "radius_mean": f"{r.radius_mean:.6f}" if math.isfinite(r.radius_mean) else "",
                    "radius_median": f"{r.radius_median:.6f}" if math.isfinite(r.radius_median) else "",
                    "radius_p10": f"{r.radius_p10:.6f}" if math.isfinite(r.radius_p10) else "",
                    "radius_p90": f"{r.radius_p90:.6f}" if math.isfinite(r.radius_p90) else "",
                    "radius_max": f"{r.radius_max:.6f}",
                    "radius_max_hit_rate": f"{r.radius_max_hit_rate:.6f}",
                    "frames_any_radius_max": f"{r.frames_any_radius_max:.6f}",
                    "in_dir": r.in_dir,
                }
            )


def _write_markdown(path: Path, rows: List[RunMetrics], *, root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# BEV 圆圈方案评估（自动汇总）")
    lines.append("")
    lines.append("核心指标（越小越好）：")
    lines.append("- `frames_any_radius_max`：有“超大圆圈（半径被夹到 radius_max）”的帧占比（粘连/合并强信号）")
    lines.append("- `radius_max_hit_rate`：所有圆圈里半径触顶（==radius_max）的比例")
    lines.append("")

    by_group: Dict[str, List[RunMetrics]] = {}
    for r in rows:
        by_group.setdefault(r.group, []).append(r)

    for group, group_rows in sorted(by_group.items(), key=lambda x: x[0]):
        group_rows_sorted = sorted(group_rows, key=lambda r: (r.frames_any_radius_max, r.radius_max_hit_rate, -r.clusters_median))
        best = group_rows_sorted[0] if group_rows_sorted else None
        lines.append(f"## {group}")
        lines.append("")
        if best is not None:
            lines.append(f"- 推荐（按合并风险排序）：`{_rel(best.run_dir, root)}`")
            lines.append(f"  - clusters median/mean: {best.clusters_median:.1f} / {best.clusters_mean:.1f}")
            lines.append(f"  - frames_any_radius_max: {_format_pct(best.frames_any_radius_max)}")
            lines.append(f"  - radius_max_hit_rate: {_format_pct(best.radius_max_hit_rate)}")
            if best.upstream_cluster_z_mode:
                lines.append(f"  - upstream cluster_z_mode: `{best.upstream_cluster_z_mode}`")
        lines.append("")

        lines.append("|run_dir|clusters(median)|clusters(mean)|frames_any_radius_max|radius_max_hit_rate|circle_z_mode|upstream_cluster_z_mode|")
        lines.append("|---|---:|---:|---:|---:|---|---|")
        for r in group_rows_sorted:
            lines.append(
                "|"
                + "|".join(
                    [
                        f"`{_rel(r.run_dir, root)}`",
                        f"{r.clusters_median:.1f}",
                        f"{r.clusters_mean:.1f}",
                        _format_pct(r.frames_any_radius_max),
                        _format_pct(r.radius_max_hit_rate),
                        f"`{r.circle_z_mode}`" if r.circle_z_mode else "",
                        f"`{r.upstream_cluster_z_mode}`" if r.upstream_cluster_z_mode else "",
                    ]
                )
                + "|"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pick_example_frames(baseline_dir: Path, improved_dir: Path, *, max_examples: int) -> List[int]:
    baseline_circles = baseline_dir / "circles"
    improved_circles = improved_dir / "circles"
    baseline_meta = _read_json(baseline_dir / "run_meta.json")
    improved_meta = _read_json(improved_dir / "run_meta.json")
    rmax = _safe_float((baseline_meta.get("radius") or {}).get("radius_max"), 1.2)
    rmax2 = _safe_float((improved_meta.get("radius") or {}).get("radius_max"), rmax)
    rmax = float(min(rmax, rmax2))

    candidates: List[Tuple[int, int]] = []  # (delta_maxhits, index)
    for p in _iter_circle_json_paths(baseline_circles):
        idx = _safe_int(p.stem.split("_")[-1], -1)
        if idx < 0:
            continue
        q = improved_circles / p.name
        if not q.is_file():
            continue
        a = _read_json(p)
        b = _read_json(q)
        ra = [float(c.get("radius")) for c in (a.get("circles") or []) if c.get("radius") is not None]
        rb = [float(c.get("radius")) for c in (b.get("circles") or []) if c.get("radius") is not None]
        max_a = sum(1 for r in ra if math.isfinite(r) and r >= rmax - 1.0e-6)
        max_b = sum(1 for r in rb if math.isfinite(r) and r >= rmax - 1.0e-6)
        delta = int(max_a) - int(max_b)
        if delta <= 0:
            continue
        candidates.append((delta, idx))

    candidates.sort(key=lambda x: (-x[0], x[1]))
    return [int(idx) for _, idx in candidates[: max(0, int(max_examples))]]


def _write_paper_trace(
    out_dir: Path,
    *,
    bev_root: Path,
    metrics: List[RunMetrics],
    preferred_best_rel: Optional[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Best per group (same sorting as metrics.md)
    best_by_group: Dict[str, RunMetrics] = {}
    for group in sorted({m.group for m in metrics}):
        group_rows = [m for m in metrics if m.group == group]
        if not group_rows:
            continue
        group_rows.sort(key=lambda r: (r.frames_any_radius_max, r.radius_max_hit_rate, -r.clusters_median))
        best_by_group[group] = group_rows[0]

    preferred_best: Optional[RunMetrics] = None
    if preferred_best_rel:
        candidate = (bev_root / Path(preferred_best_rel)).resolve()
        for m in metrics:
            if m.run_dir.resolve() == candidate:
                preferred_best = m
                break

    stamp = time.strftime("%Y%m%d_%H%M%S")
    manifest = {
        "generated_at": stamp,
        "bev_root": str(bev_root),
        "groups": sorted({m.group for m in metrics}),
        "best_by_group": {g: _rel(m.run_dir, bev_root) for g, m in best_by_group.items()},
        "preferred_best": _rel(preferred_best.run_dir, bev_root) if preferred_best else "",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines: List[str] = []
    lines.append("# 论文实验留痕（BEV 圆圈方案）")
    lines.append("")
    lines.append("这个目录用于记录：试了哪些方案、用什么指标选的最好、如何复现。")
    lines.append("")
    lines.append("## 指标")
    lines.append("")
    lines.append("- `frames_any_radius_max`：有“超大圆圈（半径被夹到 radius_max）”的帧占比（粘连/合并强信号）")
    lines.append("- `radius_max_hit_rate`：所有圆圈里半径触顶（==radius_max）的比例")
    lines.append("")
    lines.append("## 输出文件")
    lines.append("")
    lines.append("- `metrics.csv`：完整指标表（可直接丢 Excel）")
    lines.append("- `metrics.md`：指标表（按组展示）")
    lines.append("- `manifest.json`：机器可读清单（含每组推荐）")
    lines.append("")

    if preferred_best is not None:
        lines.append("## 最终采用")
        lines.append("")
        lines.append(f"- `{_rel(preferred_best.run_dir, bev_root)}`")
        lines.append("")

        # Baseline selection (for paper figures):
        # Prefer the directory the user typically browses ("DBSCAN_不切片"), otherwise fall back to the worst DBSCAN.
        baseline: Optional[RunMetrics] = None
        dbscan_candidates = [
            m
            for m in metrics
            if m.group == preferred_best.group
            and ("DBSCAN" in m.run_name)
            and (not m.upstream_cluster_z_mode)
            and m.run_name != preferred_best.run_name
        ]
        if dbscan_candidates:
            for cand in dbscan_candidates:
                if "DBSCAN_不切片" in cand.run_name:
                    baseline = cand
                    break
            if baseline is None:
                dbscan_candidates.sort(key=lambda r: (-r.frames_any_radius_max, -r.radius_max_hit_rate))
                baseline = dbscan_candidates[0]

        if baseline is not None:
            fig_dir = out_dir / "figures" / "chunk10_dbscan_baseline_vs_improved"
            if fig_dir.is_dir():
                for entry in fig_dir.iterdir():
                    if entry.is_symlink() or entry.is_file():
                        entry.unlink()
            picked = _pick_example_frames(baseline.run_dir, preferred_best.run_dir, max_examples=8)
            picked = [
                idx
                for idx in picked
                if (baseline.run_dir / "png" / f"bev_{idx:06d}.png").is_file()
                and (preferred_best.run_dir / "png" / f"bev_{idx:06d}.png").is_file()
            ]
            for idx in picked:
                _symlink_force(baseline.run_dir / "png" / f"bev_{idx:06d}.png", fig_dir / f"frame_{idx:06d}_baseline.png")
                _symlink_force(preferred_best.run_dir / "png" / f"bev_{idx:06d}.png", fig_dir / f"frame_{idx:06d}_improved.png")

            if picked:
                lines.append("## 代表性对比图（软链接）")
                lines.append("")
                lines.append("- 便于直接放进论文：")
                lines.append(f"  - `{_rel(fig_dir, out_dir)}/frame_*_baseline.png`")
                lines.append(f"  - `{_rel(fig_dir, out_dir)}/frame_*_improved.png`")
                lines.append("")

        lines.append("## 复现（关键步骤）")
        lines.append("")
        lines.append("每个导出目录都带 `run_meta.json`，里面是最完整的参数记录。下面给出“最佳方案”的关键命令模板：")
        lines.append("")
        lines.append("```bash")
        lines.append("# 1) 聚类（以 trunk slice 作为聚类输入，避免树冠桥接）")
        lines.append(
            "python3 src/orchard_row_mapping/tools/cluster_tree_frames_variants.py \\\n"
            "  --in-dir output/点云导出_20260120_整理/07_每5帧合成地图PCD/07_树点_map_TF对齐_每10帧_不裁剪 \\\n"
            "  --out-root output/点云导出_20260120_整理/09_聚类_每10帧合成树点map \\\n"
            "  --disable-cell --disable-euclid --disable-kmeans \\\n"
            "  --cluster-z-mode fixed --cluster-z-min 0.7 --cluster-z-max 1.3"
        )
        lines.append("")
        lines.append("# 2) 圆圈 + BEV（从聚类结果目录导出 circles/json + PNG）")
        out_dir_rel = _rel(preferred_best.run_dir, bev_root)
        lines.append(
            "python3 src/orchard_row_mapping/tools/clustered_frames_to_circles_bev.py \\\n"
            "  --in-dir <上一步输出的聚类_DBSCAN_*/目录> \\\n"
            f"  --out-dir output/点云导出_20260120_整理/05_BEV预览/{out_dir_rel} \\\n"
            "  --algo-name 'DBSCAN_每10帧合成树点map_聚类用树干高度0p7-1p3' \\\n"
            "  --z-min 0.7 --z-max 1.3 --x-min -20 --x-max 60 --y-abs-max 30 \\\n"
            "  --render-z-min -1000000000 --render-z-max 1000000000 \\\n"
            "  --circle-z-mode fixed --bounds-mode center --window-x 30 --window-y 20 --draw-title 1"
        )
        lines.append("```")
        lines.append("")

    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bev-root",
        type=str,
        default="output/点云导出_20260120_整理/05_BEV预览",
        help="BEV export root directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="output/点云导出_20260120_整理/10_论文实验留痕",
        help="Where to write the experiment trace (CSV/MD/manifest/figures).",
    )
    parser.add_argument(
        "--groups",
        type=str,
        default="40_每5帧合成树点map(565张),50_每10帧合成树点map(283张)",
        help="Comma-separated subfolders under bev-root to evaluate.",
    )
    parser.add_argument(
        "--preferred-best",
        type=str,
        default="50_每10帧合成树点map(283张)/52_圆圈_每10帧合成树点map_DBSCAN_聚类用树干高度0p7-1p3",
        help="Optional relative path under bev-root for the final 'best' section.",
    )
    args = parser.parse_args()

    bev_root = Path(args.bev_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    groups = [g.strip() for g in str(args.groups).split(",") if g.strip()]
    preferred_best_rel = str(args.preferred_best).strip() if str(args.preferred_best).strip() else None

    all_metrics: List[RunMetrics] = []
    for g in groups:
        group_dir = (bev_root / g).resolve()
        if not group_dir.is_dir():
            continue
        for run_dir in sorted([p for p in group_dir.iterdir() if p.is_dir() and not p.is_symlink()]):
            m = _collect_run_metrics(run_dir, group=str(g))
            if m is not None:
                all_metrics.append(m)

    if not all_metrics:
        raise RuntimeError(f"No runs found under: {bev_root} (groups={groups})")

    all_metrics.sort(key=lambda r: (r.group, r.frames_any_radius_max, r.radius_max_hit_rate, -r.clusters_median))
    _write_csv(out_dir / "metrics.csv", all_metrics, root=bev_root)
    _write_markdown(out_dir / "metrics.md", all_metrics, root=bev_root)
    _write_paper_trace(out_dir, bev_root=bev_root, metrics=all_metrics, preferred_best_rel=preferred_best_rel)
    print(f"[OK] Wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
