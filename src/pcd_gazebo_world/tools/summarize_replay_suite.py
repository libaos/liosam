#!/usr/bin/env python3
"""Summarize plot_reference_vs_replay.py report JSONs into a compact table.

This is useful when iterating Gazebo tuning (TEB/costmap) and you want a quick
"which run is better" view without opening every overlay file.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _find_latest_suite_dir(ws_dir: Path) -> Optional[Path]:
    base = ws_dir / "trajectory_data"
    if not base.is_dir():
        return None
    candidates = sorted(base.glob("replay_suite_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for c in candidates:
        if c.is_dir() and list(c.glob("*_report.json")):
            return c
    return None


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_nested(obj: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    cur: Any = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _fmt_float(x: Any, width: int = 7, prec: int = 2) -> str:
    try:
        v = float(x)
    except Exception:
        return " " * width
    return f"{v:{width}.{prec}f}"


def _fmt_pct(x: Any, width: int = 7) -> str:
    try:
        v = float(x) * 100.0
    except Exception:
        return " " * width
    return f"{v:{width}.1f}%"


def _collect_reports(report_paths: List[Path]) -> List[Tuple[str, Dict[str, Any]]]:
    items: List[Tuple[str, Dict[str, Any]]] = []
    for p in report_paths:
        obj = _read_json(p)
        label = p.name.replace("_report.json", "")
        items.append((label, obj))
    return items


def _group_from_label(label: str) -> str:
    s = str(label).strip()
    if not s:
        return "unknown"
    return s.split("_", 1)[0] or "unknown"


def _mean_std(values: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return (None, None)
    m = float(sum(vals) / len(vals))
    if len(vals) < 2:
        return (m, 0.0)
    var = float(sum((x - m) ** 2 for x in vals) / len(vals))
    return (m, float(math.sqrt(var)))


def _percentile(values: List[Optional[float]], q: float) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return None
    if q <= 0:
        return float(min(vals))
    if q >= 100:
        return float(max(vals))
    vals.sort()
    # Nearest-rank percentile.
    k = int(math.ceil((q / 100.0) * len(vals))) - 1
    k = max(0, min(k, len(vals) - 1))
    return float(vals[k])


def main() -> int:
    ws_dir = Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser(description="Summarize replay suite report JSONs")
    parser.add_argument("--dir", type=str, default="", help="Directory containing *_report.json (default: latest replay_suite_*)")
    parser.add_argument("--out-csv", type=str, default="", help="Optional CSV output path")
    parser.add_argument("--out-group-csv", type=str, default="", help="Optional grouped CSV output path")
    parser.add_argument("--out-group-plus-csv", type=str, default="", help="Optional grouped CSV output path (with replay_length stats)")
    args = parser.parse_args()

    if args.dir.strip():
        run_dir = Path(args.dir).expanduser().resolve()
    else:
        run_dir = _find_latest_suite_dir(ws_dir)
        if run_dir is None:
            raise SystemExit("No replay suite directory found under trajectory_data/")

    report_paths = sorted(run_dir.rglob("*_report.json"))
    if not report_paths:
        print(f"[warn] No *_report.json found in: {run_dir}", flush=True)
        # Still write empty CSVs if requested, to avoid leaving broken symlinks in
        # human-facing results folders when a run is aborted early.
        if str(args.out_csv).strip():
            out_csv = Path(args.out_csv).expanduser().resolve()
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            out_csv.write_text(
                "label,coverage_ratio,mean_m,p95_m,max_m,reference_length_m,replay_length_m,replay_closure_m\n",
                encoding="utf-8",
            )
            print(f"[OK] wrote: {out_csv}")
        if str(args.out_group_csv).strip():
            out_csv = Path(args.out_group_csv).expanduser().resolve()
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            out_csv.write_text(
                "group,n,coverage_ratio_mean,mean_m_mean,p95_m_mean,max_m_mean\n",
                encoding="utf-8",
            )
            print(f"[OK] wrote: {out_csv}")
        if str(args.out_group_plus_csv).strip():
            out_csv = Path(args.out_group_plus_csv).expanduser().resolve()
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            out_csv.write_text(
                "group,n,"
                "coverage_ratio_mean,mean_m_mean,p95_m_mean,max_m_mean,"
                "replay_length_m_mean,replay_length_m_p50,replay_length_m_p95,replay_length_m_min,replay_length_m_max,"
                "replay_closure_m_mean\n",
                encoding="utf-8",
            )
            print(f"[OK] wrote: {out_csv}")
        return 0

    items = _collect_reports(report_paths)

    # Sort by coverage desc, then mean error asc.
    def key(item: Tuple[str, Dict[str, Any]]):
        _label, obj = item
        cov = float(obj.get("coverage_ratio", 0.0) or 0.0)
        mean = float(_get_nested(obj, ["replay_to_reference", "mean_m"], 1.0e9) or 1.0e9)
        return (-cov, mean)

    items.sort(key=key)

    header = (
        f"run_dir: {run_dir}\n"
        "label                      cover    mean    p95    max   ref_L  rep_L  rep_close\n"
        "-------------------------  ------  ------  ------  ------  ------  ------  ---------\n"
    )
    lines = [header]
    for label, obj in items:
        cov = _fmt_pct(obj.get("coverage_ratio", None), width=6)
        mean = _fmt_float(_get_nested(obj, ["replay_to_reference", "mean_m"]), width=6, prec=2)
        p95 = _fmt_float(_get_nested(obj, ["replay_to_reference", "p95_m"]), width=6, prec=2)
        mx = _fmt_float(_get_nested(obj, ["replay_to_reference", "max_m"]), width=6, prec=2)
        ref_L = _fmt_float(obj.get("reference_length_m", None), width=6, prec=1)
        rep_L = _fmt_float(obj.get("replay_length_m", None), width=6, prec=1)
        rep_close = _fmt_float(obj.get("replay_closure_m", None), width=8, prec=2)
        lines.append(f"{label:<25}  {cov:>6}  {mean:>6}  {p95:>6}  {mx:>6}  {ref_L:>6}  {rep_L:>6}  {rep_close:>9}")

    print("\n".join(lines))

    if str(args.out_csv).strip():
        out_csv = Path(args.out_csv).expanduser().resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8") as f:
            f.write("label,coverage_ratio,mean_m,p95_m,max_m,reference_length_m,replay_length_m,replay_closure_m\n")
            for label, obj in items:
                f.write(
                    ",".join(
                        [
                            label,
                            str(obj.get("coverage_ratio", "")),
                            str(_get_nested(obj, ["replay_to_reference", "mean_m"], "")),
                            str(_get_nested(obj, ["replay_to_reference", "p95_m"], "")),
                            str(_get_nested(obj, ["replay_to_reference", "max_m"], "")),
                            str(obj.get("reference_length_m", "")),
                            str(obj.get("replay_length_m", "")),
                            str(obj.get("replay_closure_m", "")),
                        ]
                    )
                    + "\n"
                )
        print(f"[OK] wrote: {out_csv}")

    groups: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    for label, obj in items:
        groups[_group_from_label(label)].append((label, obj))

    group_lines = [
        "",
        "group         n    cover_mean  mean_m_mean  p95_m_mean  max_m_mean",
        "-----------  --  ----------  ----------  ----------  ----------",
    ]
    group_rows: List[Tuple[str, int, Optional[float], Optional[float], Optional[float], Optional[float]]] = []
    for g, g_items in sorted(groups.items(), key=lambda kv: kv[0]):
        covs = [float(obj.get("coverage_ratio")) if obj.get("coverage_ratio") is not None else None for _lbl, obj in g_items]
        means = [
            float(_get_nested(obj, ["replay_to_reference", "mean_m"])) if _get_nested(obj, ["replay_to_reference", "mean_m"]) is not None else None
            for _lbl, obj in g_items
        ]
        p95s = [
            float(_get_nested(obj, ["replay_to_reference", "p95_m"])) if _get_nested(obj, ["replay_to_reference", "p95_m"]) is not None else None
            for _lbl, obj in g_items
        ]
        maxs = [
            float(_get_nested(obj, ["replay_to_reference", "max_m"])) if _get_nested(obj, ["replay_to_reference", "max_m"]) is not None else None
            for _lbl, obj in g_items
        ]
        cov_m, cov_std = _mean_std(covs)
        mean_m, mean_std = _mean_std(means)
        p95_m, p95_std = _mean_std(p95s)
        max_m, max_std = _mean_std(maxs)
        group_rows.append((g, len(g_items), cov_m, mean_m, p95_m, max_m))

    # Sort by coverage desc, then mean error asc.
    group_rows.sort(key=lambda r: (-(r[2] or 0.0), (r[3] or 1.0e9)))
    for g, n, cov_m, mean_m, p95_m, max_m in group_rows:
        group_lines.append(
            f"{g:<11}  {n:>2}  {_fmt_pct(cov_m, width=10):>10}  {_fmt_float(mean_m, width=10, prec=3):>10}  {_fmt_float(p95_m, width=10, prec=3):>10}  {_fmt_float(max_m, width=10, prec=3):>10}"
        )
    print("\n".join(group_lines))

    if str(args.out_group_csv).strip():
        out_csv = Path(args.out_group_csv).expanduser().resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8") as f:
            f.write("group,n,coverage_ratio_mean,mean_m_mean,p95_m_mean,max_m_mean\n")
            for g, n, cov_m, mean_m, p95_m, max_m in group_rows:
                f.write(",".join([g, str(n), str(cov_m or ""), str(mean_m or ""), str(p95_m or ""), str(max_m or "")]) + "\n")
        print(f"[OK] wrote: {out_csv}")

    if str(args.out_group_plus_csv).strip():
        out_csv = Path(args.out_group_plus_csv).expanduser().resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8") as f:
            f.write(
                "group,n,"
                "coverage_ratio_mean,mean_m_mean,p95_m_mean,max_m_mean,"
                "replay_length_m_mean,replay_length_m_p50,replay_length_m_p95,replay_length_m_min,replay_length_m_max,"
                "replay_closure_m_mean\n"
            )
            for g, g_items in sorted(groups.items(), key=lambda kv: kv[0]):
                covs = [float(obj.get("coverage_ratio")) if obj.get("coverage_ratio") is not None else None for _lbl, obj in g_items]
                means = [
                    float(_get_nested(obj, ["replay_to_reference", "mean_m"])) if _get_nested(obj, ["replay_to_reference", "mean_m"]) is not None else None
                    for _lbl, obj in g_items
                ]
                p95s = [
                    float(_get_nested(obj, ["replay_to_reference", "p95_m"])) if _get_nested(obj, ["replay_to_reference", "p95_m"]) is not None else None
                    for _lbl, obj in g_items
                ]
                maxs = [
                    float(_get_nested(obj, ["replay_to_reference", "max_m"])) if _get_nested(obj, ["replay_to_reference", "max_m"]) is not None else None
                    for _lbl, obj in g_items
                ]
                rep_Ls = [float(obj.get("replay_length_m")) if obj.get("replay_length_m") is not None else None for _lbl, obj in g_items]
                rep_closes = [float(obj.get("replay_closure_m")) if obj.get("replay_closure_m") is not None else None for _lbl, obj in g_items]

                cov_m, _ = _mean_std(covs)
                mean_m, _ = _mean_std(means)
                p95_m, _ = _mean_std(p95s)
                max_m, _ = _mean_std(maxs)
                rep_L_m, _ = _mean_std(rep_Ls)
                rep_L_p50 = _percentile(rep_Ls, 50.0)
                rep_L_p95 = _percentile(rep_Ls, 95.0)
                rep_L_min = _percentile(rep_Ls, 0.0)
                rep_L_max = _percentile(rep_Ls, 100.0)
                rep_close_m, _ = _mean_std(rep_closes)

                f.write(
                    ",".join(
                        [
                            g,
                            str(len(g_items)),
                            str(cov_m or ""),
                            str(mean_m or ""),
                            str(p95_m or ""),
                            str(max_m or ""),
                            str(rep_L_m or ""),
                            str(rep_L_p50 or ""),
                            str(rep_L_p95 or ""),
                            str(rep_L_min or ""),
                            str(rep_L_max or ""),
                            str(rep_close_m or ""),
                        ]
                    )
                    + "\n"
                )
        print(f"[OK] wrote: {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
