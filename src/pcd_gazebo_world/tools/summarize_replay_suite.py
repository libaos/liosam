#!/usr/bin/env python3
"""Summarize plot_reference_vs_replay.py report JSONs into a compact table.

This is useful when iterating Gazebo tuning (TEB/costmap) and you want a quick
"which run is better" view without opening every overlay file.
"""

from __future__ import annotations

import argparse
import json
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


def main() -> int:
    ws_dir = Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser(description="Summarize replay suite report JSONs")
    parser.add_argument("--dir", type=str, default="", help="Directory containing *_report.json (default: latest replay_suite_*)")
    parser.add_argument("--out-csv", type=str, default="", help="Optional CSV output path")
    args = parser.parse_args()

    if args.dir.strip():
        run_dir = Path(args.dir).expanduser().resolve()
    else:
        run_dir = _find_latest_suite_dir(ws_dir)
        if run_dir is None:
            raise SystemExit("No replay suite directory found under trajectory_data/")

    report_paths = sorted(run_dir.glob("*_report.json"))
    if not report_paths:
        raise SystemExit(f"No *_report.json found in: {run_dir}")

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

