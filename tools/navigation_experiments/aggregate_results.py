#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _row_from_metrics(obj: Dict[str, Any], src: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "metrics_json": str(src),
        "method": obj.get("method") or "",
        "run_bag": obj.get("run_bag") or "",
        "success": obj.get("success"),
        "duration_s": obj.get("duration_s"),
        "travel_distance_m": obj.get("travel_distance_m"),
        "ref_length_m": obj.get("ref_length_m"),
        "end_error_to_goal_m": obj.get("end_error_to_goal_m"),
        "cte_mean_m": _get(obj, "cte_overall.mean"),
        "cte_rmse_m": _get(obj, "cte_overall.rmse"),
        "cte_p95_m": _get(obj, "cte_overall.p95"),
        "cte_max_m": _get(obj, "cte_overall.max"),
        "mode_switches": _get(obj, "mode_stats.mode_switches"),
        "boundary_delay_abs_mean_m": _get(obj, "mode_stats.boundary_delay_abs_mean_m"),
    }

    for lab in ("straight", "left", "right", "unknown"):
        row[f"{lab}_rmse_m"] = _get(obj, f"cte_by_label.{lab}.rmse")
        row[f"{lab}_p95_m"] = _get(obj, f"cte_by_label.{lab}.p95")
        row[f"{lab}_max_m"] = _get(obj, f"cte_by_label.{lab}.max")

    return row


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Aggregate multiple evaluate_run.py outputs into a single CSV.")
    p.add_argument("--inputs", nargs="+", required=True, help="metrics.json files (can be globbed by shell)")
    p.add_argument("--out-csv", required=True, help="Output summary CSV")
    args = p.parse_args(argv)

    inputs = [Path(x).expanduser().resolve() for x in args.inputs]
    rows: List[Dict[str, Any]] = []
    for path in inputs:
        if not path.is_file():
            raise RuntimeError(f"metrics.json not found: {path}")
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise RuntimeError(f"Invalid metrics json: {path}")
        rows.append(_row_from_metrics(obj, path))

    out_csv = Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Stable field order
    fieldnames = list(rows[0].keys()) if rows else []
    for r in rows[1:]:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] Wrote: {out_csv}  (rows={len(rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

