#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple


@dataclass(frozen=True)
class PredRow:
    idx: int
    t_sec: float
    gt: int
    pred: int
    conf: float


@dataclass(frozen=True)
class GateRow:
    idx: int
    t_sec: float
    gt: int
    pred: int
    conf: float
    valid: int
    state: str


def _read_predictions_csv(path: Path) -> List[PredRow]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"idx", "t_sec", "gt", "pred", "conf"}
        if set(reader.fieldnames or []) < required:
            raise ValueError(f"unexpected CSV header: {reader.fieldnames} (need {sorted(required)})")
        rows: List[PredRow] = []
        for r in reader:
            rows.append(
                PredRow(
                    idx=int(r["idx"]),
                    t_sec=float(r["t_sec"]),
                    gt=int(r["gt"]),
                    pred=int(r["pred"]),
                    conf=float(r["conf"]),
                )
            )
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    return rows


def _count_switches(preds: Iterable[int]) -> int:
    last: Optional[int] = None
    switches = 0
    for p in preds:
        if last is not None and int(p) != int(last):
            switches += 1
        last = int(p)
    return switches


def _read_idx_set_from_predictions_csv(path: Path) -> Set[int]:
    rows = _read_predictions_csv(path)
    return {int(r.idx) for r in rows}


def _simulate_gate(
    rows: List[PredRow],
    *,
    conf_th: float,
    stable_n: int,
    timeout_s: float,
    allowed_jump: int,
    vote_window: int,
    vote_min_samples: int,
    vote_switch_ratio: float,
    unknown_id: int,
) -> List[GateRow]:
    stable_id: Optional[int] = None
    candidate_id: Optional[int] = None
    candidate_count = 0
    low_conf_start: Optional[float] = None

    id_hist: Deque[int] = deque(maxlen=vote_window if vote_window > 0 else 1)
    voted_id: Optional[int] = None

    out: List[GateRow] = []
    for r in rows:
        now = float(r.t_sec)
        route_id = int(r.pred)
        conf = float(r.conf)

        if vote_window > 0:
            id_hist.append(route_id)

        if route_id < 0:
            candidate_id = None
            candidate_count = 0
            low_conf_start = None
            id_hist.clear()
            voted_id = None
            out.append(GateRow(r.idx, r.t_sec, r.gt, int(unknown_id), conf, 0, "UNKNOWN"))
            continue

        if conf < float(conf_th):
            if low_conf_start is None:
                low_conf_start = now
            candidate_id = None
            candidate_count = 0

            if stable_id is not None and float(timeout_s) > 0.0 and (now - float(low_conf_start)) <= float(timeout_s):
                out.append(GateRow(r.idx, r.t_sec, r.gt, int(stable_id), conf, 0, "UNSURE"))
                continue

            if vote_window > 0 and len(id_hist) >= int(vote_min_samples):
                voted, count = Counter(id_hist).most_common(1)[0]
                ratio = float(count) / float(len(id_hist))
                if voted_id is None:
                    voted_id = int(voted)
                elif int(voted) != int(voted_id) and ratio >= float(vote_switch_ratio):
                    voted_id = int(voted)
                out.append(GateRow(r.idx, r.t_sec, r.gt, int(voted_id), conf, 0, "VOTE"))
                continue

            out.append(GateRow(r.idx, r.t_sec, r.gt, int(unknown_id), conf, 0, "UNKNOWN"))
            continue

        low_conf_start = None

        if stable_id is not None and int(allowed_jump) >= 0 and abs(int(route_id) - int(stable_id)) > int(allowed_jump):
            out.append(GateRow(r.idx, r.t_sec, r.gt, int(stable_id), conf, 1, "STABLE"))
            continue

        if candidate_id == route_id:
            candidate_count += 1
        else:
            candidate_id = route_id
            candidate_count = 1

        if stable_id is None:
            if candidate_count >= int(stable_n):
                stable_id = route_id
                out.append(GateRow(r.idx, r.t_sec, r.gt, int(stable_id), conf, 1, "STABLE"))
            else:
                out.append(GateRow(r.idx, r.t_sec, r.gt, int(route_id), conf, 0, "ACQUIRE"))
            continue

        if route_id == stable_id:
            out.append(GateRow(r.idx, r.t_sec, r.gt, int(stable_id), conf, 1, "STABLE"))
        elif candidate_count >= int(stable_n):
            stable_id = route_id
            out.append(GateRow(r.idx, r.t_sec, r.gt, int(stable_id), conf, 1, "STABLE"))
        else:
            out.append(GateRow(r.idx, r.t_sec, r.gt, int(stable_id), conf, 0, "UNSURE"))

    return out


def _summarize(rows: List[GateRow], *, score_mask: Optional[List[bool]] = None) -> Dict[str, float]:
    n_all = int(len(rows))
    if n_all <= 0:
        return {"samples": 0.0, "samples_all": 0.0, "gate_switches": 0.0}

    if score_mask is None:
        score_mask = [True] * n_all
    if len(score_mask) != n_all:
        raise ValueError(f"score_mask length mismatch: {len(score_mask)} vs {n_all}")

    scored = [r for r, m in zip(rows, score_mask) if bool(m)]
    n = int(len(scored))
    switches = int(_count_switches([r.pred for r in rows]))
    if n <= 0:
        return {"samples": 0.0, "samples_all": float(n_all), "gate_switches": float(switches)}

    acc_all = float(sum(1 for r in scored if int(r.pred) == int(r.gt))) / float(n)
    valid_n = int(sum(1 for r in scored if int(r.valid) == 1))
    valid_ratio = float(valid_n) / float(n)
    acc_when_valid = (
        float(sum(1 for r in scored if int(r.valid) == 1 and int(r.pred) == int(r.gt))) / float(valid_n)
        if valid_n > 0
        else 0.0
    )
    return {
        "samples": float(n),
        "samples_all": float(n_all),
        "gate_acc_all": float(acc_all),
        "gate_valid_ratio": float(valid_ratio),
        "gate_acc_when_valid": float(acc_when_valid),
        "gate_switches": float(switches),
    }


def _write_gate_csv(path: Path, rows: List[GateRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "t_sec", "gt", "pred", "conf", "valid", "state"])
        for r in rows:
            w.writerow([int(r.idx), f"{float(r.t_sec):.6f}", int(r.gt), int(r.pred), f"{float(r.conf):.6f}", int(r.valid), str(r.state)])


def _write_report_md(path: Path, *, input_csv: Path, summary: Dict[str, float], params: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        "# Stage4 gate offline summary",
        "",
        f"- input_csv: `{input_csv}`",
        "",
        f"- samples: `{int(summary.get('samples', 0))}`",
        f"- samples_all: `{int(summary.get('samples_all', 0))}`",
        f"- gate_acc_all: `{summary.get('gate_acc_all', 0.0):.3f}`",
        f"- gate_valid_ratio: `{summary.get('gate_valid_ratio', 0.0):.3f}`",
        f"- gate_acc_when_valid: `{summary.get('gate_acc_when_valid', 0.0):.3f}`",
        f"- gate_switches: `{int(summary.get('gate_switches', 0))}`",
        "",
        "Gate params (simulated):",
    ]
    for k, v in params.items():
        lines.append(f"- {k}: `{v}`")
    lines += [
        "",
        "Notes:",
        "- `pred` column is the gate output id (simulated from per-frame raw predictions).",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Simulate orchard_corridor Stage4 route_id_gate on a predictions.csv (no ROS needed).")
    p.add_argument("--csv", required=True, help="predictions.csv produced by eval_route_id_on_bag.py")
    p.add_argument("--out-dir", required=True, help="Output directory (writes gate_predictions.csv + gate_report.md)")
    p.add_argument(
        "--score-idx-csv",
        default="",
        help="Optional predictions.csv whose idx set defines which samples to score (simulation still runs on --csv).",
    )

    p.add_argument("--conf-th", type=float, default=0.6)
    p.add_argument("--stable-n", type=int, default=3)
    p.add_argument("--timeout", type=float, default=2.0)
    p.add_argument("--allowed-jump", type=int, default=1)
    p.add_argument("--vote-window", type=int, default=0)
    p.add_argument("--vote-min-samples", type=int, default=3)
    p.add_argument("--vote-switch-ratio", type=float, default=0.7)
    p.add_argument("--unknown-id", type=int, default=-1)
    args = p.parse_args(argv)

    in_csv = Path(args.csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    rows = _read_predictions_csv(in_csv)

    score_mask: Optional[List[bool]] = None
    score_idx_csv = str(args.score_idx_csv).strip()
    if score_idx_csv:
        score_idx_path = Path(score_idx_csv).expanduser().resolve()
        score_idx_set = _read_idx_set_from_predictions_csv(score_idx_path)
        score_mask = [int(r.idx) in score_idx_set for r in rows]

    gate_rows = _simulate_gate(
        rows,
        conf_th=float(args.conf_th),
        stable_n=max(1, int(args.stable_n)),
        timeout_s=float(args.timeout),
        allowed_jump=int(args.allowed_jump),
        vote_window=max(0, int(args.vote_window)),
        vote_min_samples=max(1, int(args.vote_min_samples)),
        vote_switch_ratio=float(args.vote_switch_ratio),
        unknown_id=int(args.unknown_id),
    )

    out_csv = out_dir / "gate_predictions.csv"
    out_md = out_dir / "gate_report.md"

    _write_gate_csv(out_csv, gate_rows)
    summary = _summarize(gate_rows, score_mask=score_mask)
    params = {
        "conf_th": str(args.conf_th),
        "stable_N": str(args.stable_n),
        "allowed_jump": str(args.allowed_jump),
        "timeout": str(args.timeout),
        "vote_window": str(args.vote_window),
        "vote_min_samples": str(args.vote_min_samples),
        "vote_switch_ratio": str(args.vote_switch_ratio),
    }
    if score_idx_csv:
        params["score_idx_csv"] = str(Path(score_idx_csv).expanduser().resolve())
    _write_report_md(out_md, input_csv=out_csv, summary=summary, params=params)

    print(f"[OK] wrote: {out_csv} and {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
