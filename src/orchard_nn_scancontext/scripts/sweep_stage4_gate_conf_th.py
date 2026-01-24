#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from eval_stage4_gate_on_csv import _read_predictions_csv, _simulate_gate, _summarize


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Sweep conf_th for Stage4 gate simulation on a predictions.csv.")
    p.add_argument("--csv", required=True, help="predictions.csv produced by eval_route_id_on_bag.py")
    p.add_argument(
        "--score-idx-csv",
        default="",
        help="Optional predictions.csv whose idx set defines which samples to score (simulation still runs on --csv).",
    )
    p.add_argument("--conf-th-list", default="", help='Comma-separated list, e.g. "0.3,0.4,0.5,0.6,0.7"')
    p.add_argument("--conf-th-min", type=float, default=0.2)
    p.add_argument("--conf-th-max", type=float, default=0.9)
    p.add_argument("--conf-th-step", type=float, default=0.05)

    p.add_argument("--stable-n", type=int, default=3)
    p.add_argument("--timeout", type=float, default=2.0)
    p.add_argument("--allowed-jump", type=int, default=1)
    p.add_argument("--vote-window", type=int, default=0)
    p.add_argument("--vote-min-samples", type=int, default=3)
    p.add_argument("--vote-switch-ratio", type=float, default=0.7)
    p.add_argument("--unknown-id", type=int, default=-1)
    args = p.parse_args(argv)

    in_csv = Path(args.csv).expanduser().resolve()
    rows = _read_predictions_csv(in_csv)

    score_mask = None
    score_idx_csv = str(args.score_idx_csv).strip()
    if score_idx_csv:
        score_rows = _read_predictions_csv(Path(score_idx_csv).expanduser().resolve())
        score_idx_set = {int(r.idx) for r in score_rows}
        score_mask = [int(r.idx) in score_idx_set for r in rows]

    conf_list: List[float] = []
    if str(args.conf_th_list).strip():
        for x in str(args.conf_th_list).split(","):
            x = x.strip()
            if not x:
                continue
            conf_list.append(float(x))
    else:
        v = float(args.conf_th_min)
        vmax = float(args.conf_th_max)
        step = max(1e-6, float(args.conf_th_step))
        while v <= vmax + 1e-9:
            conf_list.append(v)
            v += step

    print("conf_th,gate_acc_all,gate_valid_ratio,gate_acc_when_valid,gate_switches")
    for conf_th in conf_list:
        gate_rows = _simulate_gate(
            rows,
            conf_th=float(conf_th),
            stable_n=max(1, int(args.stable_n)),
            timeout_s=float(args.timeout),
            allowed_jump=int(args.allowed_jump),
            vote_window=max(0, int(args.vote_window)),
            vote_min_samples=max(1, int(args.vote_min_samples)),
            vote_switch_ratio=float(args.vote_switch_ratio),
            unknown_id=int(args.unknown_id),
        )
        s = _summarize(gate_rows, score_mask=score_mask)
        print(
            f"{conf_th:.3f},{s.get('gate_acc_all',0.0):.3f},{s.get('gate_valid_ratio',0.0):.3f},{s.get('gate_acc_when_valid',0.0):.3f},{int(s.get('gate_switches',0))}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
