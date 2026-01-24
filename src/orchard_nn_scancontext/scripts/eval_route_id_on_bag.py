#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rosbag
from sensor_msgs import point_cloud2 as pc2

from orchard_nn_scancontext.scan_context import ScanContext
from orchard_nn_scancontext.sc_route_db import load_route_db, predict_route_id_cosine, predict_route_id_l2


def _downsample_xyz(points_xyz: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or points_xyz.shape[0] <= max_points:
        return points_xyz
    step = int(np.ceil(float(points_xyz.shape[0]) / float(max_points)))
    return points_xyz[::step]


def _cloud_to_xyz(msg) -> np.ndarray:
    points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
    if not points:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def _segment_bounds(total_msgs: int, num_classes: int) -> Tuple[int, List[Tuple[int, int]]]:
    per = int(total_msgs) // int(num_classes)
    if per <= 0:
        raise ValueError(f"invalid per={per} from total_msgs={total_msgs} num_classes={num_classes}")
    bounds: List[Tuple[int, int]] = []
    for seg in range(int(num_classes)):
        start = int(seg) * int(per)
        end = int(total_msgs) - 1 if seg == int(num_classes) - 1 else (int(seg) + 1) * int(per) - 1
        bounds.append((start, end))
    return per, bounds


def _mode_and_ratio(x: np.ndarray) -> Tuple[int, float]:
    if x.size == 0:
        return (-1, 0.0)
    vals, counts = np.unique(x.astype(np.int64, copy=False), return_counts=True)
    i = int(np.argmax(counts))
    return int(vals[i]), float(counts[i]) / float(x.size)


@dataclass(frozen=True)
class EvalRow:
    idx: int
    t_sec: float
    gt: int
    pred: int
    conf: float


def _write_report_md(
    path: Path,
    *,
    bag_path: Path,
    cloud_topic: str,
    method: str,
    total_msgs: int,
    per: int,
    bounds: List[Tuple[int, int]],
    rows: List[EvalRow],
) -> None:
    gts = np.asarray([r.gt for r in rows], dtype=np.int64)
    preds = np.asarray([r.pred for r in rows], dtype=np.int64)
    confs = np.asarray([r.conf for r in rows], dtype=np.float64)

    k = len(bounds)
    n = int(gts.size)
    acc = float(np.mean(preds == gts)) if n > 0 else 0.0
    err = np.abs(preds - gts)
    within1 = float(np.mean(err <= 1)) if n > 0 else 0.0
    within2 = float(np.mean(err <= 2)) if n > 0 else 0.0

    lines: List[str] = [
        "# route_id 离线评估（按帧数均分段号）",
        "",
        f"- rosbag: `{bag_path}`",
        f"- topic: `{cloud_topic}`",
        f"- method: `{method}`",
        f"- total_msgs: `{total_msgs}`",
        f"- K(num_classes): `{k}`",
        f"- per(segment_size): `{per}`",
        f"- eval_samples: `{n}`",
        "",
        f"- Top1 acc: `{acc:.3f}`",
        f"- |pred-gt|<=1: `{within1:.3f}`",
        f"- |pred-gt|<=2: `{within2:.3f}`",
        "",
        "## Per-Segment",
        "| gt(route_id) | idx_start | idx_end | N | mode(pred) | mode_ratio | acc | conf_mean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for seg, (i0, i1) in enumerate(bounds):
        mask = gts == int(seg)
        seg_preds = preds[mask]
        seg_confs = confs[mask]
        seg_n = int(seg_preds.size)
        if seg_n <= 0:
            lines.append(f"| {seg} | {i0} | {i1} | 0 | -1 | 0.000 | 0.000 | 0.000 |")
            continue
        mode, ratio = _mode_and_ratio(seg_preds)
        seg_acc = float(np.mean(seg_preds == int(seg)))
        conf_mean = float(np.mean(seg_confs)) if seg_confs.size > 0 else 0.0
        lines.append(f"| {seg} | {i0} | {i1} | {seg_n} | {mode} | {ratio:.3f} | {seg_acc:.3f} | {conf_mean:.3f} |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_rows_csv(path: Path, rows: List[EvalRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "t_sec", "gt", "pred", "conf"])
        for r in rows:
            w.writerow([int(r.idx), f"{float(r.t_sec):.6f}", int(r.gt), int(r.pred), f"{float(r.conf):.6f}"])


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Offline route_id evaluation on a rosbag (GT=uniform segments by frame count).")
    p.add_argument("--bag", required=True, help="Input .bag path")
    p.add_argument("--cloud-topic", required=True, help="sensor_msgs/PointCloud2 topic")
    p.add_argument("--out-dir", required=True, help="Output directory (writes report.md + predictions.csv)")
    p.add_argument("--process-hz", type=float, default=2.0)
    p.add_argument("--max-points", type=int, default=60000)
    p.add_argument("--sample-every", type=int, default=1)
    p.add_argument("--start-offset", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=0.0)

    p.add_argument("--method", choices=["db", "nn"], default="db")
    p.add_argument("--num-classes", type=int, default=20, help="Only used for --method nn")

    p.add_argument("--db", default="", help="Route DB .npz (required for --method db)")
    p.add_argument("--metric", choices=["cosine", "l2"], default="cosine")
    p.add_argument("--temperature", type=float, default=0.02)

    p.add_argument("--model-path", default="", help="NN weights .pth (required for --method nn)")
    p.add_argument("--model-type", default="simple2dcnn", choices=["simple2dcnn", "enhanced2dcnn", "resnet2d"])
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    p.add_argument("--num-ring", type=int, default=20)
    p.add_argument("--num-sector", type=int, default=60)
    p.add_argument("--min-range", type=float, default=0.1)
    p.add_argument("--max-range", type=float, default=80.0)
    p.add_argument("--height-lower", type=float, default=-1.0)
    p.add_argument("--height-upper", type=float, default=9.0)
    args = p.parse_args(argv)

    bag_path = Path(args.bag).expanduser().resolve()
    if not bag_path.is_file():
        raise FileNotFoundError(f"bag not found: {bag_path}")
    cloud_topic = str(args.cloud_topic).strip()
    if not cloud_topic:
        raise ValueError("--cloud-topic is required")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    method = str(args.method)
    route_db = None
    if method == "db":
        db_arg = str(args.db).strip()
        if not db_arg:
            raise ValueError("--db is required for --method db")
        db_path = Path(db_arg).expanduser().resolve()
        route_db = load_route_db(db_path)
        num_classes = int(route_db.prototypes.shape[0])
        sc = ScanContext(
            num_sectors=int(route_db.params.num_sector),
            num_rings=int(route_db.params.num_ring),
            min_range=float(route_db.params.min_range),
            max_range=float(route_db.params.max_range),
            height_lower_bound=float(route_db.params.height_lower_bound),
            height_upper_bound=float(route_db.params.height_upper_bound),
        )
        model = None
        device = None
    else:
        num_classes = int(args.num_classes)
        sc = ScanContext(
            num_sectors=int(args.num_sector),
            num_rings=int(args.num_ring),
            min_range=float(args.min_range),
            max_range=float(args.max_range),
            height_lower_bound=float(args.height_lower),
            height_upper_bound=float(args.height_upper),
        )

        import torch

        from orchard_nn_scancontext.cnn_2d_models import Enhanced2DCNN, ResNet2D, Simple2DCNN

        model_arg = str(args.model_path).strip()
        if not model_arg:
            pkg_root = Path(__file__).resolve().parents[1]
            model_path = pkg_root / "models" / "trajectory_localizer_simple2dcnn_acc97.5.pth"
        else:
            model_path = Path(model_arg).expanduser()
        model_path = model_path.resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"model not found: {model_path}")

        device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
        model_type = str(args.model_type).strip().lower()
        if model_type == "simple2dcnn":
            model = Simple2DCNN(num_classes=num_classes)
        elif model_type == "enhanced2dcnn":
            model = Enhanced2DCNN(num_classes=num_classes)
        elif model_type == "resnet2d":
            model = ResNet2D(num_classes=num_classes)
        else:
            raise ValueError(f"unknown model_type: {model_type}")

        ckpt = torch.load(str(model_path), map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        model = model.to(device)
        model.eval()

    with rosbag.Bag(str(bag_path)) as bag:
        info = bag.get_type_and_topic_info()
        if cloud_topic not in info.topics:
            raise RuntimeError(f"topic not found in bag: {cloud_topic}")
        total_msgs = int(getattr(info.topics[cloud_topic], "message_count", 0))
        if total_msgs <= 0:
            raise RuntimeError(f"invalid message_count for topic {cloud_topic}: {total_msgs}")

        per, bounds = _segment_bounds(total_msgs, int(num_classes))

        bag_start = float(bag.get_start_time())
        bag_end = float(bag.get_end_time())
        start_time = bag_start + float(args.start_offset)
        end_time = bag_end if float(args.duration) <= 0.0 else start_time + float(args.duration)

        min_dt = 0.0 if float(args.process_hz) <= 0.0 else 1.0 / float(args.process_hz)
        last_proc_t: Optional[float] = None

        idx = -1
        msg_idx = 0
        rows: List[EvalRow] = []
        for _topic, msg, t in bag.read_messages(topics=[cloud_topic]):
            idx += 1
            t_sec = float(t.to_sec())
            if t_sec < start_time:
                continue
            if t_sec > end_time:
                break

            msg_idx += 1
            if int(args.sample_every) > 1 and (msg_idx - 1) % int(args.sample_every) != 0:
                continue
            if last_proc_t is not None and min_dt > 0.0 and (t_sec - float(last_proc_t)) < float(min_dt):
                continue
            last_proc_t = t_sec

            gt = min(int(num_classes) - 1, int(idx // per))

            xyz = _cloud_to_xyz(msg)
            if xyz.size == 0:
                rows.append(EvalRow(idx=int(idx), t_sec=float(t_sec), gt=int(gt), pred=-1, conf=0.0))
                continue
            xyz = _downsample_xyz(xyz, int(args.max_points))
            desc = sc.generate_scan_context(xyz)

            if method == "db":
                assert route_db is not None
                if str(args.metric) == "cosine":
                    pred, conf = predict_route_id_cosine(desc, route_db, temperature=float(args.temperature))
                else:
                    pred, conf = predict_route_id_l2(desc, route_db, temperature=float(args.temperature))
            else:
                import torch

                assert model is not None and device is not None
                x = torch.from_numpy(desc).float().unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)
                    conf_t, pred_t = torch.max(probs, dim=1)
                pred = int(pred_t.item())
                conf = float(conf_t.item())

            rows.append(EvalRow(idx=int(idx), t_sec=float(t_sec), gt=int(gt), pred=int(pred), conf=float(conf)))

    _write_rows_csv(out_dir / "predictions.csv", rows)
    _write_report_md(
        out_dir / "report.md",
        bag_path=bag_path,
        cloud_topic=cloud_topic,
        method=method,
        total_msgs=total_msgs,
        per=per,
        bounds=bounds,
        rows=rows,
    )

    print(f"[OK] wrote: {out_dir}/report.md and {out_dir}/predictions.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
