#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import rosbag


def _norm_frame(frame_id: str) -> str:
    return str(frame_id or "").strip().lstrip("/")


def _yaw_from_quat(q) -> float:
    x = float(getattr(q, "x", 0.0))
    y = float(getattr(q, "y", 0.0))
    z = float(getattr(q, "z", 0.0))
    w = float(getattr(q, "w", 1.0))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


def _lookup_series(times: List[float], values: List[Tuple[float, float, float]], t: float) -> Tuple[float, float, float]:
    if not times:
        return (0.0, 0.0, 0.0)
    idx = bisect.bisect_right(times, float(t)) - 1
    idx = max(0, min(idx, len(times) - 1))
    return values[idx]


def _compose(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    ax, ay, ayaw = a
    bx, by, byaw = b
    c = float(math.cos(ayaw))
    s = float(math.sin(ayaw))
    x = float(ax) + c * float(bx) - s * float(by)
    y = float(ay) + s * float(bx) + c * float(by)
    yaw = float(ayaw) + float(byaw)
    return (x, y, yaw)


def _unwrap_angles(values: List[float]) -> List[float]:
    if not values:
        return []
    out = [float(values[0])]
    for v in values[1:]:
        prev = out[-1]
        dv = (float(v) - prev + math.pi) % (2.0 * math.pi) - math.pi
        out.append(prev + dv)
    return out


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


def _load_tf_series(
    bag_path: Path,
    tf_topic: str,
    parent_frame: str,
    child_frame: str,
) -> Tuple[List[float], List[Tuple[float, float, float]]]:
    parent_frame = _norm_frame(parent_frame)
    child_frame = _norm_frame(child_frame)
    if not parent_frame or not child_frame:
        raise ValueError("tf parent/child frame must be non-empty")

    times: List[float] = []
    vals: List[Tuple[float, float, float]] = []

    with rosbag.Bag(str(bag_path), "r") as bag:
        for _topic, msg, _t in bag.read_messages(topics=[tf_topic]):
            for tr in getattr(msg, "transforms", []) or []:
                parent = _norm_frame(getattr(getattr(tr, "header", None), "frame_id", ""))
                child = _norm_frame(getattr(tr, "child_frame_id", ""))
                if parent != parent_frame or child != child_frame:
                    continue
                stamp = getattr(getattr(tr, "header", None), "stamp", None)
                if stamp is None:
                    continue
                ts = float(stamp.to_sec())
                trans = tr.transform.translation
                rot = tr.transform.rotation
                times.append(ts)
                vals.append((float(trans.x), float(trans.y), _yaw_from_quat(rot)))

    if not times:
        raise RuntimeError(f"TF not found in bag: {parent_frame} -> {child_frame} on {tf_topic}")

    order = sorted(range(len(times)), key=times.__getitem__)
    return ([times[i] for i in order], [vals[i] for i in order])


def _load_times(bag_path: Path, topic: str, min_dt: float) -> List[float]:
    times: List[float] = []
    last_t: Optional[float] = None
    with rosbag.Bag(str(bag_path), "r") as bag:
        for _topic, msg, _t in bag.read_messages(topics=[topic]):
            hdr = getattr(msg, "header", None)
            stamp = getattr(hdr, "stamp", None)
            if stamp is None:
                continue
            ts = float(stamp.to_sec())
            if last_t is not None and min_dt > 0.0 and (ts - last_t) < float(min_dt):
                continue
            times.append(ts)
            last_t = ts
    if not times:
        raise RuntimeError(f"No header.stamp found on topic: {topic}")
    return times


@dataclass(frozen=True)
class FrameRow:
    idx: int
    t_rel: float
    x: float
    y: float
    yaw: float
    ds: float
    dt: float
    dyaw: float
    yaw_rate: float
    route_id: int


@dataclass(frozen=True)
class SegmentRow:
    route_id: int
    idx_start: int
    idx_end: int
    t_rel_start: float
    t_rel_end: float
    dist_m: float
    total_abs_dyaw: float
    net_dyaw: float
    max_abs_yaw_rate: float
    max_abs_yaw_rate_t_rel: float


def _write_rows_csv(path: Path, rows: List[FrameRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "t_rel", "x", "y", "yaw", "ds", "dt", "dyaw", "yaw_rate", "route_id"])
        for r in rows:
            w.writerow(
                [
                    int(r.idx),
                    f"{float(r.t_rel):.6f}",
                    f"{float(r.x):.6f}",
                    f"{float(r.y):.6f}",
                    f"{float(r.yaw):.6f}",
                    f"{float(r.ds):.6f}",
                    f"{float(r.dt):.6f}",
                    f"{float(r.dyaw):.6f}",
                    f"{float(r.yaw_rate):.6f}",
                    int(r.route_id),
                ]
            )


def _write_report_md(
    path: Path,
    *,
    bag_path: Path,
    times_topic: str,
    tf_topic: str,
    output_frame: str,
    odom_frame: str,
    base_frame: str,
    per: int,
    segments: List[SegmentRow],
    top_instants: List[FrameRow],
) -> None:
    lines: List[str] = [
        "# 拐弯检测（基于 /tf 的 base_link_est yaw 变化）",
        "",
        f"- rosbag: `{bag_path}`",
        f"- times_topic: `{times_topic}`",
        f"- tf_topic: `{tf_topic}`",
        f"- frames: `{output_frame}->{odom_frame}->{base_frame}`",
        f"- K(num_classes): `{len(segments)}`",
        f"- per(segment_size): `{per}`",
        "",
        "## Top Segments（按 total_abs_dyaw 排序）",
        "| route_id | idx_start | idx_end | t_rel_start | t_rel_end | dist_m | total_abs_dyaw(rad) | net_dyaw(rad) | max|yaw_rate|(rad/s) | t_rel@max |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for seg in sorted(segments, key=lambda s: s.total_abs_dyaw, reverse=True):
        lines.append(
            "| {rid} | {i0} | {i1} | {t0:.1f} | {t1:.1f} | {dist:.1f} | {tot:.2f} | {net:.2f} | {mr:.2f} | {mrt:.1f} |".format(
                rid=int(seg.route_id),
                i0=int(seg.idx_start),
                i1=int(seg.idx_end),
                t0=float(seg.t_rel_start),
                t1=float(seg.t_rel_end),
                dist=float(seg.dist_m),
                tot=float(seg.total_abs_dyaw),
                net=float(seg.net_dyaw),
                mr=float(seg.max_abs_yaw_rate),
                mrt=float(seg.max_abs_yaw_rate_t_rel),
            )
        )

    lines += [
        "",
        "## Top Instants（按 |yaw_rate| 排序）",
        "| idx | route_id | t_rel | |yaw_rate|(rad/s) | |dyaw|(rad) | ds(m) | dt(s) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in top_instants:
        lines.append(
            "| {idx} | {rid} | {t:.1f} | {yr:.2f} | {dy:.3f} | {ds:.3f} | {dt:.3f} |".format(
                idx=int(r.idx),
                rid=int(r.route_id),
                t=float(r.t_rel),
                yr=abs(float(r.yaw_rate)),
                dy=abs(float(r.dyaw)),
                ds=float(r.ds),
                dt=float(r.dt),
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Detect turn segments/instants from TF yaw changes (offline).")
    p.add_argument("--bag", required=True, help="Input .bag path")
    p.add_argument("--times-topic", default="/points_raw", help="Topic whose header.stamp defines the frame times")
    p.add_argument("--tf-topic", default="/tf")
    p.add_argument("--output-frame", default="map")
    p.add_argument("--odom-frame", default="odom_est")
    p.add_argument("--base-frame", default="base_link_est")
    p.add_argument("--num-classes", type=int, default=20, help="Used only for uniform-by-frame segmenting")
    p.add_argument("--min-dt", type=float, default=0.05, help="Ignore extremely small dt when ranking instants")
    p.add_argument("--min-ds", type=float, default=0.02, help="Ignore extremely small ds when ranking instants")
    p.add_argument("--out-dir", default="output/turn_analysis", help="Output directory")
    p.add_argument("--top-k", type=int, default=12, help="Top turning instants to list in report")
    args = p.parse_args(argv)

    bag_path = Path(args.bag).expanduser().resolve()
    if not bag_path.is_file():
        raise FileNotFoundError(f"bag not found: {bag_path}")

    times_topic = str(args.times_topic).strip()
    tf_topic = str(args.tf_topic).strip()

    output_frame = _norm_frame(args.output_frame)
    odom_frame = _norm_frame(args.odom_frame)
    base_frame = _norm_frame(args.base_frame)
    if not output_frame or not odom_frame or not base_frame:
        raise ValueError("output/odom/base frames must be non-empty")

    times = _load_times(bag_path, times_topic, min_dt=0.0)
    total_msgs = len(times)
    per, bounds = _segment_bounds(total_msgs, int(args.num_classes))
    t0 = float(times[0])

    tf1_t: List[float] = []
    tf1_v: List[Tuple[float, float, float]] = []
    if output_frame != odom_frame:
        tf1_t, tf1_v = _load_tf_series(bag_path, tf_topic, output_frame, odom_frame)
    tf2_t, tf2_v = _load_tf_series(bag_path, tf_topic, odom_frame, base_frame)

    xs: List[float] = []
    ys: List[float] = []
    yaws: List[float] = []
    for t in times:
        a = (0.0, 0.0, 0.0) if output_frame == odom_frame else _lookup_series(tf1_t, tf1_v, t)
        b = _lookup_series(tf2_t, tf2_v, t)
        x, y, yaw = _compose(a, b)
        xs.append(float(x))
        ys.append(float(y))
        yaws.append(float(yaw))

    yaws_u = _unwrap_angles(yaws)

    frames: List[FrameRow] = []
    for idx in range(total_msgs):
        route_id = min(int(args.num_classes) - 1, int(idx // per))
        if idx == 0:
            frames.append(
                FrameRow(
                    idx=0,
                    t_rel=float(times[0] - t0),
                    x=xs[0],
                    y=ys[0],
                    yaw=yaws_u[0],
                    ds=0.0,
                    dt=0.0,
                    dyaw=0.0,
                    yaw_rate=0.0,
                    route_id=route_id,
                )
            )
            continue

        dt = float(times[idx] - times[idx - 1])
        dx = float(xs[idx] - xs[idx - 1])
        dy = float(ys[idx] - ys[idx - 1])
        ds = float(math.hypot(dx, dy))
        dyaw = float(yaws_u[idx] - yaws_u[idx - 1])
        yaw_rate = float(dyaw / dt) if dt > 1e-6 else 0.0
        frames.append(
            FrameRow(
                idx=int(idx),
                t_rel=float(times[idx] - t0),
                x=float(xs[idx]),
                y=float(ys[idx]),
                yaw=float(yaws_u[idx]),
                ds=float(ds),
                dt=float(dt),
                dyaw=float(dyaw),
                yaw_rate=float(yaw_rate),
                route_id=route_id,
            )
        )

    segments: List[SegmentRow] = []
    for rid, (i0, i1) in enumerate(bounds):
        t_rel_start = float(times[int(i0)] - t0)
        t_rel_end = float(times[int(i1)] - t0)

        dist_m = 0.0
        total_abs_dyaw = 0.0
        max_abs_yaw_rate = 0.0
        max_abs_yaw_rate_t_rel = t_rel_start

        for idx in range(int(i0) + 1, int(i1) + 1):
            r = frames[idx]
            dist_m += float(r.ds)
            if float(r.ds) >= float(args.min_ds):
                total_abs_dyaw += abs(float(r.dyaw))
            if float(r.dt) >= float(args.min_dt) and float(r.ds) >= float(args.min_ds):
                if abs(float(r.yaw_rate)) > float(max_abs_yaw_rate):
                    max_abs_yaw_rate = abs(float(r.yaw_rate))
                    max_abs_yaw_rate_t_rel = float(r.t_rel)

        net_dyaw = float(yaws_u[int(i1)] - yaws_u[int(i0)])
        segments.append(
            SegmentRow(
                route_id=int(rid),
                idx_start=int(i0),
                idx_end=int(i1),
                t_rel_start=float(t_rel_start),
                t_rel_end=float(t_rel_end),
                dist_m=float(dist_m),
                total_abs_dyaw=float(total_abs_dyaw),
                net_dyaw=float(net_dyaw),
                max_abs_yaw_rate=float(max_abs_yaw_rate),
                max_abs_yaw_rate_t_rel=float(max_abs_yaw_rate_t_rel),
            )
        )

    instants = [r for r in frames if float(r.dt) >= float(args.min_dt) and float(r.ds) >= float(args.min_ds)]
    instants_sorted = sorted(instants, key=lambda r: abs(float(r.yaw_rate)), reverse=True)
    top_instants = instants_sorted[: max(1, int(args.top_k))]

    out_dir = Path(args.out_dir).expanduser().resolve() / f"{bag_path.stem}_{times_topic.strip('/').replace('/', '_')}_K{int(args.num_classes)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_rows_csv(out_dir / "turn_frames.csv", frames)
    _write_report_md(
        out_dir / "turn_report.md",
        bag_path=bag_path,
        times_topic=times_topic,
        tf_topic=tf_topic,
        output_frame=output_frame,
        odom_frame=odom_frame,
        base_frame=base_frame,
        per=per,
        segments=segments,
        top_instants=top_instants,
    )

    print(f"[OK] wrote: {out_dir}/turn_report.md and {out_dir}/turn_frames.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

