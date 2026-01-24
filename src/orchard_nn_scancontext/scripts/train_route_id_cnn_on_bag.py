#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rosbag
import sensor_msgs.point_cloud2 as pc2
import torch
from torch.utils.data import DataLoader, TensorDataset

from orchard_nn_scancontext.cnn_2d_models import Enhanced2DCNN, ResNet2D, Simple2DCNN
from orchard_nn_scancontext.scan_context import ScanContext


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


def _make_model(model_type: str, num_classes: int) -> torch.nn.Module:
    model_type = (model_type or "simple2dcnn").strip().lower()
    if model_type == "simple2dcnn":
        return Simple2DCNN(num_classes=int(num_classes))
    if model_type == "enhanced2dcnn":
        return Enhanced2DCNN(num_classes=int(num_classes))
    if model_type == "resnet2d":
        return ResNet2D(num_classes=int(num_classes))
    raise ValueError(f"unknown model_type: {model_type}")


@dataclass(frozen=True)
class DatasetMeta:
    bag_path: Path
    cloud_topic: str
    total_msgs: int
    per: int
    bounds: List[Tuple[int, int]]
    num_classes: int


def _load_dataset_from_bag(
    *,
    bag_path: Path,
    cloud_topic: str,
    num_classes: int,
    sc: ScanContext,
    process_hz: float,
    max_points: int,
    sample_every: int,
    start_offset: float,
    duration: float,
    limit: int,
) -> Tuple[DatasetMeta, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        start_time = bag_start + float(start_offset)
        end_time = bag_end if float(duration) <= 0.0 else start_time + float(duration)

        min_dt = 0.0 if float(process_hz) <= 0.0 else 1.0 / float(process_hz)
        last_proc_t: Optional[float] = None

        idx = -1
        msg_idx = 0

        idxs: List[int] = []
        t_secs: List[float] = []
        labels: List[int] = []
        descs: List[np.ndarray] = []

        t0 = time.time()
        for _topic, msg, t in bag.read_messages(topics=[cloud_topic]):
            idx += 1
            t_sec = float(t.to_sec())
            if t_sec < start_time:
                continue
            if t_sec > end_time:
                break

            msg_idx += 1
            if int(sample_every) > 1 and (msg_idx - 1) % int(sample_every) != 0:
                continue
            if last_proc_t is not None and min_dt > 0.0 and (t_sec - float(last_proc_t)) < float(min_dt):
                continue
            last_proc_t = t_sec

            xyz = _cloud_to_xyz(msg)
            if xyz.size == 0:
                continue
            xyz = _downsample_xyz(xyz, int(max_points))
            desc = sc.generate_scan_context(xyz)

            gt = min(int(num_classes) - 1, int(idx // per))
            idxs.append(int(idx))
            t_secs.append(float(t_sec))
            labels.append(int(gt))
            descs.append(desc.astype(np.float32, copy=False))

            if int(limit) > 0 and len(idxs) >= int(limit):
                break

        dt = time.time() - t0
        print(f"[INFO] loaded {len(idxs)} samples from bag in {dt:.1f}s (process_hz={process_hz})")

    meta = DatasetMeta(
        bag_path=bag_path,
        cloud_topic=cloud_topic,
        total_msgs=int(total_msgs),
        per=int(per),
        bounds=bounds,
        num_classes=int(num_classes),
    )
    return (
        meta,
        np.asarray(idxs, dtype=np.int64),
        np.asarray(t_secs, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(descs, dtype=np.float32),
    )


def _stratified_split(
    labels: np.ndarray,
    *,
    num_classes: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if float(val_ratio) < 0.0 or float(test_ratio) < 0.0 or (float(val_ratio) + float(test_ratio)) >= 1.0:
        raise ValueError(f"invalid split ratios: val_ratio={val_ratio} test_ratio={test_ratio}")
    rng = np.random.RandomState(int(seed))

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for c in range(int(num_classes)):
        idxs = np.nonzero(labels == int(c))[0]
        if idxs.size == 0:
            continue
        rng.shuffle(idxs)
        n = int(idxs.size)
        n_test = int(round(float(test_ratio) * float(n)))
        n_val = int(round(float(val_ratio) * float(n)))
        test_idx.extend(idxs[:n_test].tolist())
        val_idx.extend(idxs[n_test : n_test + n_val].tolist())
        train_idx.extend(idxs[n_test + n_val :].tolist())

    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(val_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    )


def _contiguous_split(
    labels: np.ndarray,
    *,
    num_classes: int,
    idx_order: np.ndarray,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if idx_order.shape[0] != labels.shape[0]:
        raise ValueError(f"idx_order must match labels: {idx_order.shape} vs {labels.shape}")
    if float(val_ratio) < 0.0 or float(test_ratio) < 0.0 or (float(val_ratio) + float(test_ratio)) >= 1.0:
        raise ValueError(f"invalid split ratios: val_ratio={val_ratio} test_ratio={test_ratio}")

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for c in range(int(num_classes)):
        idxs = np.nonzero(labels == int(c))[0]
        if idxs.size == 0:
            continue

        # Keep chronological order within each class (segment) to reduce near-identical frame leakage.
        idxs = idxs[np.argsort(idx_order[idxs], kind="stable")]

        n = int(idxs.size)
        n_test = int(round(float(test_ratio) * float(n)))
        n_val = int(round(float(val_ratio) * float(n)))

        if n_test + n_val >= n:
            n_val = max(0, n - 1 - n_test)
        if n_test >= n:
            n_test = max(0, n - 1)

        test_idx.extend(idxs[:n_test].tolist())
        val_idx.extend(idxs[n_test : n_test + n_val].tolist())
        train_idx.extend(idxs[n_test + n_val :].tolist())

    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(val_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    )


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    return float((pred == y).float().mean().item())


def _eval_loop(model: torch.nn.Module, loader: DataLoader, loss_fn: torch.nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_n = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            bs = int(y.shape[0])
            total_loss += float(loss.item()) * bs
            total_n += bs
            correct += int((torch.argmax(logits, dim=1) == y).sum().item())
    if total_n <= 0:
        return 0.0, 0.0
    return total_loss / float(total_n), float(correct) / float(total_n)


def _write_train_history(path: Path, rows: List[Tuple[int, float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for epoch, tl, ta, vl, va in rows:
            w.writerow([int(epoch), f"{float(tl):.6f}", f"{float(ta):.6f}", f"{float(vl):.6f}", f"{float(va):.6f}"])


def _write_eval_predictions_csv(path: Path, *, idxs: np.ndarray, t_secs: np.ndarray, gts: np.ndarray, preds: np.ndarray, confs: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "t_sec", "gt", "pred", "conf"])
        for i in range(int(idxs.size)):
            w.writerow([int(idxs[i]), f"{float(t_secs[i]):.6f}", int(gts[i]), int(preds[i]), f"{float(confs[i]):.6f}"])


def _mode_and_ratio(x: np.ndarray) -> Tuple[int, float]:
    if x.size == 0:
        return (-1, 0.0)
    vals, counts = np.unique(x.astype(np.int64, copy=False), return_counts=True)
    i = int(np.argmax(counts))
    return int(vals[i]), float(counts[i]) / float(x.size)


def _write_eval_report_md(
    path: Path,
    *,
    meta: DatasetMeta,
    model_type: str,
    model_path: Path,
    process_hz: float,
    rows: List[Tuple[int, float, int, int, float]],
) -> None:
    gts = np.asarray([r[2] for r in rows], dtype=np.int64)
    preds = np.asarray([r[3] for r in rows], dtype=np.int64)
    confs = np.asarray([r[4] for r in rows], dtype=np.float64)
    n = int(gts.size)

    acc = float(np.mean(preds == gts)) if n > 0 else 0.0
    err = np.abs(preds - gts)
    within1 = float(np.mean(err <= 1)) if n > 0 else 0.0
    within2 = float(np.mean(err <= 2)) if n > 0 else 0.0

    lines: List[str] = [
        "# route_id 离线评估（按帧数均分段号）",
        "",
        f"- rosbag: `{meta.bag_path}`",
        f"- topic: `{meta.cloud_topic}`",
        f"- method: `nn`",
        f"- model_type: `{model_type}`",
        f"- model_path: `{model_path}`",
        f"- total_msgs: `{meta.total_msgs}`",
        f"- K(num_classes): `{meta.num_classes}`",
        f"- per(segment_size): `{meta.per}`",
        f"- eval_samples: `{n}`",
        f"- process_hz: `{process_hz}`",
        "",
        f"- Top1 acc: `{acc:.3f}`",
        f"- |pred-gt|<=1: `{within1:.3f}`",
        f"- |pred-gt|<=2: `{within2:.3f}`",
        "",
        "## Per-Segment",
        "| gt(route_id) | idx_start | idx_end | N | mode(pred) | mode_ratio | acc | conf_mean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for seg, (i0, i1) in enumerate(meta.bounds):
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


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Train a 2D-CNN ScanContext classifier on a rosbag (GT=uniform segments by frame count).")
    p.add_argument("--bag", required=True, help="Input .bag path")
    p.add_argument("--cloud-topic", required=True, help="sensor_msgs/PointCloud2 topic")
    p.add_argument("--out-dir", required=True, help="Output dir (writes model_best.pth + train_history.csv + eval/predictions.csv)")

    p.add_argument("--model-type", default="simple2dcnn", choices=["simple2dcnn", "enhanced2dcnn", "resnet2d"])
    p.add_argument("--num-classes", type=int, default=20)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.0)

    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--test-ratio", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-mode", default="stratified", choices=["stratified", "contiguous"], help="Validation split strategy")

    p.add_argument("--process-hz", type=float, default=2.0)
    p.add_argument("--max-points", type=int, default=60000)
    p.add_argument("--sample-every", type=int, default=1)
    p.add_argument("--start-offset", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=0, help="Limit number of processed samples (debug)")

    p.add_argument("--num-ring", type=int, default=20)
    p.add_argument("--num-sector", type=int, default=60)
    p.add_argument("--min-range", type=float, default=0.1)
    p.add_argument("--max-range", type=float, default=80.0)
    p.add_argument("--height-lower", type=float, default=-1.0)
    p.add_argument("--height-upper", type=float, default=9.0)

    p.add_argument("--init-from", default="", help="Optional checkpoint .pth to initialize weights (fine-tune)")
    p.add_argument("--eval-softmax-temperature", type=float, default=1.0)
    args = p.parse_args(argv)

    bag_path = Path(args.bag).expanduser().resolve()
    if not bag_path.is_file():
        raise FileNotFoundError(f"bag not found: {bag_path}")
    cloud_topic = str(args.cloud_topic).strip()
    if not cloud_topic:
        raise ValueError("--cloud-topic is required")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sc = ScanContext(
        num_sectors=int(args.num_sector),
        num_rings=int(args.num_ring),
        min_range=float(args.min_range),
        max_range=float(args.max_range),
        height_lower_bound=float(args.height_lower),
        height_upper_bound=float(args.height_upper),
    )

    meta, idxs, t_secs, labels, descs = _load_dataset_from_bag(
        bag_path=bag_path,
        cloud_topic=cloud_topic,
        num_classes=int(args.num_classes),
        sc=sc,
        process_hz=float(args.process_hz),
        max_points=int(args.max_points),
        sample_every=int(args.sample_every),
        start_offset=float(args.start_offset),
        duration=float(args.duration),
        limit=int(args.limit),
    )
    if descs.size == 0:
        raise RuntimeError("empty dataset: no non-empty pointclouds after filtering")

    if str(args.split_mode) == "stratified":
        train_idx, val_idx, _test_idx = _stratified_split(
            labels,
            num_classes=int(args.num_classes),
            val_ratio=float(args.val_ratio),
            test_ratio=float(args.test_ratio),
            seed=int(args.seed),
        )
    else:
        train_idx, val_idx, _test_idx = _contiguous_split(
            labels,
            num_classes=int(args.num_classes),
            idx_order=idxs,
            val_ratio=float(args.val_ratio),
            test_ratio=float(args.test_ratio),
        )
    if train_idx.size == 0 or val_idx.size == 0:
        raise RuntimeError(f"invalid split: train={train_idx.size} val={val_idx.size}")

    x = torch.from_numpy(descs).float().unsqueeze(1)  # (N,1,20,60)
    y = torch.from_numpy(labels).long()

    train_ds = TensorDataset(x[train_idx], y[train_idx])
    val_ds = TensorDataset(x[val_idx], y[val_idx])

    batch_size = max(1, int(args.batch_size))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if str(args.device) == "cuda" and torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    model = _make_model(str(args.model_type), int(args.num_classes)).to(device)
    init_from = str(args.init_from).strip()
    if init_from:
        init_path = Path(init_from).expanduser().resolve()
        if not init_path.is_file():
            raise FileNotFoundError(f"--init-from not found: {init_path}")
        ckpt_init = torch.load(str(init_path), map_location=device)
        state = ckpt_init["model_state_dict"] if isinstance(ckpt_init, dict) and "model_state_dict" in ckpt_init else ckpt_init
        model.load_state_dict(state, strict=True)
        print(f"[INFO] initialized weights from: {init_path}")
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    history: List[Tuple[int, float, float, float, float]] = []
    best_val_acc = -1.0
    best_path = out_dir / "model_best.pth"

    t0 = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_loss = 0.0
        train_n = 0
        train_correct = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            bs = int(yb.shape[0])
            train_loss += float(loss.item()) * bs
            train_n += bs
            train_correct += int((torch.argmax(logits, dim=1) == yb).sum().item())

        train_loss = train_loss / float(max(train_n, 1))
        train_acc = float(train_correct) / float(max(train_n, 1))
        val_loss, val_acc = _eval_loop(model, val_loader, loss_fn, device)
        history.append((int(epoch), float(train_loss), float(train_acc), float(val_loss), float(val_acc)))

        print(f"[E{epoch:03d}] train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_type": str(args.model_type),
                    "num_classes": int(args.num_classes),
                    "scan_context": {
                        "num_ring": int(args.num_ring),
                        "num_sector": int(args.num_sector),
                        "min_range": float(args.min_range),
                        "max_range": float(args.max_range),
                        "height_lower_bound": float(args.height_lower),
                        "height_upper_bound": float(args.height_upper),
                    },
                    "train": {
                        "epochs": int(args.epochs),
                        "batch_size": int(args.batch_size),
                        "lr": float(args.lr),
                        "weight_decay": float(args.weight_decay),
                        "label_smoothing": float(args.label_smoothing),
                        "seed": int(args.seed),
                    },
                    "data": {
                        "bag": str(bag_path),
                        "cloud_topic": str(cloud_topic),
                        "total_msgs": int(meta.total_msgs),
                        "per": int(meta.per),
                        "process_hz": float(args.process_hz),
                        "max_points": int(args.max_points),
                    },
                },
                str(best_path),
            )

    _write_train_history(out_dir / "train_history.csv", history)
    print(f"[OK] best_val_acc={best_val_acc:.3f} model={best_path}")
    print(f"[INFO] training time: {(time.time()-t0):.1f}s")

    # Evaluate on the same dataset in bag order (useful for plotting + Stage4 gate simulation).
    ckpt = torch.load(str(best_path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
    model.eval()
    temp = max(1e-6, float(args.eval_softmax_temperature))

    preds = np.zeros((int(labels.size),), dtype=np.int64)
    confs = np.zeros((int(labels.size),), dtype=np.float32)

    eval_loader = DataLoader(TensorDataset(x, y), batch_size=256, shuffle=False, drop_last=False)
    cursor = 0
    with torch.no_grad():
        for xb, _yb in eval_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits / float(temp), dim=1)
            conf_t, pred_t = torch.max(probs, dim=1)
            bs = int(pred_t.shape[0])
            preds[cursor : cursor + bs] = pred_t.cpu().numpy().astype(np.int64, copy=False)
            confs[cursor : cursor + bs] = conf_t.cpu().numpy().astype(np.float32, copy=False)
            cursor += bs

    eval_dir = out_dir / "eval"
    pred_csv = eval_dir / "predictions.csv"
    rep_md = eval_dir / "report.md"
    _write_eval_predictions_csv(pred_csv, idxs=idxs, t_secs=t_secs, gts=labels, preds=preds, confs=confs)
    all_rows = [(int(idxs[i]), float(t_secs[i]), int(labels[i]), int(preds[i]), float(confs[i])) for i in range(int(labels.size))]
    _write_eval_report_md(rep_md, meta=meta, model_type=str(args.model_type), model_path=best_path, process_hz=float(args.process_hz), rows=all_rows)
    print(f"[OK] wrote: {pred_csv} and {rep_md}")

    def write_subset(name: str, subset_idx: np.ndarray) -> None:
        if subset_idx.size <= 0:
            return
        order = np.argsort(idxs[subset_idx], kind="stable")
        si = subset_idx[order]
        sub_dir = out_dir / f"eval_{name}"
        sub_csv = sub_dir / "predictions.csv"
        sub_md = sub_dir / "report.md"
        _write_eval_predictions_csv(sub_csv, idxs=idxs[si], t_secs=t_secs[si], gts=labels[si], preds=preds[si], confs=confs[si])
        sub_rows = [(int(idxs[j]), float(t_secs[j]), int(labels[j]), int(preds[j]), float(confs[j])) for j in si.tolist()]
        _write_eval_report_md(sub_md, meta=meta, model_type=str(args.model_type), model_path=best_path, process_hz=float(args.process_hz), rows=sub_rows)
        print(f"[OK] wrote: {sub_csv} and {sub_md}")

    write_subset("train", train_idx)
    write_subset("val", val_idx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
