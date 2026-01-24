#!/usr/bin/env python3
"""Generate quick PNG previews (BEV) from PCD files (no GUI required).

Works well for the exported PCDs in this workspace:
  - raw frames:    x y z intensity
  - tree frames:   x y z
  - colored frames x y z intensity rgb label
  - clusters:      x y z rgb cluster
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PcdData:
    fields: List[str]
    data_mode: str
    points: int
    arrays: Dict[str, np.ndarray]


def _dtype_from_type_size(type_char: str, size: int) -> np.dtype:
    type_char = str(type_char).upper()
    size = int(size)
    if type_char == "F":
        if size == 4:
            return np.dtype("<f4")
        if size == 8:
            return np.dtype("<f8")
    if type_char == "U":
        if size == 1:
            return np.dtype("<u1")
        if size == 2:
            return np.dtype("<u2")
        if size == 4:
            return np.dtype("<u4")
        if size == 8:
            return np.dtype("<u8")
    if type_char == "I":
        if size == 1:
            return np.dtype("<i1")
        if size == 2:
            return np.dtype("<i2")
        if size == 4:
            return np.dtype("<i4")
        if size == 8:
            return np.dtype("<i8")
    raise ValueError(f"Unsupported PCD dtype: TYPE={type_char} SIZE={size}")


def _parse_label_colors(spec: str) -> Dict[int, Tuple[int, int, int]]:
    text = (spec or "").strip()
    if not text:
        return {0: (56, 188, 75), 1: (180, 180, 180)}
    out: Dict[int, Tuple[int, int, int]] = {}
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid label-colors item (missing ':'): {item!r}")
        k_str, v_str = item.split(":", 1)
        k = int(k_str.strip())
        parts = [p.strip() for p in v_str.split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError(f"Invalid label-colors RGB triple for label={k}: {v_str!r}")
        r, g, b = (int(parts[0]), int(parts[1]), int(parts[2]))
        out[k] = (r, g, b)
    return out


def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    h = float(h) % 1.0
    s = float(max(0.0, min(s, 1.0)))
    v = float(max(0.0, min(v, 1.0)))
    i = int(h * 6.0)
    f = h * 6.0 - float(i)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return int(round(r * 255.0)), int(round(g * 255.0)), int(round(b * 255.0))


def _cluster_id_to_rgb(cluster_id: int) -> Tuple[int, int, int]:
    phi = 0.618033988749895  # golden ratio
    h = (float(cluster_id) * phi) % 1.0
    return _hsv_to_rgb(h, 0.95, 1.0)


def _read_pcd(path: Path) -> PcdData:
    with path.open("rb") as handle:
        header: Dict[str, str] = {}
        data_mode: Optional[str] = None
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError(f"Invalid PCD header: {path}")
            decoded = line.decode("utf-8", errors="ignore").strip()
            if decoded and not decoded.startswith("#"):
                parts = decoded.split(maxsplit=1)
                if len(parts) == 2:
                    header[parts[0].upper()] = parts[1].strip()
            if decoded.upper().startswith("DATA"):
                parts = decoded.split()
                data_mode = parts[1].lower() if len(parts) >= 2 else "ascii"
                break

        fields = header.get("FIELDS", "").split()
        sizes = [int(x) for x in header.get("SIZE", "").split()] if header.get("SIZE") else []
        types = header.get("TYPE", "").split()
        counts = [int(x) for x in header.get("COUNT", "").split()] if header.get("COUNT") else [1] * len(fields)

        if not fields:
            raise RuntimeError(f"PCD missing FIELDS: {path}")
        if len(sizes) != len(fields) or len(types) != len(fields):
            raise RuntimeError(f"PCD header mismatch (FIELDS/SIZE/TYPE): {path}")
        if len(counts) != len(fields):
            raise RuntimeError(f"PCD header mismatch (FIELDS/COUNT): {path}")

        points = int(header.get("POINTS", "0") or "0")
        if points <= 0:
            width = int(header.get("WIDTH", "0") or "0")
            height = int(header.get("HEIGHT", "1") or "1")
            points = int(width) * int(height)
        points = max(0, int(points))

        if data_mode is None:
            raise RuntimeError(f"Missing DATA line in PCD: {path}")

        arrays: Dict[str, np.ndarray] = {}
        if points == 0:
            return PcdData(fields=list(fields), data_mode=str(data_mode), points=0, arrays=arrays)

        if data_mode == "ascii":
            mat = np.loadtxt(handle, dtype=np.float32)
            if mat.ndim == 1:
                mat = mat.reshape(1, -1)
            if mat.shape[1] < len(fields):
                raise RuntimeError(f"PCD ascii columns < fields: {path}")
            for idx, name in enumerate(fields):
                arrays[name] = mat[:, idx]
            return PcdData(fields=list(fields), data_mode=str(data_mode), points=int(mat.shape[0]), arrays=arrays)

        if data_mode != "binary":
            raise RuntimeError(f"Unsupported PCD DATA mode: {data_mode} ({path})")

        offsets: List[int] = []
        names: List[str] = []
        formats: List[np.dtype] = []
        offset = 0
        for name, size, typ, cnt in zip(fields, sizes, types, counts):
            cnt = int(cnt)
            if cnt != 1:
                raise ValueError(f"Unsupported PCD COUNT={cnt} for field={name} (only COUNT=1 supported)")
            offsets.append(int(offset))
            names.append(str(name))
            formats.append(_dtype_from_type_size(typ, size))
            offset += int(size) * cnt
        point_step = int(offset)

        dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": point_step})
        raw = handle.read(int(points) * int(point_step))
        if len(raw) < int(points) * int(point_step):
            raise RuntimeError(f"PCD data too short: {path}")
        struct = np.frombuffer(raw, dtype=dtype, count=int(points))
        for name in names:
            arrays[name] = struct[name]
        return PcdData(fields=list(fields), data_mode=str(data_mode), points=int(points), arrays=arrays)


def _finite_mask(*cols: np.ndarray) -> np.ndarray:
    if not cols:
        return np.zeros((0,), dtype=bool)
    mask = np.ones((cols[0].shape[0],), dtype=bool)
    for col in cols:
        mask &= np.isfinite(col.astype(np.float64, copy=False))
    return mask


def _auto_bounds(x: np.ndarray, y: np.ndarray, percentile: float, pad_frac: float) -> Tuple[float, float, float, float]:
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    p = float(max(0.0, min(percentile, 49.9)))
    if x.size == 0:
        return -10.0, 10.0, -10.0, 10.0
    if p > 0.0:
        xmin, xmax = np.percentile(x, [p, 100.0 - p]).tolist()
        ymin, ymax = np.percentile(y, [p, 100.0 - p]).tolist()
    else:
        xmin, xmax = float(np.min(x)), float(np.max(x))
        ymin, ymax = float(np.min(y)), float(np.max(y))

    dx = max(1.0e-6, float(xmax) - float(xmin))
    dy = max(1.0e-6, float(ymax) - float(ymin))
    pad = float(max(0.0, pad_frac))
    xmin -= dx * pad
    xmax += dx * pad
    ymin -= dy * pad
    ymax += dy * pad
    return float(xmin), float(xmax), float(ymin), float(ymax)


def _xy_from_arrays(arrays: Dict[str, np.ndarray], plane: str) -> Tuple[np.ndarray, np.ndarray]:
    plane = str(plane).lower()
    if plane == "xy":
        return arrays["x"].astype(np.float32, copy=False), arrays["y"].astype(np.float32, copy=False)
    if plane == "xz":
        return arrays["x"].astype(np.float32, copy=False), arrays["z"].astype(np.float32, copy=False)
    if plane == "yz":
        return arrays["y"].astype(np.float32, copy=False), arrays["z"].astype(np.float32, copy=False)
    raise ValueError(f"Unsupported --plane {plane!r} (use xy/xz/yz)")


def _project_to_pixels(
    x: np.ndarray,
    y: np.ndarray,
    bounds: Tuple[float, float, float, float],
    size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax = bounds
    w = h = int(size)
    dx = max(1.0e-6, float(xmax) - float(xmin))
    dy = max(1.0e-6, float(ymax) - float(ymin))
    px = ((x.astype(np.float64, copy=False) - float(xmin)) / dx * float(w - 1)).astype(np.int32, copy=False)
    py = ((float(ymax) - y.astype(np.float64, copy=False)) / dy * float(h - 1)).astype(np.int32, copy=False)
    px = np.clip(px, 0, w - 1)
    py = np.clip(py, 0, h - 1)
    return px, py


def _render_density(px: np.ndarray, py: np.ndarray, size: int) -> np.ndarray:
    w = h = int(size)
    acc = np.zeros((h, w), dtype=np.uint32)
    if px.size:
        np.add.at(acc, (py, px), 1)
    img = np.log1p(acc.astype(np.float32))
    m = float(np.max(img)) if img.size else 0.0
    if not (m > 0.0):
        out = np.zeros((h, w, 3), dtype=np.uint8)
        return out
    norm = (img / m * 255.0).astype(np.uint8)
    return np.stack([norm, norm, norm], axis=-1)


def _render_intensity(px: np.ndarray, py: np.ndarray, intensity: np.ndarray, size: int, percentile: float) -> np.ndarray:
    w = h = int(size)
    count = np.zeros((h, w), dtype=np.uint32)
    summ = np.zeros((h, w), dtype=np.float32)
    if px.size:
        np.add.at(count, (py, px), 1)
        np.add.at(summ, (py, px), intensity.astype(np.float32, copy=False))
    img = np.zeros((h, w), dtype=np.float32)
    mask = count > 0
    img[mask] = summ[mask] / count[mask].astype(np.float32)
    vals = img[mask]
    if vals.size == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)
    p = float(max(0.0, min(percentile, 49.9)))
    if p > 0.0:
        vmin, vmax = np.percentile(vals.astype(np.float64, copy=False), [p, 100.0 - p]).tolist()
    else:
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
    span = max(1.0e-6, float(vmax) - float(vmin))
    norm = np.clip((img - float(vmin)) / span, 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)
    u8[~mask] = 0
    return np.stack([u8, u8, u8], axis=-1)


def _render_rgb(px: np.ndarray, py: np.ndarray, rgb_u8: np.ndarray, size: int) -> np.ndarray:
    w = h = int(size)
    count = np.zeros((h, w), dtype=np.uint32)
    sr = np.zeros((h, w), dtype=np.float32)
    sg = np.zeros((h, w), dtype=np.float32)
    sb = np.zeros((h, w), dtype=np.float32)
    if px.size:
        np.add.at(count, (py, px), 1)
        np.add.at(sr, (py, px), rgb_u8[:, 0].astype(np.float32))
        np.add.at(sg, (py, px), rgb_u8[:, 1].astype(np.float32))
        np.add.at(sb, (py, px), rgb_u8[:, 2].astype(np.float32))
    mask = count > 0
    out = np.zeros((h, w, 3), dtype=np.uint8)
    if np.any(mask):
        denom = count[mask].astype(np.float32)
        out[..., 0][mask] = np.clip(sr[mask] / denom, 0.0, 255.0).astype(np.uint8)
        out[..., 1][mask] = np.clip(sg[mask] / denom, 0.0, 255.0).astype(np.uint8)
        out[..., 2][mask] = np.clip(sb[mask] / denom, 0.0, 255.0).astype(np.uint8)
    return out


def _decode_pcl_rgb_float(rgb_float: np.ndarray) -> np.ndarray:
    if rgb_float.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    rgb_u32 = rgb_float.astype(np.float32, copy=False).view(np.uint32)
    r = ((rgb_u32 >> np.uint32(16)) & np.uint32(255)).astype(np.uint8)
    g = ((rgb_u32 >> np.uint32(8)) & np.uint32(255)).astype(np.uint8)
    b = (rgb_u32 & np.uint32(255)).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def _labels_to_rgb_u8(labels: np.ndarray, colors: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
    labels_i = labels.astype(np.int32, copy=False).reshape((-1,))
    out = np.full((labels_i.shape[0], 3), 200, dtype=np.uint8)
    for k, (r, g, b) in colors.items():
        mask = labels_i == int(k)
        if not np.any(mask):
            continue
        out[mask, 0] = np.uint8(int(r) & 255)
        out[mask, 1] = np.uint8(int(g) & 255)
        out[mask, 2] = np.uint8(int(b) & 255)
    return out


def _clusters_to_rgb_u8(cluster: np.ndarray) -> np.ndarray:
    c = cluster.astype(np.int32, copy=False).reshape((-1,))
    out = np.full((c.shape[0], 3), 200, dtype=np.uint8)
    if c.size == 0:
        return out
    for cid in sorted(int(v) for v in set(c.tolist()) if int(v) >= 0):
        r, g, b = _cluster_id_to_rgb(int(cid))
        mask = c == int(cid)
        out[mask, 0] = np.uint8(r)
        out[mask, 1] = np.uint8(g)
        out[mask, 2] = np.uint8(b)
    return out


def _render_one(
    pcd_path: Path,
    out_path: Path,
    *,
    mode: str,
    plane: str,
    size: int,
    bounds: Optional[Tuple[float, float, float, float]],
    percentile: float,
    pad: float,
    label_colors: Dict[int, Tuple[int, int, int]],
) -> None:
    pcd = _read_pcd(pcd_path)
    for name in ("x", "y", "z"):
        if name not in pcd.arrays:
            raise RuntimeError(f"PCD missing field '{name}': {pcd_path}")

    x, y = _xy_from_arrays(pcd.arrays, plane)
    keep = _finite_mask(x, y)
    x = x[keep]
    y = y[keep]

    if bounds is None:
        bounds = _auto_bounds(x, y, percentile=percentile, pad_frac=pad)

    px, py = _project_to_pixels(x, y, bounds=bounds, size=size)

    selected_mode = str(mode).lower()
    if selected_mode == "auto":
        if "rgb" in pcd.arrays:
            selected_mode = "rgb"
        elif "intensity" in pcd.arrays:
            selected_mode = "intensity"
        elif "label" in pcd.arrays:
            selected_mode = "label"
        elif "cluster" in pcd.arrays:
            selected_mode = "cluster"
        else:
            selected_mode = "density"

    if selected_mode == "density":
        img = _render_density(px, py, size=size)
    elif selected_mode == "intensity":
        if "intensity" not in pcd.arrays:
            raise RuntimeError(f"PCD missing field 'intensity' for --mode intensity: {pcd_path}")
        intensity = pcd.arrays["intensity"].astype(np.float32, copy=False)[keep]
        img = _render_intensity(px, py, intensity, size=size, percentile=percentile)
    elif selected_mode == "rgb":
        if "rgb" not in pcd.arrays:
            raise RuntimeError(f"PCD missing field 'rgb' for --mode rgb: {pcd_path}")
        rgb_u8 = _decode_pcl_rgb_float(pcd.arrays["rgb"][keep])
        img = _render_rgb(px, py, rgb_u8, size=size)
    elif selected_mode == "label":
        if "label" not in pcd.arrays:
            raise RuntimeError(f"PCD missing field 'label' for --mode label: {pcd_path}")
        rgb_u8 = _labels_to_rgb_u8(pcd.arrays["label"][keep], label_colors)
        img = _render_rgb(px, py, rgb_u8, size=size)
    elif selected_mode == "cluster":
        if "cluster" not in pcd.arrays:
            raise RuntimeError(f"PCD missing field 'cluster' for --mode cluster: {pcd_path}")
        rgb_u8 = _clusters_to_rgb_u8(pcd.arrays["cluster"][keep])
        img = _render_rgb(px, py, rgb_u8, size=size)
    else:
        raise ValueError(f"Unknown --mode {mode!r} (use auto/density/rgb/intensity/label/cluster)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="RGB").save(out_path)


def _parse_bounds(text: str) -> Optional[Tuple[float, float, float, float]]:
    text = (text or "").strip()
    if not text:
        return None
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("--bounds must be 'xmin,xmax,ymin,ymax'")
    xmin, xmax, ymin, ymax = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
    return float(xmin), float(xmax), float(ymin), float(ymax)


def main() -> int:
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pcd", type=str, default="", help="Single PCD file.")
    src.add_argument("--pcd-dir", type=str, default="", help="Directory of PCD files.")

    parser.add_argument("--out", type=str, default="", help="Output PNG path (single --pcd mode).")
    parser.add_argument("--out-dir", type=str, default="", help="Output directory (dir mode).")
    parser.add_argument("--pattern", type=str, default="*.pcd", help="Glob in --pcd-dir mode.")
    parser.add_argument("--every", type=int, default=1, help="Process every Nth file (dir mode).")
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of files (dir mode).")

    parser.add_argument("--mode", type=str, default="auto", help="auto/density/rgb/intensity/label/cluster")
    parser.add_argument("--plane", type=str, default="xy", help="xy/xz/yz (BEV=xy)")
    parser.add_argument("--size", type=int, default=1024, help="Output image size (square).")
    parser.add_argument("--bounds", type=str, default="", help="xmin,xmax,ymin,ymax (in selected plane).")
    parser.add_argument("--bounds-from", type=str, default="", help="Compute bounds once from this PCD, then reuse.")
    parser.add_argument("--percentile", type=float, default=1.0, help="Robust bounds/normalization percentile (0 disables).")
    parser.add_argument("--pad", type=float, default=0.02, help="Extra padding fraction for auto bounds.")
    parser.add_argument("--label-colors", type=str, default="", help="e.g. '0:56,188,75;1:180,180,180' (RGB)")
    args = parser.parse_args()

    ws_dir = Path(__file__).resolve().parents[3]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    default_out_dir = ws_dir / "output" / f"预览图_{stamp}"

    label_colors = _parse_label_colors(str(args.label_colors))
    bounds = _parse_bounds(str(args.bounds))
    if bounds is None and str(args.bounds_from).strip():
        ref = Path(args.bounds_from).expanduser().resolve()
        if not ref.is_file():
            raise FileNotFoundError(f"--bounds-from not found: {ref}")
        ref_pcd = _read_pcd(ref)
        for name in ("x", "y", "z"):
            if name not in ref_pcd.arrays:
                raise RuntimeError(f"--bounds-from PCD missing field '{name}': {ref}")
        rx, ry = _xy_from_arrays(ref_pcd.arrays, str(args.plane))
        keep = _finite_mask(rx, ry)
        bounds = _auto_bounds(
            rx[keep].astype(np.float32, copy=False),
            ry[keep].astype(np.float32, copy=False),
            percentile=float(args.percentile),
            pad_frac=float(args.pad),
        )
    size = max(32, int(args.size))

    if str(args.pcd).strip():
        pcd_path = Path(args.pcd).expanduser().resolve()
        if not pcd_path.is_file():
            raise FileNotFoundError(f"PCD not found: {pcd_path}")
        out_path = (
            Path(args.out).expanduser().resolve()
            if str(args.out).strip()
            else (default_out_dir / f"{pcd_path.stem}_{str(args.plane).lower()}_{str(args.mode).lower()}.png")
        )
        _render_one(
            pcd_path,
            out_path,
            mode=str(args.mode),
            plane=str(args.plane),
            size=size,
            bounds=bounds,
            percentile=float(args.percentile),
            pad=float(args.pad),
            label_colors=label_colors,
        )
        print(f"[OK] {pcd_path} -> {out_path}")
        return 0

    pcd_dir = Path(args.pcd_dir).expanduser().resolve()
    if not pcd_dir.is_dir():
        raise FileNotFoundError(f"PCD dir not found: {pcd_dir}")
    out_dir = Path(args.out_dir).expanduser().resolve() if str(args.out_dir).strip() else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(pcd_dir.glob(str(args.pattern)))
    every = max(1, int(args.every))
    max_files = int(args.max_files)
    processed = 0
    for idx, pcd_path in enumerate(files):
        if idx % every != 0:
            continue
        out_path = out_dir / f"{pcd_path.stem}_{str(args.plane).lower()}_{str(args.mode).lower()}.png"
        _render_one(
            pcd_path,
            out_path,
            mode=str(args.mode),
            plane=str(args.plane),
            size=size,
            bounds=bounds,
            percentile=float(args.percentile),
            pad=float(args.pad),
            label_colors=label_colors,
        )
        processed += 1
        if processed % 50 == 0:
            print(f"[OK] rendered {processed} images")
        if max_files > 0 and processed >= max_files:
            break

    print(f"[OK] Done. Output: {out_dir} ({processed} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
