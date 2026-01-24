from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ScanContextParams:
    num_ring: int
    num_sector: int
    min_range: float
    max_range: float
    height_lower_bound: float
    height_upper_bound: float


@dataclass(frozen=True)
class RouteDb:
    prototypes: np.ndarray  # (K, R, S) float32
    proto_norm_flat: np.ndarray  # (K, R*S) float32, L2-normalized
    counts: np.ndarray  # (K,) int32
    params: ScanContextParams
    meta: Dict[str, Any]


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def _normalize_rows(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, float(eps))
    return x / denom


def load_route_db(path: Path) -> RouteDb:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"route db not found: {path}")

    data = np.load(str(path), allow_pickle=True)
    prototypes = np.asarray(data["prototypes"], dtype=np.float32)
    counts = np.asarray(data.get("counts", np.zeros((prototypes.shape[0],), dtype=np.int32)), dtype=np.int32)

    params_arr = np.asarray(data["params"], dtype=np.float64).reshape(-1)
    if params_arr.size < 6:
        raise ValueError(f"invalid params in db: expected >=6 values, got {params_arr.size} ({path})")
    params = ScanContextParams(
        num_ring=int(params_arr[0]),
        num_sector=int(params_arr[1]),
        min_range=float(params_arr[2]),
        max_range=float(params_arr[3]),
        height_lower_bound=float(params_arr[4]),
        height_upper_bound=float(params_arr[5]),
    )

    meta: Dict[str, Any] = {}
    for key in ("bag_path", "cloud_topic", "labeling", "process_hz", "max_points", "total_msgs", "per"):
        if key in data:
            raw = data[key]
            if isinstance(raw, np.ndarray) and raw.shape == ():
                raw = raw.item()
            if isinstance(raw, (bytes, np.bytes_)):
                raw = raw.decode("utf-8", errors="ignore")
            meta[str(key)] = raw

    flat = prototypes.reshape(prototypes.shape[0], -1)
    proto_norm_flat = _normalize_rows(flat)
    return RouteDb(
        prototypes=prototypes,
        proto_norm_flat=proto_norm_flat.astype(np.float32, copy=False),
        counts=counts,
        params=params,
        meta=meta,
    )


def predict_route_id_cosine(
    desc: np.ndarray,
    db: RouteDb,
    *,
    temperature: float = 0.02,
) -> Tuple[int, float]:
    if desc.size == 0:
        return (-1, 0.0)

    flat = np.asarray(desc, dtype=np.float32).reshape(1, -1)
    flat = _normalize_rows(flat)
    sims = (flat @ db.proto_norm_flat.T).reshape(-1)  # (K,)
    if db.counts.shape[0] == sims.shape[0]:
        invalid = db.counts <= 0
        if np.any(invalid):
            sims = sims.copy()
            sims[invalid] = -np.inf
    if not np.any(np.isfinite(sims)):
        return (-1, 0.0)
    if float(temperature) <= 0.0:
        route_id = int(np.argmax(sims))
        conf = float(np.max(sims))
        return route_id, conf

    probs = _softmax(sims / float(temperature), axis=0)
    route_id = int(np.argmax(probs))
    conf = float(probs[route_id])
    return route_id, conf


def predict_route_id_l2(
    desc: np.ndarray,
    db: RouteDb,
    *,
    temperature: float = 0.05,
) -> Tuple[int, float]:
    if desc.size == 0:
        return (-1, 0.0)

    q = np.asarray(desc, dtype=np.float32).reshape(1, -1)
    p = db.prototypes.reshape(db.prototypes.shape[0], -1).astype(np.float32, copy=False)
    d2 = ((p - q) ** 2).mean(axis=1)  # (K,)
    if db.counts.shape[0] == d2.shape[0]:
        invalid = db.counts <= 0
        if np.any(invalid):
            d2 = d2.copy()
            d2[invalid] = np.inf
    if not np.any(np.isfinite(d2)):
        return (-1, 0.0)
    if float(temperature) <= 0.0:
        route_id = int(np.argmin(d2))
        conf = float(-d2[route_id])
        return route_id, conf

    probs = _softmax((-d2) / float(temperature), axis=0)
    route_id = int(np.argmax(probs))
    conf = float(probs[route_id])
    return route_id, conf


def load_route_db_from_npz(path: Path) -> RouteDb:
    return load_route_db(path)
