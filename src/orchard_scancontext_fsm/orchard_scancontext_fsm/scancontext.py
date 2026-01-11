from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ScanContextParams:
    num_ring: int = 20
    num_sector: int = 60
    max_radius: float = 80.0
    lidar_height: float = 2.0
    search_ratio: float = 0.1

    @property
    def sector_angle_rad(self) -> float:
        return float(2.0 * math.pi / float(self.num_sector))


def make_scancontext(points_xyz: np.ndarray, params: ScanContextParams) -> np.ndarray:
    if points_xyz.size == 0:
        return np.zeros((params.num_ring, params.num_sector), dtype=np.float32)

    x = points_xyz[:, 0].astype(np.float32, copy=False)
    y = points_xyz[:, 1].astype(np.float32, copy=False)
    z = points_xyz[:, 2].astype(np.float32, copy=False) + float(params.lidar_height)

    r = np.hypot(x, y)
    mask = r <= float(params.max_radius)
    if not bool(np.any(mask)):
        return np.zeros((params.num_ring, params.num_sector), dtype=np.float32)

    x = x[mask]
    y = y[mask]
    z = z[mask]
    r = r[mask]

    ang = np.degrees(np.arctan2(y, x))
    ang = np.mod(ang + 360.0, 360.0)

    ring = np.ceil((r / float(params.max_radius)) * float(params.num_ring)).astype(np.int32)
    ring = np.clip(ring, 1, int(params.num_ring)) - 1

    sector = np.ceil((ang / 360.0) * float(params.num_sector)).astype(np.int32)
    sector = np.clip(sector, 1, int(params.num_sector)) - 1

    desc = np.full((int(params.num_ring), int(params.num_sector)), -np.inf, dtype=np.float32)
    np.maximum.at(desc, (ring, sector), z)
    desc[~np.isfinite(desc)] = 0.0
    return desc


def _sector_key(desc: np.ndarray) -> np.ndarray:
    return desc.mean(axis=0, dtype=np.float32)


def _circshift_cols(desc: np.ndarray, num_shift: int) -> np.ndarray:
    if num_shift == 0:
        return desc
    return np.roll(desc, int(num_shift), axis=1)


def _dist_direct(sc1: np.ndarray, sc2: np.ndarray) -> float:
    a_norm = np.linalg.norm(sc1, axis=0)
    b_norm = np.linalg.norm(sc2, axis=0)
    mask = (a_norm > 0.0) & (b_norm > 0.0)
    if not bool(np.any(mask)):
        return 1.0
    dots = np.sum(sc1 * sc2, axis=0)
    sim = dots[mask] / (a_norm[mask] * b_norm[mask])
    return float(1.0 - float(np.mean(sim)))


def distance_between_scancontexts(sc1: np.ndarray, sc2: np.ndarray, params: ScanContextParams) -> Tuple[float, float]:
    vkey1 = _sector_key(sc1)
    vkey2 = _sector_key(sc2)

    best_shift = 0
    best_norm = float("inf")
    for shift in range(int(params.num_sector)):
        diff_norm = float(np.linalg.norm(vkey1 - np.roll(vkey2, shift)))
        if diff_norm < best_norm:
            best_norm = diff_norm
            best_shift = int(shift)

    search_radius = int(round(0.5 * float(params.search_ratio) * float(params.num_sector)))
    shifts = {best_shift}
    for i in range(1, search_radius + 1):
        shifts.add((best_shift + i) % int(params.num_sector))
        shifts.add((best_shift - i) % int(params.num_sector))

    min_dist = float("inf")
    min_shift = 0
    for shift in sorted(shifts):
        d = _dist_direct(sc1, _circshift_cols(sc2, shift))
        if d < min_dist:
            min_dist = d
            min_shift = int(shift)

    signed_shift = int(min_shift) if int(min_shift) <= int(params.num_sector) // 2 else int(min_shift) - int(params.num_sector)
    yaw_diff_rad = float(signed_shift) * float(params.sector_angle_rad)
    return float(min_dist), float(yaw_diff_rad)


def downsample_xyz(points_xyz: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0:
        return points_xyz
    if points_xyz.shape[0] <= int(max_points):
        return points_xyz
    step = int(math.ceil(float(points_xyz.shape[0]) / float(max_points)))
    return points_xyz[::step]

