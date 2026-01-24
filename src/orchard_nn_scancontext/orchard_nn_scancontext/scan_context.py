#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ScanContext feature generator.

Copied from: RandLA-Net-pytorch/回环检测/loop_closure_2dcnn_scancontext_final/utils/scan_context.py
and kept behavior-compatible for inference.
"""

from __future__ import annotations

import numpy as np


class ScanContext:
    """ScanContext scene descriptor generator."""

    def __init__(
        self,
        num_sectors: int = 60,
        num_rings: int = 20,
        min_range: float = 0.1,
        max_range: float = 80.0,
        height_lower_bound: float = -1.0,
        height_upper_bound: float = 9.0,
    ) -> None:
        self.num_sectors = int(num_sectors)
        self.num_rings = int(num_rings)
        self.min_range = float(min_range)
        self.max_range = float(max_range)
        self.height_lower_bound = float(height_lower_bound)
        self.height_upper_bound = float(height_upper_bound)

        self.ring_width = (self.max_range - self.min_range) / float(self.num_rings)
        self.sector_angle = 2.0 * np.pi / float(self.num_sectors)

    def _xy_to_polar(self, x: float, y: float) -> tuple[float, float]:
        r = float(np.sqrt(x**2 + y**2))
        theta = float(np.arctan2(y, x))
        if theta < 0.0:
            theta += float(2.0 * np.pi)
        return r, theta

    def _get_ring_idx(self, r: float) -> int:
        ring_idx = int((float(r) - self.min_range) / float(self.ring_width))
        if ring_idx < 0:
            ring_idx = 0
        if ring_idx >= self.num_rings:
            ring_idx = int(self.num_rings - 1)
        return ring_idx

    def _get_sector_idx(self, theta: float) -> int:
        sector_idx = int(float(theta) / float(self.sector_angle))
        if sector_idx < 0:
            sector_idx = 0
        if sector_idx >= self.num_sectors:
            sector_idx = int(self.num_sectors - 1)
        return sector_idx

    def generate_scan_context(self, point_cloud: np.ndarray) -> np.ndarray:
        return self.make_scan_context(point_cloud)

    def make_scan_context(self, point_cloud: np.ndarray) -> np.ndarray:
        scan_context = np.full((self.num_rings, self.num_sectors), self.height_lower_bound, dtype=np.float32)
        point_count = np.zeros((self.num_rings, self.num_sectors), dtype=np.int32)

        if point_cloud.size == 0:
            return np.zeros((self.num_rings, self.num_sectors), dtype=np.float32)

        for i in range(point_cloud.shape[0]):
            x, y, z = float(point_cloud[i, 0]), float(point_cloud[i, 1]), float(point_cloud[i, 2])

            r, theta = self._xy_to_polar(x, y)
            if r < self.min_range or r > self.max_range:
                continue
            if z < self.height_lower_bound or z > self.height_upper_bound:
                continue

            ring_idx = self._get_ring_idx(r)
            sector_idx = self._get_sector_idx(theta)

            if z > float(scan_context[ring_idx, sector_idx]):
                scan_context[ring_idx, sector_idx] = float(z)

            point_count[ring_idx, sector_idx] += 1

        scan_context[point_count == 0] = 0.0
        sc_min = float(np.min(scan_context))
        sc_max = float(np.max(scan_context))
        if sc_max > sc_min:
            scan_context = (scan_context - sc_min) / (sc_max - sc_min)
        return scan_context.astype(np.float32, copy=False)

