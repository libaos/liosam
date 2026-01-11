"""Segmentation inference helpers."""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import torch


def preprocess_points(
    points: np.ndarray,
    num_points: int,
    clear_rgb: bool = True,
    sampling: str = "random",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Ensure (N,6) XYZ000 and sample/pad to num_points."""
    if points.ndim != 2:
        raise ValueError(f"points must be 2D, got {points.shape}")
    if points.shape[1] == 3:
        points = np.hstack([points.astype(np.float32), np.zeros((points.shape[0], 3), dtype=np.float32)])
    elif points.shape[1] >= 6:
        points = points[:, :6].astype(np.float32)
    else:
        raise ValueError(f"Unexpected points shape: {points.shape}")
    if clear_rgb:
        points[:, 3:6] = 0.0

    sampling = (sampling or "random").lower()
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    n = points.shape[0]
    if n > num_points:
        if sampling == "random":
            idx = rng.choice(n, num_points, replace=False)
        elif sampling == "stride":
            idx = np.linspace(0, n - 1, num_points, dtype=np.int64)
        elif sampling == "first":
            idx = np.arange(num_points, dtype=np.int64)
        else:
            raise ValueError(f"Unknown sampling method: {sampling}")
        points = points[idx]
    elif n < num_points and n > 0:
        if sampling == "random":
            pad_idx = rng.choice(n, num_points - n, replace=True)
        else:
            pad_idx = np.resize(np.arange(n, dtype=np.int64), num_points - n)
        points = np.vstack([points, points[pad_idx]])
    elif n == 0:
        points = np.zeros((num_points, 6), dtype=np.float32)

    return points.astype(np.float32)


def run_inference(model: torch.nn.Module, points6: np.ndarray, device: torch.device, num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Run model inference; returns (labels, probabilities)."""
    model = model.to(device)
    with torch.no_grad():
        x = torch.from_numpy(points6).float().unsqueeze(0).to(device)  # (1, N, 6)
        logits = model(x)  # could be (1, C, N) or (1, N, C)
        if logits.dim() == 3 and logits.shape[1] == num_classes:
            logits = logits.permute(0, 2, 1)  # (1, N, C)
        logits = logits.squeeze(0)  # (N, C)
        probs = torch.softmax(logits, dim=-1)
        labels = torch.argmax(probs, dim=-1)
    return labels.cpu().numpy(), probs.cpu().numpy()
