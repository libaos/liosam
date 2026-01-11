"""Model loader for Fixed4DRandLANet with state_dict adaptation.

Searches checkpoint path from config and falls back if needed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


def load_model(num_classes: int, device: torch.device, checkpoint_path: Optional[Path]) -> torch.nn.Module:
    """Load the vendored Fixed4DRandLANet checkpoint for inference."""
    from .vendor.model_fix_v2 import Fixed4DRandLANet

    model = Fixed4DRandLANet(d_in=6, num_classes=num_classes, num_neighbors=16, decimation=4, device=device)
    model.to(device)
    model.eval()

    if checkpoint_path and checkpoint_path.exists():
        try:
            ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(str(checkpoint_path), map_location=device)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        new_state = {}
        for k, v in state.items():
            if k.startswith("module.model."):
                new_state[k[len("module.model."):]] = v
            elif k.startswith("model."):
                new_state[k[len("model."):]] = v
            else:
                new_state[k] = v
        try:
            model.load_state_dict(new_state, strict=False)
        except Exception:
            # Attempt without strict to be safe
            model.load_state_dict(new_state, strict=False)
    return model
