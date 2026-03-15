from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def resolve_runtime_device(requested_device: str | None = None) -> tuple[str, dict[str, Any]]:
    requested = (requested_device or "auto").strip().lower()
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0

    if requested == "auto":
        selected = "cuda" if cuda_available else "cpu"
        reason = "auto_cuda" if cuda_available else "auto_cpu_no_cuda"
    elif requested.startswith("cuda"):
        if cuda_available:
            selected = requested
            reason = "requested_cuda"
        else:
            selected = "cpu"
            reason = "fallback_cpu_no_cuda"
    else:
        selected = requested
        reason = "requested_non_cuda"

    metadata = {
        "requested_device": requested,
        "selected_device": selected,
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "device_resolution_reason": reason,
    }
    return selected, metadata
