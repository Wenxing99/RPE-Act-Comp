from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_pt(payload: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def load_pt(path: str | Path) -> Any:
    return torch.load(Path(path), map_location="cpu", weights_only=False)
