from __future__ import annotations

import torch


def orthonormalize(matrix: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q


def gaussian_orthoprojector(
    dim: int,
    rank: int,
    *,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    raw = torch.randn(dim, rank, generator=generator, dtype=dtype, device=device)
    return orthonormalize(raw)
