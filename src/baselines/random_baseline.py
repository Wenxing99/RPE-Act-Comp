from __future__ import annotations

import torch

from projections.orthogonal import gaussian_orthoprojector


def fit_random_basis(
    head_dim: int,
    rank: int,
    *,
    seed: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return gaussian_orthoprojector(
        head_dim,
        rank,
        generator=generator,
        dtype=dtype,
        device=device,
    )
