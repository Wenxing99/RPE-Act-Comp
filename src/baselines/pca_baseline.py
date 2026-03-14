from __future__ import annotations

import torch


def fit_pca_basis(activations: torch.Tensor, rank: int) -> torch.Tensor:
    if activations.ndim != 2:
        raise ValueError(f"expected 2D activations, got {tuple(activations.shape)}")
    _, _, vh = torch.linalg.svd(activations, full_matrices=False)
    return vh[:rank].transpose(0, 1).contiguous()
