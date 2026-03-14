from __future__ import annotations

import torch

from compression.vo_folding import projector_from_basis


def reconstruction_mse(activations: torch.Tensor, basis: torch.Tensor) -> float:
    projector = projector_from_basis(basis).to(activations.device)
    reconstruction = activations @ projector
    return float(torch.mean((activations - reconstruction) ** 2).item())
