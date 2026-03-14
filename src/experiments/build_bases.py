from __future__ import annotations

import torch

from baselines.pca_baseline import fit_pca_basis
from baselines.random_baseline import fit_random_basis


def build_random_bases(bundle: dict, layer_index: int, rank: int, seed: int) -> dict[int, torch.Tensor]:
    del layer_index
    head_dim = int(bundle["metadata"]["head_dim"])
    num_heads = int(bundle["metadata"]["num_heads"])
    return {
        head_index: fit_random_basis(head_dim, rank, seed=seed + head_index)
        for head_index in range(num_heads)
    }


def build_pca_bases(bundle: dict, layer_index: int, rank: int) -> dict[int, torch.Tensor]:
    outputs = bundle["activations"]["output"][layer_index]
    return {
        head_index: fit_pca_basis(outputs[:, head_index, :], rank)
        for head_index in range(outputs.shape[1])
    }
