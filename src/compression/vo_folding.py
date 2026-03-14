from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FoldedHeadWeights:
    value_weight: torch.Tensor
    value_bias: torch.Tensor
    output_weight: torch.Tensor


def projector_from_basis(basis: torch.Tensor) -> torch.Tensor:
    return basis @ basis.transpose(0, 1)


def fold_head_basis(
    value_weight: torch.Tensor,
    value_bias: torch.Tensor,
    output_weight: torch.Tensor,
    basis: torch.Tensor,
) -> FoldedHeadWeights:
    return FoldedHeadWeights(
        value_weight=value_weight @ basis,
        value_bias=value_bias @ basis,
        output_weight=basis.transpose(0, 1) @ output_weight,
    )


def runtime_project_head(
    hidden_states: torch.Tensor,
    value_weight: torch.Tensor,
    value_bias: torch.Tensor,
    output_weight: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    projector = projector_from_basis(basis)
    values = hidden_states @ value_weight + value_bias
    return values @ projector @ output_weight


def folded_project_head(
    hidden_states: torch.Tensor,
    value_weight: torch.Tensor,
    value_bias: torch.Tensor,
    output_weight: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    folded = fold_head_basis(value_weight, value_bias, output_weight, basis)
    low_rank_values = hidden_states @ folded.value_weight + folded.value_bias
    return low_rank_values @ folded.output_weight
