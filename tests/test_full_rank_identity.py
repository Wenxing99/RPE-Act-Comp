import torch

from baselines.pca_baseline import fit_pca_basis
from compression.vo_folding import runtime_project_head


def test_full_rank_identity_matches_dense_path():
    hidden_states = torch.randn(9, 12)
    value_weight = torch.randn(12, 8)
    value_bias = torch.randn(8)
    output_weight = torch.randn(8, 12)
    basis = fit_pca_basis(torch.randn(40, 8), rank=8)

    dense = (hidden_states @ value_weight + value_bias) @ output_weight
    projected = runtime_project_head(
        hidden_states,
        value_weight,
        value_bias,
        output_weight,
        basis,
    )
    assert torch.allclose(projected, dense, atol=1e-5)
