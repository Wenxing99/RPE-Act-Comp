import torch

from compression.vo_folding import folded_project_head, runtime_project_head
from projections.orthogonal import gaussian_orthoprojector


def test_runtime_projection_matches_folded_weights():
    hidden_states = torch.randn(11, 10)
    value_weight = torch.randn(10, 6)
    value_bias = torch.randn(6)
    output_weight = torch.randn(6, 10)
    basis = gaussian_orthoprojector(6, 4)

    runtime = runtime_project_head(hidden_states, value_weight, value_bias, output_weight, basis)
    folded = folded_project_head(hidden_states, value_weight, value_bias, output_weight, basis)
    assert torch.allclose(runtime, folded, atol=1e-5)
