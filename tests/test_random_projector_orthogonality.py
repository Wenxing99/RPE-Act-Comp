import torch

from projections.orthogonal import gaussian_orthoprojector


def test_random_projector_is_orthonormal():
    basis = gaussian_orthoprojector(16, 5)
    gram = basis.transpose(0, 1) @ basis
    assert torch.allclose(gram, torch.eye(5), atol=1e-5)
