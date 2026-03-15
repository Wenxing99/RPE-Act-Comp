import torch

from experiments.collect_activations import collect_head_activations
from experiments.evaluate_lm import evaluate_causal_lm
from experiments.single_head_rpedr import (
    extract_single_head_matrix,
    run_rpedr_full,
    run_rpedr_m1,
    run_rpedr_single_best,
)
from models.gpt2_adapter import CompressionSpec, GPT2Adapter
from utils.config import load_yaml


def _scorer(basis: torch.Tensor) -> float:
    return float(-basis[0, 0].item())


def _check_basis(result, rank: int, dim: int) -> None:
    assert result.basis.shape == (dim, rank)
    gram = result.basis.transpose(0, 1) @ result.basis
    assert torch.allclose(gram, torch.eye(rank), atol=1e-5)
    assert result.method in {"rpedr_m1", "rpedr_single_best", "rpedr_full"}
    assert "num_groups" in result.metadata


def test_rpedr_variants_return_orthonormal_bases_and_are_deterministic():
    torch.manual_seed(0)
    activations = torch.randn(32, 8)
    output_weight = torch.randn(8, 12)

    r1 = run_rpedr_m1(
        local_activations=activations,
        output_weight=output_weight,
        head_dim=8,
        rank=4,
        seed=7,
        scorer=_scorer,
        num_groups=4,
    )
    r2 = run_rpedr_m1(
        local_activations=activations,
        output_weight=output_weight,
        head_dim=8,
        rank=4,
        seed=7,
        scorer=_scorer,
        num_groups=4,
    )
    assert torch.allclose(r1.basis, r2.basis)
    _check_basis(r1, rank=4, dim=8)

    best = run_rpedr_single_best(
        local_activations=activations,
        output_weight=output_weight,
        head_dim=8,
        rank=4,
        seed=7,
        scorer=_scorer,
        num_groups=4,
        group_size=3,
        topk=2,
    )
    full = run_rpedr_full(
        local_activations=activations,
        output_weight=output_weight,
        head_dim=8,
        rank=4,
        seed=7,
        scorer=_scorer,
        num_groups=4,
        group_size=3,
        topk=2,
    )
    _check_basis(best, rank=4, dim=8)
    _check_basis(full, rank=4, dim=8)


def test_rpedr_basis_integrates_with_current_compress_and_eval_path():
    model_config = load_yaml("configs/model/tiny_random_gpt2.yaml")
    adapter = GPT2Adapter.from_config(model_config).to("cpu")
    bundle = collect_head_activations(
        adapter,
        ["single head integration sample", "second sample for hooks"],
        max_length=16,
        device="cpu",
    )
    local_output = extract_single_head_matrix(bundle, "output", layer_index=0, head_index=0)
    output_weight = adapter.get_head_output_weight(0, 0)
    result = run_rpedr_full(
        local_activations=local_output,
        output_weight=output_weight,
        head_dim=adapter.head_dim,
        rank=4,
        seed=3,
        scorer=_scorer,
        num_groups=2,
        group_size=2,
        topk=1,
    )
    spec = CompressionSpec(layer_index=0, rank=4, bases={0: result.basis})
    compressed = adapter.apply_compression(spec).to("cpu")
    metrics = evaluate_causal_lm(
        compressed,
        ["evaluation text for compressed adapter"],
        max_length=16,
        device="cpu",
        teacher_adapter=adapter,
    )
    assert "nll" in metrics
    assert "teacher_logit_kl" in metrics
