from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch

from baselines.pca_baseline import fit_pca_basis
from baselines.random_baseline import fit_random_basis
from compression.vo_folding import projector_from_basis
from models.gpt2_adapter import CompressedGPT2Attention, CompressionSpec


@dataclass
class CandidateRecord:
    basis: torch.Tensor
    local_score: float
    select_score: float | None = None


@dataclass
class SearchResult:
    method: str
    basis: torch.Tensor
    projector_trace: float
    local_score: float | None
    select_score: float | None
    elapsed_sec: float
    metadata: dict


def extract_single_head_matrix(bundle: dict, kind: str, layer_index: int, head_index: int) -> torch.Tensor:
    return bundle["activations"][kind][layer_index][:, head_index, :]


def compute_local_score(
    activations: torch.Tensor,
    output_weight: torch.Tensor,
    basis: torch.Tensor,
) -> float:
    output_weight = output_weight.to(activations.device)
    projector = projector_from_basis(basis).to(activations.device)
    dense_output = activations @ output_weight
    projected_output = activations @ projector @ output_weight
    return float(torch.mean((dense_output - projected_output) ** 2).item())


def make_teacher_kl_scorer(
    adapter,
    layer_index: int,
    head_index: int,
    texts: list[str],
    max_length: int,
    device: str,
) -> Callable[[torch.Tensor], float]:
    tokenized = adapter.tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    tokenized = {key: value.to(device) for key, value in tokenized.items()}
    adapter.model.eval()
    with torch.no_grad():
        teacher_logits = adapter.model(**tokenized).logits

    scorer_adapter = adapter.clone().to(device)
    scorer_basis = torch.eye(scorer_adapter.head_dim, dtype=teacher_logits.dtype, device=device)[:, :1]
    scorer_attention = scorer_adapter.model.transformer.h[layer_index].attn
    scorer_adapter.model.transformer.h[layer_index].attn = CompressedGPT2Attention(
        scorer_attention,
        {head_index: scorer_basis},
    )
    scorer_adapter.model.eval()
    teacher_probs = torch.softmax(teacher_logits, dim=-1)

    def score(basis: torch.Tensor) -> float:
        scorer_adapter.model.transformer.h[layer_index].attn.set_head_basis(head_index, basis.to(device))
        with torch.no_grad():
            student_logits = scorer_adapter.model(**tokenized).logits
        student_log_probs = torch.log_softmax(student_logits, dim=-1)
        kl = teacher_probs * (torch.log(teacher_probs.clamp_min(1e-9)) - student_log_probs)
        return float(kl.sum(dim=-1).mean().item())

    return score


def build_single_head_random_basis(head_dim: int, rank: int, seed: int) -> torch.Tensor:
    return fit_random_basis(head_dim, rank, seed=seed)


def build_single_head_pca_basis(activations: torch.Tensor, rank: int) -> torch.Tensor:
    return fit_pca_basis(activations, rank)


def _sample_candidates(
    head_dim: int,
    rank: int,
    total: int,
    seed: int,
) -> list[torch.Tensor]:
    return [
        build_single_head_random_basis(head_dim, rank, seed=seed + idx)
        for idx in range(total)
    ]


def _top_eigen_basis(projectors: list[torch.Tensor], rank: int) -> torch.Tensor:
    ensemble = torch.stack(projectors, dim=0).mean(dim=0)
    evals, evecs = torch.linalg.eigh(ensemble)
    order = torch.argsort(evals, descending=True)
    return evecs[:, order[:rank]].contiguous()


def run_rpedr_variant(
    *,
    method: str,
    local_activations: torch.Tensor,
    output_weight: torch.Tensor,
    head_dim: int,
    rank: int,
    seed: int,
    scorer: Callable[[torch.Tensor], float],
    num_groups: int,
    group_size: int,
    topk: int,
) -> SearchResult:
    start = time.perf_counter()
    candidates = _sample_candidates(head_dim, rank, num_groups * group_size, seed=seed)
    grouped_records: list[list[CandidateRecord]] = []

    for group_index in range(num_groups):
        group_records = []
        for inner_index in range(group_size):
            basis = candidates[group_index * group_size + inner_index]
            group_records.append(
                CandidateRecord(
                    basis=basis,
                    local_score=compute_local_score(local_activations, output_weight, basis),
                )
            )
        grouped_records.append(group_records)

    group_winners: list[CandidateRecord] = []
    for group_records in grouped_records:
        finalists = sorted(group_records, key=lambda item: item.local_score)[: max(1, topk)]
        for record in finalists:
            record.select_score = scorer(record.basis)
        winner = min(finalists, key=lambda item: item.select_score if item.select_score is not None else float("inf"))
        group_winners.append(winner)

    if method == "rpedr_single_best":
        best = min(group_winners, key=lambda item: item.select_score if item.select_score is not None else float("inf"))
        final_basis = best.basis
        projector_trace = float(torch.trace(projector_from_basis(final_basis)).item())
        elapsed = time.perf_counter() - start
        return SearchResult(
            method=method,
            basis=final_basis,
            projector_trace=projector_trace,
            local_score=best.local_score,
            select_score=best.select_score,
            elapsed_sec=elapsed,
            metadata={
                "num_groups": num_groups,
                "group_size": group_size,
                "topk": topk,
                "num_candidates": num_groups * group_size,
            },
        )

    projector_list = [projector_from_basis(record.basis) for record in group_winners]
    final_basis = _top_eigen_basis(projector_list, rank)
    elapsed = time.perf_counter() - start
    select_scores = [record.select_score for record in group_winners if record.select_score is not None]
    local_scores = [record.local_score for record in group_winners]
    return SearchResult(
        method=method,
        basis=final_basis,
        projector_trace=float(torch.trace(projector_from_basis(final_basis)).item()),
        local_score=float(sum(local_scores) / len(local_scores)),
        select_score=float(sum(select_scores) / len(select_scores)) if select_scores else None,
        elapsed_sec=elapsed,
        metadata={
            "num_groups": num_groups,
            "group_size": group_size,
            "topk": topk,
            "num_candidates": num_groups * group_size,
        },
    )


def run_rpedr_m1(
    *,
    local_activations: torch.Tensor,
    output_weight: torch.Tensor,
    head_dim: int,
    rank: int,
    seed: int,
    scorer: Callable[[torch.Tensor], float],
    num_groups: int,
) -> SearchResult:
    return run_rpedr_variant(
        method="rpedr_m1",
        local_activations=local_activations,
        output_weight=output_weight,
        head_dim=head_dim,
        rank=rank,
        seed=seed,
        scorer=scorer,
        num_groups=num_groups,
        group_size=1,
        topk=1,
    )


def run_rpedr_single_best(
    *,
    local_activations: torch.Tensor,
    output_weight: torch.Tensor,
    head_dim: int,
    rank: int,
    seed: int,
    scorer: Callable[[torch.Tensor], float],
    num_groups: int,
    group_size: int,
    topk: int,
) -> SearchResult:
    return run_rpedr_variant(
        method="rpedr_single_best",
        local_activations=local_activations,
        output_weight=output_weight,
        head_dim=head_dim,
        rank=rank,
        seed=seed,
        scorer=scorer,
        num_groups=num_groups,
        group_size=group_size,
        topk=topk,
    )


def run_rpedr_full(
    *,
    local_activations: torch.Tensor,
    output_weight: torch.Tensor,
    head_dim: int,
    rank: int,
    seed: int,
    scorer: Callable[[torch.Tensor], float],
    num_groups: int,
    group_size: int,
    topk: int,
) -> SearchResult:
    return run_rpedr_variant(
        method="rpedr_full",
        local_activations=local_activations,
        output_weight=output_weight,
        head_dim=head_dim,
        rank=rank,
        seed=seed,
        scorer=scorer,
        num_groups=num_groups,
        group_size=group_size,
        topk=topk,
    )


