from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.collect_activations import collect_head_activations
from experiments.evaluate_lm import evaluate_causal_lm
from experiments.single_head_rpedr import (
    build_single_head_pca_basis,
    build_single_head_random_basis,
    compute_local_score,
    extract_single_head_matrix,
    make_teacher_kl_scorer,
    run_rpedr_full,
    run_rpedr_m1,
    run_rpedr_single_best,
)
from models.gpt2_adapter import CompressionSpec, GPT2Adapter
from data.text_data import load_named_splits
from utils.config import ensure_dir, load_yaml, resolve_runtime_device
from utils.io import save_json, save_pt


def _combine_splits(named_splits: dict[str, list[str]], split_names: list[str]) -> list[str]:
    texts: list[str] = []
    for split_name in split_names:
        texts.extend(named_splits[split_name])
    return texts


def _rank_from_ratio(head_dim: int, rank_ratio: float) -> int:
    return max(1, min(head_dim, int(round(head_dim * rank_ratio))))


def _save_basis(artifact_root: Path, method: str, basis, metadata: dict) -> None:
    method_dir = artifact_root / method
    method_dir.mkdir(parents=True, exist_ok=True)
    save_pt({"basis": basis}, method_dir / "basis.pt")
    save_json(metadata, method_dir / "result.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model/distilgpt2_single_head.yaml")
    parser.add_argument("--data-config", default="configs/data/wikitext2_single_head.yaml")
    parser.add_argument("--exp-config", default="configs/exp/single_head_distilgpt2.yaml")
    args = parser.parse_args()

    model_config = load_yaml(args.model_config)
    data_config = load_yaml(args.data_config)
    exp_config = load_yaml(args.exp_config)

    artifact_root = ensure_dir(exp_config["artifact_root"])
    named_splits = load_named_splits(data_config)
    device, device_meta = resolve_runtime_device(model_config.get("device"))
    print(f"[device] requested={device_meta['requested_device']} selected={device_meta['selected_device']} cuda_available={device_meta['cuda_available']}")
    adapter = GPT2Adapter.from_config(model_config).to(device)

    target_layer = int(exp_config["target"]["layer_index"])
    target_head = int(exp_config["target"]["head_index"])
    rank = _rank_from_ratio(adapter.head_dim, float(exp_config["target"]["rank_ratio"]))

    local_split_name = exp_config["splits"]["local"]
    select_split_name = exp_config["splits"]["select"]
    test_split_name = exp_config["splits"]["test"]
    fit_split_names = list(exp_config["baseline_fit_splits"])

    local_bundle = collect_head_activations(
        adapter,
        named_splits[local_split_name],
        max_length=int(data_config["max_length"]),
        device=device,
    )
    fit_bundle = collect_head_activations(
        adapter,
        _combine_splits(named_splits, fit_split_names),
        max_length=int(data_config["max_length"]),
        device=device,
    )

    local_output_matrix = extract_single_head_matrix(local_bundle, "output", target_layer, target_head)
    fit_output_matrix = extract_single_head_matrix(fit_bundle, "output", target_layer, target_head)
    output_weight = adapter.get_head_output_weight(target_layer, target_head)
    selection_scorer = make_teacher_kl_scorer(
        adapter,
        layer_index=target_layer,
        head_index=target_head,
        texts=named_splits[select_split_name],
        max_length=int(data_config["max_length"]),
        device=device,
    )

    methods = {}
    methods["random"] = build_single_head_random_basis(adapter.head_dim, rank, seed=int(exp_config["seed"]))
    methods["pca"] = build_single_head_pca_basis(fit_output_matrix, rank)

    search_cfg = exp_config["search"]
    m1_result = run_rpedr_m1(
        local_activations=local_output_matrix,
        output_weight=output_weight,
        head_dim=adapter.head_dim,
        rank=rank,
        seed=int(exp_config["seed"]),
        scorer=selection_scorer,
        num_groups=int(search_cfg["num_groups"]),
    )
    single_best_result = run_rpedr_single_best(
        local_activations=local_output_matrix,
        output_weight=output_weight,
        head_dim=adapter.head_dim,
        rank=rank,
        seed=int(exp_config["seed"]),
        scorer=selection_scorer,
        num_groups=int(search_cfg["num_groups"]),
        group_size=int(search_cfg["group_size"]),
        topk=int(search_cfg["topk"]),
    )
    full_result = run_rpedr_full(
        local_activations=local_output_matrix,
        output_weight=output_weight,
        head_dim=adapter.head_dim,
        rank=rank,
        seed=int(exp_config["seed"]),
        scorer=selection_scorer,
        num_groups=int(search_cfg["num_groups"]),
        group_size=int(search_cfg["group_size"]),
        topk=int(search_cfg["topk"]),
    )

    dense_metrics = evaluate_causal_lm(
        adapter,
        named_splits[test_split_name],
        max_length=int(data_config["max_length"]),
        device=device,
    )

    rows: list[dict] = []
    for method in exp_config["methods"]:
        if method == "dense":
            result = {
                "method": method,
                "layer_index": target_layer,
                "head_index": target_head,
                "rank": adapter.head_dim,
                "rank_ratio": 1.0,
                "local_score": None,
                "select_score": None,
                "elapsed_sec": 0.0,
                "test_nll": dense_metrics["nll"],
                "test_ppl": dense_metrics["perplexity"],
                "test_teacher_kl": None,
                "split_provenance": exp_config["splits"],
                "device": device_meta,
            }
            rows.append(result)
            _save_basis(artifact_root, method, None, result)
            continue

        if method == "random":
            basis = methods["random"]
            search_meta = {"method": method}
            local_score = compute_local_score(local_output_matrix, output_weight, basis)
            select_score = selection_scorer(basis)
            elapsed_sec = 0.0
        elif method == "pca":
            basis = methods["pca"]
            search_meta = {"method": method, "fit_splits": fit_split_names}
            local_score = compute_local_score(local_output_matrix, output_weight, basis)
            select_score = selection_scorer(basis)
            elapsed_sec = 0.0
        elif method == "rpedr_m1":
            basis = m1_result.basis
            search_meta = m1_result.metadata
            local_score = m1_result.local_score
            select_score = m1_result.select_score
            elapsed_sec = m1_result.elapsed_sec
        elif method == "rpedr_single_best":
            basis = single_best_result.basis
            search_meta = single_best_result.metadata
            local_score = single_best_result.local_score
            select_score = single_best_result.select_score
            elapsed_sec = single_best_result.elapsed_sec
        elif method == "rpedr_full":
            basis = full_result.basis
            search_meta = full_result.metadata
            local_score = full_result.local_score
            select_score = full_result.select_score
            elapsed_sec = full_result.elapsed_sec
        else:
            raise ValueError(f"unknown method '{method}'")

        spec = CompressionSpec(layer_index=target_layer, rank=rank, bases={target_head: basis})
        compressed = adapter.apply_compression(spec).to(device)
        test_metrics = evaluate_causal_lm(
            compressed,
            named_splits[test_split_name],
            max_length=int(data_config["max_length"]),
            device=device,
            teacher_adapter=adapter,
        )
        result = {
            "method": method,
            "layer_index": target_layer,
            "head_index": target_head,
            "rank": rank,
            "rank_ratio": float(exp_config["target"]["rank_ratio"]),
            "local_score": local_score,
            "select_score": select_score,
            "elapsed_sec": elapsed_sec,
            "test_nll": test_metrics["nll"],
            "test_ppl": test_metrics["perplexity"],
            "test_teacher_kl": test_metrics.get("teacher_logit_kl"),
            "split_provenance": exp_config["splits"],
            "search_hparams": search_meta,
            "device": device_meta,
        }
        rows.append(result)
        _save_basis(artifact_root, method, basis, result)

    save_json(
        {
            "model_config": args.model_config,
            "data_config": args.data_config,
            "exp_config": args.exp_config,
            "device": device_meta,
            "rows": rows,
        },
        artifact_root / "summary.json",
    )
    with (artifact_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "layer_index",
                "head_index",
                "rank",
                "rank_ratio",
                "local_score",
                "select_score",
                "elapsed_sec",
                "test_nll",
                "test_ppl",
                "test_teacher_kl",
                "selected_device",
            ],
        )
        writer.writeheader()
        for row in rows:
            csv_row = {key: row.get(key) for key in writer.fieldnames}
            csv_row["selected_device"] = row["device"]["selected_device"]
            writer.writerow(csv_row)


if __name__ == "__main__":
    main()
