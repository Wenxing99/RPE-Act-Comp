from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.text_data import load_named_splits
from experiments.evaluate_lm import evaluate_causal_lm
from models.gpt2_adapter import CompressionSpec, GPT2Adapter
from utils.config import ensure_dir, load_yaml, resolve_runtime_device
from utils.io import load_pt, save_json, save_pt


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None


def _rank_from_ratio(head_dim: int, rank_ratio: float) -> int:
    return max(1, min(head_dim, int(round(head_dim * rank_ratio))))


def _load_basis(round_dir: Path, method: str, head_index: int, seed: int | None) -> torch.Tensor:
    run_name = method if seed is None else f"{method}_seed{seed}"
    payload = load_pt(round_dir / "per_head" / run_name / f"head_{head_index}" / "basis.pt")
    return payload["basis"].detach().cpu()


def _evaluate_joint(adapter: GPT2Adapter, texts: list[str], *, max_length: int, device: str, spec: CompressionSpec | None) -> dict:
    eval_adapter = adapter if spec is None else adapter.apply_compression(spec).to(device)
    teacher_adapter = None if spec is None else adapter
    return evaluate_causal_lm(
        eval_adapter,
        texts,
        max_length=max_length,
        device=device,
        teacher_adapter=teacher_adapter,
    )


def _aggregate_rows(rows: list[dict], stochastic_methods: list[str]) -> list[dict]:
    aggregates: list[dict] = []
    for subset_name in sorted({row["subset_name"] for row in rows}):
        for method in stochastic_methods:
            method_rows = [row for row in rows if row["subset_name"] == subset_name and row["method"] == method]
            if not method_rows:
                continue
            for metric_key in ["S3_test_nll", "S3_test_ppl", "S3_test_teacher_kl"]:
                values = [row[metric_key] for row in method_rows if row[metric_key] is not None]
                if not values:
                    continue
                aggregates.append(
                    {
                        "subset_name": subset_name,
                        "method": method,
                        "metric": metric_key,
                        "count": len(values),
                        "mean": mean(values),
                        "std": pstdev(values) if len(values) > 1 else 0.0,
                        "min": min(values),
                        "max": max(values),
                    }
                )
    return aggregates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model/distilgpt2_single_head.yaml")
    parser.add_argument("--data-config", default="configs/data/wikitext2_single_head_cached.yaml")
    parser.add_argument("--exp-config", default="configs/exp/multi_head_subset_diagnosis_layer3.yaml")
    args = parser.parse_args()

    model_config = load_yaml(args.model_config)
    data_config = load_yaml(args.data_config)
    exp_config = load_yaml(args.exp_config)

    artifact_root = ensure_dir(exp_config["artifact_root"])
    named_splits = load_named_splits(data_config)
    device, device_meta = resolve_runtime_device(model_config.get("device"))
    print(
        f"[device] requested={device_meta['requested_device']} "
        f"selected={device_meta['selected_device']} cuda_available={device_meta['cuda_available']}"
    )
    adapter = GPT2Adapter.from_config(model_config).to(device)

    layer_index = int(exp_config["target"]["layer_index"])
    rank_ratio = float(exp_config["target"]["rank_ratio"])
    rank = _rank_from_ratio(adapter.head_dim, rank_ratio)
    subset_specs = list(exp_config["subset_specs"])
    methods = list(exp_config["methods"])
    seeds = [int(seed) for seed in exp_config.get("seeds", [])]
    source_head_rounds = {int(k): ROOT / v for k, v in exp_config["source_head_rounds"].items()}
    test_texts = named_splits[exp_config["splits"]["test"]]
    max_length = int(data_config["max_length"])

    dense_metrics = _evaluate_joint(adapter, test_texts, max_length=max_length, device=device, spec=None)
    save_json(dense_metrics, artifact_root / "evals" / "dense" / "eval.json")

    rows: list[dict] = []
    source_manifest = {str(head): str(path.as_posix()) for head, path in source_head_rounds.items()}
    save_json(source_manifest, artifact_root / "source_head_rounds.json")

    for subset_spec in subset_specs:
        subset_name = subset_spec["name"]
        head_indices = [int(head) for head in subset_spec["head_indices"]]
        print(f"[subset] name={subset_name} heads={head_indices}")
        rows.append(
            {
                "subset_name": subset_name,
                "method": "dense",
                "seed": None,
                "model": model_config.get("pretrained_name", model_config.get("model_type", "unknown")),
                "layer": layer_index,
                "heads": head_indices,
                "rank_ratio": 1.0,
                "rank": adapter.head_dim,
                "L": None,
                "M": None,
                "topk": None,
                "final_report_split": exp_config["splits"]["test"],
                "selected_device": device_meta["selected_device"],
                "S3_test_nll": dense_metrics["nll"],
                "S3_test_ppl": dense_metrics["perplexity"],
                "S3_test_teacher_kl": dense_metrics.get("teacher_logit_kl"),
                "joint_spec_path": None,
                "eval_path": str((artifact_root / "evals" / "dense" / "eval.json").as_posix()),
                "stochastic": False,
                "source_rounds_by_head": json.dumps({str(head): str(source_head_rounds[head].relative_to(ROOT).as_posix()) for head in head_indices}),
                "rationale": subset_spec.get("rationale"),
            }
        )

        if "pca" in methods:
            pca_bases = {head: _load_basis(source_head_rounds[head], "pca", head, None) for head in head_indices}
            spec = CompressionSpec(layer_index=layer_index, rank=rank, bases=pca_bases)
            spec_dir = artifact_root / "joint_specs" / subset_name / "pca"
            spec_dir.mkdir(parents=True, exist_ok=True)
            spec_path = spec_dir / "compression_spec.pt"
            save_pt(spec, spec_path)
            save_json({"heads": head_indices, "method": "pca", "subset_name": subset_name}, spec_dir / "compression_spec.json")
            metrics = _evaluate_joint(adapter, test_texts, max_length=max_length, device=device, spec=spec)
            eval_path = artifact_root / "evals" / subset_name / "pca" / "eval.json"
            save_json(metrics, eval_path)
            rows.append(
                {
                    "subset_name": subset_name,
                    "method": "pca",
                    "seed": None,
                    "model": model_config.get("pretrained_name", model_config.get("model_type", "unknown")),
                    "layer": layer_index,
                    "heads": head_indices,
                    "rank_ratio": rank_ratio,
                    "rank": rank,
                    "L": None,
                    "M": None,
                    "topk": None,
                    "final_report_split": exp_config["splits"]["test"],
                    "selected_device": device_meta["selected_device"],
                    "S3_test_nll": metrics["nll"],
                    "S3_test_ppl": metrics["perplexity"],
                    "S3_test_teacher_kl": metrics.get("teacher_logit_kl"),
                    "joint_spec_path": str(spec_path.as_posix()),
                    "eval_path": str(eval_path.as_posix()),
                    "stochastic": False,
                    "source_rounds_by_head": json.dumps({str(head): str(source_head_rounds[head].relative_to(ROOT).as_posix()) for head in head_indices}),
                    "rationale": subset_spec.get("rationale"),
                }
            )

        for method_name in ["rpedr_single_best", "rpedr_full"]:
            if method_name not in methods:
                continue
            for seed in seeds:
                bases = {head: _load_basis(source_head_rounds[head], method_name, head, seed) for head in head_indices}
                spec = CompressionSpec(layer_index=layer_index, rank=rank, bases=bases)
                spec_dir = artifact_root / "joint_specs" / subset_name / f"{method_name}_seed{seed}"
                spec_dir.mkdir(parents=True, exist_ok=True)
                spec_path = spec_dir / "compression_spec.pt"
                save_pt(spec, spec_path)
                save_json({"heads": head_indices, "method": method_name, "seed": seed, "subset_name": subset_name}, spec_dir / "compression_spec.json")
                metrics = _evaluate_joint(adapter, test_texts, max_length=max_length, device=device, spec=spec)
                eval_path = artifact_root / "evals" / subset_name / f"{method_name}_seed{seed}" / "eval.json"
                save_json(metrics, eval_path)
                rows.append(
                    {
                        "subset_name": subset_name,
                        "method": method_name,
                        "seed": seed,
                        "model": model_config.get("pretrained_name", model_config.get("model_type", "unknown")),
                        "layer": layer_index,
                        "heads": head_indices,
                        "rank_ratio": rank_ratio,
                        "rank": rank,
                        "L": 256,
                        "M": 32,
                        "topk": 2,
                        "final_report_split": exp_config["splits"]["test"],
                        "selected_device": device_meta["selected_device"],
                        "S3_test_nll": metrics["nll"],
                        "S3_test_ppl": metrics["perplexity"],
                        "S3_test_teacher_kl": metrics.get("teacher_logit_kl"),
                        "joint_spec_path": str(spec_path.as_posix()),
                        "eval_path": str(eval_path.as_posix()),
                        "stochastic": True,
                        "source_rounds_by_head": json.dumps({str(head): str(source_head_rounds[head].relative_to(ROOT).as_posix()) for head in head_indices}),
                        "rationale": subset_spec.get("rationale"),
                    }
                )

    aggregate_rows = _aggregate_rows(rows, stochastic_methods=["rpedr_single_best", "rpedr_full"])
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "model_config": args.model_config,
        "data_config": args.data_config,
        "exp_config": args.exp_config,
        "model": model_config.get("pretrained_name", model_config.get("model_type", "unknown")),
        "layer": layer_index,
        "rank_ratio": rank_ratio,
        "rank": rank,
        "budget": {"L": 256, "M": 32, "topk": 2},
        "splits": exp_config["splits"],
        "methods": methods,
        "seeds": seeds,
        "subset_specs": subset_specs,
        "source_head_rounds": source_manifest,
        "reuse_existing_per_head_artifacts": True,
        "device": device_meta,
        "execution": {
            "new_search_performed": False,
            "autotune_applicable": False,
            "note": "This diagnosis reuses saved per-head bases from round1 and round2 and runs only joint evals.",
        },
    }
    save_json(manifest, artifact_root / "run_manifest.json")
    save_json({"rows": rows}, artifact_root / "summary.json")
    save_json({"rows": aggregate_rows}, artifact_root / "aggregate_summary.json")

    with (artifact_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "subset_name",
            "method",
            "seed",
            "model",
            "layer",
            "heads",
            "rank_ratio",
            "rank",
            "L",
            "M",
            "topk",
            "final_report_split",
            "selected_device",
            "S3_test_nll",
            "S3_test_ppl",
            "S3_test_teacher_kl",
            "joint_spec_path",
            "eval_path",
            "stochastic",
            "source_rounds_by_head",
            "rationale",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["heads"] = ",".join(str(head) for head in row["heads"])
            writer.writerow(csv_row)

    with (artifact_root / "aggregate_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["subset_name", "method", "metric", "count", "mean", "std", "min", "max"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregate_rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
