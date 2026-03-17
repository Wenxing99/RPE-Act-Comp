from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.text_data import load_named_splits
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
    run_rpedr_single_best_and_full,
)
from models.gpt2_adapter import CompressionSpec, GPT2Adapter
from utils.config import ensure_dir, load_yaml, resolve_runtime_device
from utils.io import save_json, save_pt


def _combine_splits(named_splits: dict[str, list[str]], split_names: list[str]) -> list[str]:
    texts: list[str] = []
    for split_name in split_names:
        texts.extend(named_splits[split_name])
    return texts


def _rank_from_ratio(head_dim: int, rank_ratio: float) -> int:
    return max(1, min(head_dim, int(round(head_dim * rank_ratio))))


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


def _query_nvidia_smi() -> dict | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 4:
        return {"raw": completed.stdout.strip()}
    return {
        "name": parts[0],
        "memory_total_mb": int(parts[1]),
        "memory_used_mb": int(parts[2]),
        "utilization_gpu_pct": int(parts[3]),
    }


def _save_joint_spec(
    artifact_root: Path,
    method: str,
    seed: int | None,
    spec: CompressionSpec | None,
) -> Path | None:
    if spec is None:
        return None
    run_name = method if seed is None else f"{method}_seed{seed}"
    target_dir = artifact_root / "joint_specs" / run_name
    target_dir.mkdir(parents=True, exist_ok=True)
    spec_path = target_dir / "compression_spec.pt"
    save_pt(spec, spec_path)
    save_json(
        {
            "method": method,
            "seed": seed,
            "layer_index": spec.layer_index,
            "rank": spec.rank,
            "heads": sorted(spec.bases.keys()),
        },
        target_dir / "compression_spec.json",
    )
    return spec_path


def _save_per_head_result(
    artifact_root: Path,
    method: str,
    head_index: int,
    seed: int | None,
    payload: dict,
) -> None:
    run_name = method if seed is None else f"{method}_seed{seed}"
    target_dir = artifact_root / "per_head" / run_name / f"head_{head_index}"
    target_dir.mkdir(parents=True, exist_ok=True)
    save_pt(payload, target_dir / "basis.pt")
    save_json(payload["metadata"], target_dir / "result.json")


def _evaluate_joint(
    adapter: GPT2Adapter,
    texts: list[str],
    *,
    max_length: int,
    device: str,
    spec: CompressionSpec | None,
) -> dict:
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
    for method in stochastic_methods:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        for metric_key in ["S3_test_nll", "S3_test_ppl", "S3_test_teacher_kl"]:
            values = [row[metric_key] for row in method_rows if row[metric_key] is not None]
            if not values:
                continue
            aggregates.append(
                {
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

def _search_worker(
    *,
    base_adapter: GPT2Adapter,
    device: str,
    device_meta: dict,
    local_bundle: dict,
    layer_index: int,
    head_index: int,
    seed: int,
    rank: int,
    rank_ratio: float,
    search_cfg: dict,
    select_texts: list[str],
    max_length: int,
    artifact_root: Path,
    needs_no_selection: bool,
    needs_single_best: bool,
    needs_full: bool,
    concurrency_level: int,
) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    worker_adapter = base_adapter.clone().to(device)
    local_output_matrix = extract_single_head_matrix(local_bundle, "output", layer_index, head_index).to(device)
    output_weight = worker_adapter.get_head_output_weight(layer_index, head_index)
    output_paths = {
        "rpedr_no_selection": artifact_root / "per_head" / f"rpedr_no_selection_seed{seed}" / f"head_{head_index}",
        "rpedr_single_best": artifact_root / "per_head" / f"rpedr_single_best_seed{seed}" / f"head_{head_index}",
        "rpedr_full": artifact_root / "per_head" / f"rpedr_full_seed{seed}" / f"head_{head_index}",
    }
    active_methods = [
        method_name
        for method_name, enabled in [
            ("rpedr_no_selection", needs_no_selection),
            ("rpedr_single_best", needs_single_best),
            ("rpedr_full", needs_full),
        ]
        if enabled
    ]
    worker_log = {
        "method": "+".join(active_methods),
        "head": head_index,
        "seed": seed,
        "selected_device": device_meta["selected_device"],
        "batch_settings": {
            "max_length": max_length,
            "select_text_count": len(select_texts),
            "num_groups": int(search_cfg["num_groups"]),
            "group_size": int(search_cfg["group_size"]),
            "topk": int(search_cfg["topk"]),
            "concurrency_level": concurrency_level,
        },
        "output_artifact_paths": {key: str(value.as_posix()) for key, value in output_paths.items()},
        "cuda_oom": False,
        "cpu_fallback": False,
        "started_at_utc": started_at,
        "ended_at_utc": None,
        "search_elapsed_sec": None,
    }

    try:
        scorer = make_teacher_kl_scorer(
            worker_adapter,
            layer_index=layer_index,
            head_index=head_index,
            texts=select_texts,
            max_length=max_length,
            device=device,
        )
        if needs_single_best and needs_full and not needs_no_selection:
            single_best_result, full_result = run_rpedr_single_best_and_full(
                local_activations=local_output_matrix,
                output_weight=output_weight,
                head_dim=worker_adapter.head_dim,
                rank=rank,
                seed=seed,
                scorer=scorer,
                num_groups=int(search_cfg["num_groups"]),
                group_size=int(search_cfg["group_size"]),
                topk=int(search_cfg["topk"]),
            )
            result_by_method = {
                "rpedr_single_best": single_best_result,
                "rpedr_full": full_result,
            }
        else:
            result_by_method = {}
            if needs_no_selection:
                result_by_method["rpedr_no_selection"] = run_rpedr_m1(
                    local_activations=local_output_matrix,
                    output_weight=output_weight,
                    head_dim=worker_adapter.head_dim,
                    rank=rank,
                    seed=seed,
                    scorer=scorer,
                    num_groups=int(search_cfg["num_groups"]),
                )
            if needs_single_best:
                result_by_method["rpedr_single_best"] = run_rpedr_single_best(
                    local_activations=local_output_matrix,
                    output_weight=output_weight,
                    head_dim=worker_adapter.head_dim,
                    rank=rank,
                    seed=seed,
                    scorer=scorer,
                    num_groups=int(search_cfg["num_groups"]),
                    group_size=int(search_cfg["group_size"]),
                    topk=int(search_cfg["topk"]),
                )
            if needs_full:
                result_by_method["rpedr_full"] = run_rpedr_full(
                    local_activations=local_output_matrix,
                    output_weight=output_weight,
                    head_dim=worker_adapter.head_dim,
                    rank=rank,
                    seed=seed,
                    scorer=scorer,
                    num_groups=int(search_cfg["num_groups"]),
                    group_size=int(search_cfg["group_size"]),
                    topk=int(search_cfg["topk"]),
                )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            worker_log["cuda_oom"] = True
        worker_log["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        raise RuntimeError(f"search worker failed for head={head_index} seed={seed}: {exc}") from exc

    worker_log["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload_by_method = {}
    for method_name, result in result_by_method.items():
        payload_by_method[method_name] = {
            "basis": result.basis.detach().cpu(),
            "metadata": {
                "method": method_name,
                "seed": seed,
                "layer_index": layer_index,
                "head_index": head_index,
                "rank_ratio": rank_ratio,
                "rank": rank,
                "local_score": result.local_score,
                "select_score": result.select_score,
                "elapsed_sec": result.elapsed_sec,
                "search_hparams": result.metadata,
                "selected_device": device_meta["selected_device"],
            },
        }
    if payload_by_method:
        first_payload = next(iter(payload_by_method.values()))
        worker_log["search_elapsed_sec"] = first_payload["metadata"]["elapsed_sec"]
    return {
        "head_index": head_index,
        "seed": seed,
        "worker_log": worker_log,
        "results": payload_by_method,
    }
def _run_search_jobs(
    *,
    adapter: GPT2Adapter,
    device: str,
    device_meta: dict,
    local_bundle: dict,
    layer_index: int,
    head_indices: list[int],
    seeds: list[int],
    rank: int,
    rank_ratio: float,
    search_cfg: dict,
    split_cfg: dict,
    named_splits: dict[str, list[str]],
    data_config: dict,
    artifact_root: Path,
    methods: list[str],
    concurrency_level: int,
    persist_artifacts: bool,
    search_jobs: list[tuple[int, int]] | None = None,
) -> tuple[dict[tuple[str, int], dict[int, object]], list[dict]]:
    stochastic_results: dict[tuple[str, int], dict[int, object]] = {}
    worker_logs: list[dict] = []
    needs_no_selection = "rpedr_no_selection" in methods
    needs_single_best = "rpedr_single_best" in methods
    needs_full = "rpedr_full" in methods

    if not (needs_no_selection or needs_single_best or needs_full):
        return stochastic_results, worker_logs

    for seed in seeds:
        if needs_no_selection:
            stochastic_results[("rpedr_no_selection", seed)] = {}
        if needs_single_best:
            stochastic_results[("rpedr_single_best", seed)] = {}
        if needs_full:
            stochastic_results[("rpedr_full", seed)] = {}

    if search_jobs is None:
        search_jobs = [(head_index, seed) for seed in seeds for head_index in head_indices]
    print(
        f"[search] concurrency={concurrency_level} jobs={len(search_jobs)} "
        f"layer={layer_index} heads={head_indices} seeds={seeds} persist={persist_artifacts}"
    )
    with ThreadPoolExecutor(max_workers=concurrency_level) as executor:
        future_map = {
            executor.submit(
                _search_worker,
                base_adapter=adapter,
                device=device,
                device_meta=device_meta,
                local_bundle=local_bundle,
                layer_index=layer_index,
                head_index=head_index,
                seed=seed,
                rank=rank,
                rank_ratio=rank_ratio,
                search_cfg=search_cfg,
                select_texts=named_splits[split_cfg["select"]],
                max_length=int(data_config["max_length"]),
                artifact_root=artifact_root,
                needs_no_selection=needs_no_selection,
                needs_single_best=needs_single_best,
                needs_full=needs_full,
                concurrency_level=concurrency_level,
            ): (head_index, seed)
            for head_index, seed in search_jobs
        }
        for future in as_completed(future_map):
            head_index, seed = future_map[future]
            print(f"[search-done] seed={seed} layer={layer_index} head={head_index}")
            worker_output = future.result()
            worker_logs.append(worker_output["worker_log"])
            for method_name, payload in worker_output["results"].items():
                stochastic_results[(method_name, seed)][head_index] = payload["basis"]
                if persist_artifacts:
                    _save_per_head_result(
                        artifact_root,
                        method=method_name,
                        head_index=head_index,
                        seed=seed,
                        payload=payload,
                    )

    return stochastic_results, worker_logs
def _autotune_search_workers(
    *,
    adapter: GPT2Adapter,
    device: str,
    device_meta: dict,
    local_bundle: dict,
    layer_index: int,
    head_indices: list[int],
    seeds: list[int],
    rank: int,
    rank_ratio: float,
    search_cfg: dict,
    split_cfg: dict,
    named_splits: dict[str, list[str]],
    data_config: dict,
    artifact_root: Path,
    methods: list[str],
    execution_cfg: dict,
) -> tuple[int, dict]:
    autotune_cfg = execution_cfg.get("autotune", {})
    max_search_workers = min(10, int(autotune_cfg.get("max_search_workers", 10)))
    base_candidates = [int(worker) for worker in autotune_cfg.get("candidate_workers", [2, 4, 6])]
    candidate_workers = sorted({worker for worker in base_candidates if 1 <= worker <= max_search_workers})
    default_warmup_job_limit = min(2, len(head_indices) * len(seeds))
    warmup_job_limit = int(autotune_cfg.get("warmup_job_limit", default_warmup_job_limit))
    warmup_search_jobs = [(head_index, seed) for seed in seeds for head_index in head_indices][:warmup_job_limit]
    if not candidate_workers:
        raise ValueError("autotune candidate_workers must contain at least one positive integer")
    if not warmup_search_jobs:
        raise ValueError("autotune warmup produced no search jobs")

    autotune_dir = artifact_root / "autotune"
    autotune_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[autotune] base_candidates={candidate_workers} warmup_job_limit={warmup_job_limit} "
        f"warmup_jobs={warmup_search_jobs} max_search_workers={max_search_workers}"
    )

    candidate_rows: list[dict] = []
    tested_workers: set[int] = set()
    pending_workers = list(candidate_workers)
    expansion_trace: list[dict] = []

    def _measure_candidate(worker_count: int) -> dict:
        gpu_before = _query_nvidia_smi()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        _, worker_logs = _run_search_jobs(
            adapter=adapter,
            device=device,
            device_meta=device_meta,
            local_bundle=local_bundle,
            layer_index=layer_index,
            head_indices=head_indices,
            seeds=seeds,
            rank=rank,
            rank_ratio=rank_ratio,
            search_cfg=search_cfg,
            split_cfg=split_cfg,
            named_splits=named_splits,
            data_config=data_config,
            artifact_root=autotune_dir,
            methods=methods,
            concurrency_level=worker_count,
            persist_artifacts=False,
            search_jobs=warmup_search_jobs,
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            peak_memory_bytes = int(torch.cuda.max_memory_allocated())
        else:
            peak_memory_bytes = None
        elapsed_sec = time.perf_counter() - started
        gpu_after = _query_nvidia_smi()
        return {
            "search_workers": worker_count,
            "warmup_job_count": len(warmup_search_jobs),
            "warmup_jobs": warmup_search_jobs,
            "elapsed_sec": elapsed_sec,
            "cuda_oom": any(log["cuda_oom"] for log in worker_logs),
            "cpu_fallback": any(log["cpu_fallback"] for log in worker_logs),
            "worker_elapsed_sec": [log.get("search_elapsed_sec") for log in worker_logs],
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "peak_cuda_memory_bytes": peak_memory_bytes,
            "worker_log_count": len(worker_logs),
        }

    while pending_workers:
        worker_count = pending_workers.pop(0)
        if worker_count in tested_workers:
            continue
        tested_workers.add(worker_count)
        candidate_rows.append(_measure_candidate(worker_count))

        safe_rows = [row for row in candidate_rows if not row["cuda_oom"] and not row["cpu_fallback"]]
        if not safe_rows:
            continue
        best_safe = min(safe_rows, key=lambda row: (row["elapsed_sec"], row["search_workers"]))
        current_upper = max(tested_workers)
        expanded = False
        next_candidate = None
        if best_safe["search_workers"] == current_upper and current_upper < max_search_workers:
            next_candidate = min(max_search_workers, current_upper + 2)
            if next_candidate not in tested_workers and next_candidate not in pending_workers:
                pending_workers.append(next_candidate)
                expanded = True
        expansion_trace.append(
            {
                "tested_workers": sorted(tested_workers),
                "best_safe_workers": best_safe["search_workers"],
                "current_upper_boundary": current_upper,
                "expanded_upward": expanded,
                "next_candidate": next_candidate,
            }
        )

    safe_rows = [row for row in candidate_rows if not row["cuda_oom"] and not row["cpu_fallback"]]
    if not safe_rows:
        raise RuntimeError("autotune found no safe candidate worker counts")

    chosen = min(safe_rows, key=lambda row: (row["elapsed_sec"], row["search_workers"]))
    summary = {
        "enabled": True,
        "mode": "pre_run_fixed_worker_autotune",
        "selection_rule": "Choose the safe candidate with the shortest warmup wall-clock; discard any candidate with CUDA OOM or CPU fallback; break ties by smaller search_workers.",
        "boundary_expansion_rule": "Start from the configured discrete candidate set. If the current best safe candidate equals the current upper tested boundary, test exactly one additional higher candidate at +2 workers. Repeat only while the best safe candidate remains on the upper boundary, and never exceed 10 search_workers.",
        "candidate_workers": candidate_workers,
        "max_search_workers": max_search_workers,
        "warmup_job_limit": warmup_job_limit,
        "warmup_jobs": warmup_search_jobs,
        "candidate_results": candidate_rows,
        "expansion_trace": expansion_trace,
        "chosen_workers": chosen["search_workers"],
    }
    save_json(summary, autotune_dir / "autotune_summary.json")
    with (autotune_dir / "autotune_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "search_workers",
            "warmup_job_count",
            "elapsed_sec",
            "cuda_oom",
            "cpu_fallback",
            "peak_cuda_memory_bytes",
            "worker_log_count",
            "warmup_jobs",
            "worker_elapsed_sec",
            "gpu_before",
            "gpu_after",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in candidate_rows:
            csv_row = dict(row)
            csv_row["warmup_jobs"] = json.dumps(row["warmup_jobs"])
            csv_row["worker_elapsed_sec"] = json.dumps(row["worker_elapsed_sec"])
            csv_row["gpu_before"] = json.dumps(row["gpu_before"])
            csv_row["gpu_after"] = json.dumps(row["gpu_after"])
            writer.writerow(csv_row)
    return chosen["search_workers"], summary
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model/distilgpt2_single_head.yaml")
    parser.add_argument("--data-config", default="configs/data/wikitext2_single_head.yaml")
    parser.add_argument(
        "--exp-config",
        default="configs/exp/multi_head_replication_distilgpt2_round1.yaml",
    )
    parser.add_argument("--autotune-only", action="store_true")
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
    head_indices = [int(head) for head in exp_config["target"]["head_indices"]]
    rank_ratio = float(exp_config["target"]["rank_ratio"])
    rank = _rank_from_ratio(adapter.head_dim, rank_ratio)
    split_cfg = exp_config["splits"]
    fit_split_names = list(exp_config["baseline_fit_splits"])
    methods = list(exp_config["methods"])
    seeds = [int(seed) for seed in exp_config.get("seeds", [])]
    search_cfg = exp_config["search"]
    execution_cfg = exp_config.get("execution", {})
    requested_search_workers = int(execution_cfg.get("search_workers", 2))

    local_bundle = collect_head_activations(
        adapter,
        named_splits[split_cfg["local"]],
        max_length=int(data_config["max_length"]),
        device=device,
    )

    chosen_search_workers = requested_search_workers
    autotune_summary = None
    autotune_enabled = bool(execution_cfg.get("autotune", {}).get("enabled", False))
    if ("rpedr_no_selection" in methods or "rpedr_single_best" in methods or "rpedr_full" in methods) and autotune_enabled:
        chosen_search_workers, autotune_summary = _autotune_search_workers(
            adapter=adapter,
            device=device,
            device_meta=device_meta,
            local_bundle=local_bundle,
            layer_index=layer_index,
            head_indices=head_indices,
            seeds=seeds,
            rank=rank,
            rank_ratio=rank_ratio,
            search_cfg=search_cfg,
            split_cfg=split_cfg,
            named_splits=named_splits,
            data_config=data_config,
            artifact_root=artifact_root,
            methods=methods,
            execution_cfg=execution_cfg,
        )
        print(f"[autotune-chosen] search_workers={chosen_search_workers}")

    if args.autotune_only:
        print("[autotune-only] completed pre-run autotune; skipping formal run")
        return

    fit_bundle = collect_head_activations(
        adapter,
        _combine_splits(named_splits, fit_split_names),
        max_length=int(data_config["max_length"]),
        device=device,
    )

    dense_metrics = _evaluate_joint(
        adapter,
        named_splits[split_cfg["test"]],
        max_length=int(data_config["max_length"]),
        device=device,
        spec=None,
    )
    save_json(dense_metrics, artifact_root / "evals" / "dense" / "eval.json")

    rows: list[dict] = []
    rows.append(
        {
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
            "final_report_split": split_cfg["test"],
            "selected_device": device_meta["selected_device"],
            "S3_test_nll": dense_metrics["nll"],
            "S3_test_ppl": dense_metrics["perplexity"],
            "S3_test_teacher_kl": dense_metrics.get("teacher_logit_kl"),
            "joint_spec_path": None,
            "eval_path": str((artifact_root / "evals" / "dense" / "eval.json").as_posix()),
            "stochastic": False,
        }
    )

    if "random" in methods:
        for seed in seeds:
            bases: dict[int, object] = {}
            for head_index in head_indices:
                local_output_matrix = extract_single_head_matrix(local_bundle, "output", layer_index, head_index).to(device)
                output_weight = adapter.get_head_output_weight(layer_index, head_index)
                basis = build_single_head_random_basis(adapter.head_dim, rank, seed=seed + head_index)
                random_select_scorer = make_teacher_kl_scorer(
                    adapter,
                    layer_index=layer_index,
                    head_index=head_index,
                    texts=named_splits[split_cfg["select"]],
                    max_length=int(data_config["max_length"]),
                    device=device,
                )
                metadata = {
                    "method": "random",
                    "seed": seed,
                    "layer_index": layer_index,
                    "head_index": head_index,
                    "rank_ratio": rank_ratio,
                    "rank": rank,
                    "local_score": compute_local_score(local_output_matrix, output_weight, basis),
                    "select_score": random_select_scorer(basis),
                }
                bases[head_index] = basis
                _save_per_head_result(
                    artifact_root,
                    method="random",
                    head_index=head_index,
                    seed=seed,
                    payload={"basis": basis, "metadata": metadata},
                )

            spec = CompressionSpec(layer_index=layer_index, rank=rank, bases=bases)
            spec_path = _save_joint_spec(artifact_root, "random", seed, spec)
            metrics = _evaluate_joint(
                adapter,
                named_splits[split_cfg["test"]],
                max_length=int(data_config["max_length"]),
                device=device,
                spec=spec,
            )
            eval_path = artifact_root / "evals" / f"random_seed{seed}" / "eval.json"
            save_json(metrics, eval_path)
            rows.append(
                {
                    "method": "random",
                    "seed": seed,
                    "model": model_config.get("pretrained_name", model_config.get("model_type", "unknown")),
                    "layer": layer_index,
                    "heads": head_indices,
                    "rank_ratio": rank_ratio,
                    "rank": rank,
                    "L": None,
                    "M": None,
                    "topk": None,
                    "final_report_split": split_cfg["test"],
                    "selected_device": device_meta["selected_device"],
                    "S3_test_nll": metrics["nll"],
                    "S3_test_ppl": metrics["perplexity"],
                    "S3_test_teacher_kl": metrics.get("teacher_logit_kl"),
                    "joint_spec_path": str(spec_path.as_posix()) if spec_path is not None else None,
                    "eval_path": str(eval_path.as_posix()),
                    "stochastic": True,
                }
            )
    if "pca" in methods:
        bases: dict[int, object] = {}
        for head_index in head_indices:
            fit_output_matrix = extract_single_head_matrix(fit_bundle, "output", layer_index, head_index).to(device)
            output_weight = adapter.get_head_output_weight(layer_index, head_index)
            basis = build_single_head_pca_basis(fit_output_matrix, rank)
            pca_select_scorer = make_teacher_kl_scorer(
                adapter,
                layer_index=layer_index,
                head_index=head_index,
                texts=named_splits[split_cfg["select"]],
                max_length=int(data_config["max_length"]),
                device=device,
            )
            metadata = {
                "method": "pca",
                "seed": None,
                "layer_index": layer_index,
                "head_index": head_index,
                "rank_ratio": rank_ratio,
                "rank": rank,
                "fit_splits": fit_split_names,
                "local_score": compute_local_score(
                    extract_single_head_matrix(local_bundle, "output", layer_index, head_index).to(device),
                    output_weight,
                    basis,
                ),
                "select_score": pca_select_scorer(basis),
            }
            bases[head_index] = basis
            _save_per_head_result(
                artifact_root,
                method="pca",
                head_index=head_index,
                seed=None,
                payload={"basis": basis, "metadata": metadata},
            )

        spec = CompressionSpec(layer_index=layer_index, rank=rank, bases=bases)
        spec_path = _save_joint_spec(artifact_root, "pca", None, spec)
        metrics = _evaluate_joint(
            adapter,
            named_splits[split_cfg["test"]],
            max_length=int(data_config["max_length"]),
            device=device,
            spec=spec,
        )
        eval_path = artifact_root / "evals" / "pca" / "eval.json"
        save_json(metrics, eval_path)
        rows.append(
            {
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
                "final_report_split": split_cfg["test"],
                "selected_device": device_meta["selected_device"],
                "S3_test_nll": metrics["nll"],
                "S3_test_ppl": metrics["perplexity"],
                "S3_test_teacher_kl": metrics.get("teacher_logit_kl"),
                "joint_spec_path": str(spec_path.as_posix()) if spec_path is not None else None,
                "eval_path": str(eval_path.as_posix()),
                "stochastic": False,
            }
        )

    stochastic_results, worker_logs = _run_search_jobs(
        adapter=adapter,
        device=device,
        device_meta=device_meta,
        local_bundle=local_bundle,
        layer_index=layer_index,
        head_indices=head_indices,
        seeds=seeds,
        rank=rank,
        rank_ratio=rank_ratio,
        search_cfg=search_cfg,
        split_cfg=split_cfg,
        named_splits=named_splits,
        data_config=data_config,
        artifact_root=artifact_root,
        methods=methods,
        concurrency_level=chosen_search_workers,
        persist_artifacts=True,
    )

    for method_name in ["rpedr_no_selection", "rpedr_single_best", "rpedr_full"]:
        if method_name not in methods:
            continue
        for seed in seeds:
            bases = stochastic_results[(method_name, seed)]
            spec = CompressionSpec(layer_index=layer_index, rank=rank, bases=bases)
            spec_path = _save_joint_spec(artifact_root, method_name, seed, spec)
            metrics = _evaluate_joint(
                adapter,
                named_splits[split_cfg["test"]],
                max_length=int(data_config["max_length"]),
                device=device,
                spec=spec,
            )
            eval_path = artifact_root / "evals" / f"{method_name}_seed{seed}" / "eval.json"
            save_json(metrics, eval_path)
            rows.append(
                {
                    "method": method_name,
                    "seed": seed,
                    "model": model_config.get("pretrained_name", model_config.get("model_type", "unknown")),
                    "layer": layer_index,
                    "heads": head_indices,
                    "rank_ratio": rank_ratio,
                    "rank": rank,
                    "L": int(search_cfg["num_groups"]),
                    "M": 1 if method_name == "rpedr_no_selection" else int(search_cfg["group_size"]),
                    "topk": 1 if method_name == "rpedr_no_selection" else int(search_cfg["topk"]),
                    "final_report_split": split_cfg["test"],
                    "selected_device": device_meta["selected_device"],
                    "S3_test_nll": metrics["nll"],
                    "S3_test_ppl": metrics["perplexity"],
                    "S3_test_teacher_kl": metrics.get("teacher_logit_kl"),
                    "joint_spec_path": str(spec_path.as_posix()) if spec_path is not None else None,
                    "eval_path": str(eval_path.as_posix()),
                    "stochastic": True,
                }
            )

    aggregate_rows = _aggregate_rows(rows, stochastic_methods=[method for method in ["random", "rpedr_no_selection", "rpedr_single_best", "rpedr_full"] if method in methods])

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "model_config": args.model_config,
        "data_config": args.data_config,
        "exp_config": args.exp_config,
        "model": model_config.get("pretrained_name", model_config.get("model_type", "unknown")),
        "layer": layer_index,
        "heads": head_indices,
        "methods": methods,
        "seeds": seeds,
        "budget": {
            "rank_ratio": rank_ratio,
            "rank": rank,
            "L": int(search_cfg["num_groups"]),
            "M": int(search_cfg["group_size"]),
            "topk": int(search_cfg["topk"]),
        },
        "splits": split_cfg,
        "baseline_fit_splits": fit_split_names,
        "device": device_meta,
        "execution": {
            "requested_search_workers": requested_search_workers,
            "search_workers": chosen_search_workers,
            "autotune_enabled": autotune_enabled,
            "autotune_candidate_workers": execution_cfg.get("autotune", {}).get("candidate_workers"),
            "autotune_selection_rule": None if autotune_summary is None else autotune_summary["selection_rule"],
            "autotune_chosen_workers": None if autotune_summary is None else autotune_summary["chosen_workers"],
        },
        "sanity_checks": {
            "unified_eval_path": "evaluate_causal_lm is used for dense and all compressed variants",
            "targeted_layer": layer_index,
            "targeted_heads": head_indices,
            "no_S3_in_search": True,
            "pre_run_fixed_worker_autotune_only": True,
        },
    }
    save_json(manifest, artifact_root / "run_manifest.json")
    save_json({"rows": rows}, artifact_root / "summary.json")
    save_json({"rows": aggregate_rows}, artifact_root / "aggregate_summary.json")
    save_json({"rows": worker_logs}, artifact_root / "worker_log.json")

    with (artifact_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
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
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["heads"] = ",".join(str(head) for head in row["heads"])
            writer.writerow(csv_row)

    with (artifact_root / "aggregate_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["method", "metric", "count", "mean", "std", "min", "max"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregate_rows:
            writer.writerow(row)

    with (artifact_root / "worker_log.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "method",
            "head",
            "seed",
            "selected_device",
            "batch_settings",
            "output_artifact_paths",
            "cuda_oom",
            "cpu_fallback",
            "search_elapsed_sec",
            "started_at_utc",
            "ended_at_utc",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in worker_logs:
            csv_row = dict(row)
            csv_row["batch_settings"] = str(row["batch_settings"])
            csv_row["output_artifact_paths"] = str(row["output_artifact_paths"])
            writer.writerow(csv_row)


if __name__ == "__main__":
    main()


















