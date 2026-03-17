from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.config import ensure_dir, load_yaml
from utils.io import save_json


def _read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_cell_config(path: Path, payload: dict) -> None:
    lines: list[str] = []
    lines.append(f"artifact_root: {payload['artifact_root']}")
    lines.append("target:")
    lines.append(f"  layer_index: {payload['target']['layer_index']}")
    lines.append("  head_indices:")
    for head_index in payload["target"]["head_indices"]:
        lines.append(f"    - {head_index}")
    lines.append(f"  rank_ratio: {payload['target']['rank_ratio']}")
    lines.append("splits:")
    for key, value in payload["splits"].items():
        lines.append(f"  {key}: {value}")
    lines.append("baseline_fit_splits:")
    for split_name in payload["baseline_fit_splits"]:
        lines.append(f"  - {split_name}")
    lines.append("search:")
    lines.append(f"  num_groups: {payload['search']['num_groups']}")
    lines.append(f"  group_size: {payload['search']['group_size']}")
    lines.append(f"  topk: {payload['search']['topk']}")
    lines.append("methods:")
    for method_name in payload["methods"]:
        lines.append(f"  - {method_name}")
    lines.append("seeds:")
    for seed in payload["seeds"]:
        lines.append(f"  - {seed}")
    lines.append("execution:")
    lines.append(f"  search_workers: {payload['execution']['search_workers']}")
    lines.append("  autotune:")
    lines.append(f"    enabled: {'true' if payload['execution']['autotune']['enabled'] else 'false'}")
    lines.append("    candidate_workers:")
    for worker in payload["execution"]["autotune"]["candidate_workers"]:
        lines.append(f"      - {worker}")
    lines.append(f"    warmup_job_limit: {payload['execution']['autotune']['warmup_job_limit']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_cell(runner_script: Path, model_config: str, data_config: str, cell_config_path: Path) -> None:
    command = [
        sys.executable,
        str(runner_script),
        "--model-config",
        model_config,
        "--data-config",
        data_config,
        "--exp-config",
        str(cell_config_path),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model/distilgpt2_single_head.yaml")
    parser.add_argument("--data-config", default="configs/data/wikitext2_single_head_cached.yaml")
    parser.add_argument("--exp-config", default="configs/exp/all_head_m_topk_diagnosis_layer3.yaml")
    args = parser.parse_args()

    exp_config = load_yaml(args.exp_config)
    artifact_root = ensure_dir(exp_config["artifact_root"])
    reference_root = ROOT / exp_config["reference_run_artifact_root"]
    runner_script = ROOT / "scripts" / "09_run_multi_head_replication.py"
    generated_config_dir = artifact_root / "generated_configs"
    generated_config_dir.mkdir(parents=True, exist_ok=True)

    methods = list(exp_config["methods"])
    seeds = [int(seed) for seed in exp_config["seeds"]]
    head_indices = [int(head) for head in exp_config["target"]["head_indices"]]
    layer_index = int(exp_config["target"]["layer_index"])
    rank_ratio = float(exp_config["target"]["rank_ratio"])
    num_groups = int(exp_config["search"]["num_groups"])
    execution_cfg = exp_config["execution"]
    group_sizes = [int(value) for value in exp_config["grid"]["group_sizes"]]
    topk_values = [int(value) for value in exp_config["grid"]["topk_values"]]
    valid_cells = [(group_size, topk) for group_size in group_sizes for topk in topk_values if topk <= group_size]

    result_rows: list[dict] = []
    cell_summaries: list[dict] = []

    for group_size, topk in valid_cells:
        cell_name = f"M{group_size}_topk{topk}"
        cell_artifact_root = artifact_root / "cells" / cell_name
        if cell_artifact_root.exists():
            shutil.rmtree(cell_artifact_root)

        cell_config = {
            "artifact_root": str(cell_artifact_root.as_posix()),
            "target": {
                "layer_index": layer_index,
                "head_indices": head_indices,
                "rank_ratio": rank_ratio,
            },
            "splits": exp_config["splits"],
            "baseline_fit_splits": exp_config["baseline_fit_splits"],
            "search": {
                "num_groups": num_groups,
                "group_size": group_size,
                "topk": topk,
            },
            "methods": methods,
            "seeds": seeds,
            "execution": execution_cfg,
        }
        cell_config_path = generated_config_dir / f"{cell_name}.yaml"
        _write_cell_config(cell_config_path, cell_config)
        _run_cell(runner_script, args.model_config, args.data_config, cell_config_path)

        summary_rows = _read_csv_rows(cell_artifact_root / "summary.csv")
        aggregate_rows = _read_csv_rows(cell_artifact_root / "aggregate_summary.csv")
        autotune_summary = json.loads((cell_artifact_root / "autotune" / "autotune_summary.json").read_text(encoding="utf-8"))
        aggregate_by_metric = {row["metric"]: row for row in aggregate_rows if row["method"] == "rpedr_full"}

        for row in summary_rows:
            if row["method"] != "rpedr_full":
                continue
            result_rows.append(
                {
                    "cell_name": cell_name,
                    "group_size": group_size,
                    "topk": topk,
                    "seed": row["seed"],
                    "S3_test_nll": row["S3_test_nll"],
                    "S3_test_ppl": row["S3_test_ppl"],
                    "S3_test_teacher_kl": row["S3_test_teacher_kl"],
                    "autotune_chosen_workers": autotune_summary["chosen_workers"],
                }
            )

        cell_summaries.append(
            {
                "cell_name": cell_name,
                "group_size": group_size,
                "topk": topk,
                "autotune_chosen_workers": autotune_summary["chosen_workers"],
                "autotune_candidate_elapsed_sec": {
                    str(candidate["search_workers"]): candidate["elapsed_sec"]
                    for candidate in autotune_summary["candidate_results"]
                },
                "nll_mean": aggregate_by_metric["S3_test_nll"]["mean"],
                "nll_std": aggregate_by_metric["S3_test_nll"]["std"],
                "ppl_mean": aggregate_by_metric["S3_test_ppl"]["mean"],
                "ppl_std": aggregate_by_metric["S3_test_ppl"]["std"],
                "kl_mean": aggregate_by_metric["S3_test_teacher_kl"]["mean"],
                "kl_std": aggregate_by_metric["S3_test_teacher_kl"]["std"],
                "cell_artifact_root": str(cell_artifact_root.as_posix()),
            }
        )

    with (artifact_root / "rpedr_full_grid_results.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "cell_name",
            "group_size",
            "topk",
            "seed",
            "S3_test_nll",
            "S3_test_ppl",
            "S3_test_teacher_kl",
            "autotune_chosen_workers",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result_rows:
            writer.writerow(row)

    with (artifact_root / "rpedr_full_grid_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "cell_name",
            "group_size",
            "topk",
            "autotune_chosen_workers",
            "autotune_candidate_elapsed_sec",
            "nll_mean",
            "nll_std",
            "ppl_mean",
            "ppl_std",
            "kl_mean",
            "kl_std",
            "cell_artifact_root",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in cell_summaries:
            csv_row = dict(row)
            csv_row["autotune_candidate_elapsed_sec"] = json.dumps(row["autotune_candidate_elapsed_sec"])
            writer.writerow(csv_row)

    reference_rows = _read_csv_rows(reference_root / "summary.csv")
    save_json(
        {
            "reference_run_artifact_root": str(reference_root.as_posix()),
            "reference_rows": [row for row in reference_rows if row["method"] in {"dense", "pca", "rpedr_full"}],
            "valid_cells": [{"M": group_size, "topk": topk} for group_size, topk in valid_cells],
        },
        artifact_root / "reference_summary.json",
    )


if __name__ == "__main__":
    main()
