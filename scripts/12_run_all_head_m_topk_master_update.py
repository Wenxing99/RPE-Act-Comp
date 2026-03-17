from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import statistics
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


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


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


def _make_base_meta(exp_config: dict, model: str, heads: str) -> dict:
    return {
        "model": model,
        "layer": int(exp_config["target"]["layer_index"]),
        "heads": heads,
        "rank_ratio": float(exp_config["target"]["rank_ratio"]),
        "rank": 32,
        "L": int(exp_config["search"]["num_groups"]),
    }


def _add_master_row(rows: list[dict], *, base: dict, method: str, M: int | None, topk: int | None, seed_or_aggregate: str, source_run: str, artifact_dir: str, reused_or_fresh: str, nll: float | None, ppl: float | None, kl: float | None, nll_mean: float | None = None, nll_std: float | None = None, ppl_mean: float | None = None, ppl_std: float | None = None, kl_mean: float | None = None, kl_std: float | None = None, dense_nll: float | None = None, dense_ppl: float | None = None, pca_nll: float | None = None, pca_ppl: float | None = None) -> None:
    metric_nll = nll_mean if nll_mean is not None else nll
    metric_ppl = ppl_mean if ppl_mean is not None else ppl
    rows.append(
        {
            **base,
            "M": M,
            "topk": topk,
            "method": method,
            "seed_or_aggregate": seed_or_aggregate,
            "source_run": source_run,
            "artifact_dir": artifact_dir,
            "reused_or_fresh": reused_or_fresh,
            "S3_test_nll": nll,
            "S3_test_ppl": ppl,
            "S3_test_teacher_kl": kl,
            "S3_test_nll_mean": nll_mean,
            "S3_test_nll_std": nll_std,
            "S3_test_ppl_mean": ppl_mean,
            "S3_test_ppl_std": ppl_std,
            "S3_test_teacher_kl_mean": kl_mean,
            "S3_test_teacher_kl_std": kl_std,
            "delta_vs_dense_nll": None if metric_nll is None or dense_nll is None else metric_nll - dense_nll,
            "delta_vs_dense_ppl": None if metric_ppl is None or dense_ppl is None else metric_ppl - dense_ppl,
            "delta_vs_pca_nll": None if metric_nll is None or pca_nll is None else metric_nll - pca_nll,
            "delta_vs_pca_ppl": None if metric_ppl is None or pca_ppl is None else metric_ppl - pca_ppl,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model/distilgpt2_single_head.yaml")
    parser.add_argument("--data-config", default="configs/data/wikitext2_single_head_cached.yaml")
    parser.add_argument("--exp-config", default="configs/exp/all_head_m_topk_diagnosis_layer3_stage2.yaml")
    parser.add_argument("--refresh-only", action="store_true", help="Only aggregate existing cells into the master table; do not run missing cells.")
    args = parser.parse_args()

    exp_config = load_yaml(args.exp_config)
    artifact_root = ensure_dir(exp_config["artifact_root"])
    master_root = ensure_dir(exp_config["master_artifact_root"])
    reference_all_head_root = ROOT / exp_config["reference_all_head_run_artifact_root"]
    reference_diag_root = ROOT / exp_config["reference_m_topk_diagnosis_artifact_root"]
    runner_script = ROOT / "scripts" / "09_run_multi_head_replication.py"
    generated_config_dir = artifact_root / "generated_configs"
    generated_config_dir.mkdir(parents=True, exist_ok=True)

    grid_group_sizes = [int(value) for value in exp_config["grid"]["group_sizes"]]
    grid_topk_values = [int(value) for value in exp_config["grid"]["topk_values"]]
    valid_target_cells = [(M, topk) for M in grid_group_sizes for topk in grid_topk_values if topk <= M]

    all_head_summary = _read_csv_rows(reference_all_head_root / "summary.csv")
    model = all_head_summary[0]["model"]
    heads = all_head_summary[0]["heads"]
    base_meta = _make_base_meta(exp_config, model, heads)

    dense_row = next(row for row in all_head_summary if row["method"] == "dense")
    pca_row = next(row for row in all_head_summary if row["method"] == "pca")
    dense_nll = float(dense_row["S3_test_nll"])
    dense_ppl = float(dense_row["S3_test_ppl"])
    pca_nll = float(pca_row["S3_test_nll"])
    pca_ppl = float(pca_row["S3_test_ppl"])

    master_rows: list[dict] = []
    _add_master_row(master_rows, base=base_meta, method="dense", M=None, topk=None, seed_or_aggregate="aggregate", source_run=exp_config["reference_all_head_run_artifact_root"], artifact_dir=str(reference_all_head_root.as_posix()), reused_or_fresh="reused", nll=dense_nll, ppl=dense_ppl, kl=_to_float(dense_row["S3_test_teacher_kl"]), dense_nll=dense_nll, dense_ppl=dense_ppl, pca_nll=pca_nll, pca_ppl=pca_ppl)
    _add_master_row(master_rows, base=base_meta, method="pca", M=None, topk=None, seed_or_aggregate="aggregate", source_run=exp_config["reference_all_head_run_artifact_root"], artifact_dir=str(reference_all_head_root.as_posix()), reused_or_fresh="reused", nll=pca_nll, ppl=pca_ppl, kl=_to_float(pca_row["S3_test_teacher_kl"]), dense_nll=dense_nll, dense_ppl=dense_ppl, pca_nll=pca_nll, pca_ppl=pca_ppl)

    existing_cells: dict[tuple[int, int], dict] = {}

    for row in all_head_summary:
        if row["method"] != "rpedr_full":
            continue
        key = (int(row["M"]), int(row["topk"]))
        existing_cells.setdefault(key, {"summary_rows": [], "source_run": exp_config["reference_all_head_run_artifact_root"], "artifact_dir": str(reference_all_head_root.as_posix()), "reused_or_fresh": "reused"})
        existing_cells[key]["summary_rows"].append(row)

    diag_summary = _read_csv_rows(reference_diag_root / "rpedr_full_grid_summary.csv")
    diag_seed_rows = _read_csv_rows(reference_diag_root / "rpedr_full_grid_results.csv")
    for row in diag_summary:
        key = (int(row["group_size"]), int(row["topk"]))
        cell_dir = row["cell_artifact_root"]
        cell_seed_rows = [seed_row for seed_row in diag_seed_rows if int(seed_row["group_size"]) == key[0] and int(seed_row["topk"]) == key[1]]
        existing_cells[key] = {"summary_rows": cell_seed_rows, "summary_agg": row, "source_run": exp_config["reference_m_topk_diagnosis_artifact_root"], "artifact_dir": cell_dir, "reused_or_fresh": "reused"}


    stage2_cells_root = artifact_root / "cells"
    if stage2_cells_root.exists():
        for cell_dir in sorted(stage2_cells_root.iterdir()):
            if not cell_dir.is_dir():
                continue
            summary_path = cell_dir / "summary.csv"
            aggregate_path = cell_dir / "aggregate_summary.csv"
            if not summary_path.exists() or not aggregate_path.exists():
                continue
            name_parts = cell_dir.name.split("_topk")
            if len(name_parts) != 2 or not name_parts[0].startswith("M"):
                continue
            M = int(name_parts[0][1:])
            topk = int(name_parts[1])
            existing_cells[(M, topk)] = {
                "summary_rows": [row for row in _read_csv_rows(summary_path) if row["method"] == "rpedr_full"],
                "source_run": exp_config["artifact_root"],
                "artifact_dir": str(cell_dir.as_posix()),
                "reused_or_fresh": "fresh",
            }
    missing_cells = [cell for cell in valid_target_cells if cell not in existing_cells]

    methods = list(exp_config["methods"])
    seeds = [int(seed) for seed in exp_config["seeds"]]
    head_indices = [int(head) for head in exp_config["target"]["head_indices"]]
    layer_index = int(exp_config["target"]["layer_index"])
    rank_ratio = float(exp_config["target"]["rank_ratio"])
    num_groups = int(exp_config["search"]["num_groups"])
    execution_cfg = exp_config["execution"]

    newly_run_cells: list[dict] = []
    cells_to_run = [] if args.refresh_only else missing_cells

    for M, topk in cells_to_run:
        cell_name = f"M{M}_topk{topk}"
        cell_artifact_root = artifact_root / "cells" / cell_name
        if cell_artifact_root.exists():
            shutil.rmtree(cell_artifact_root)
        cell_config = {
            "artifact_root": str(cell_artifact_root.as_posix()),
            "target": {"layer_index": layer_index, "head_indices": head_indices, "rank_ratio": rank_ratio},
            "splits": exp_config["splits"],
            "baseline_fit_splits": exp_config["baseline_fit_splits"],
            "search": {"num_groups": num_groups, "group_size": M, "topk": topk},
            "methods": methods,
            "seeds": seeds,
            "execution": execution_cfg,
        }
        cell_config_path = generated_config_dir / f"{cell_name}.yaml"
        _write_cell_config(cell_config_path, cell_config)
        _run_cell(runner_script, args.model_config, args.data_config, cell_config_path)

        summary_rows = _read_csv_rows(cell_artifact_root / "summary.csv")
        aggregate_rows = _read_csv_rows(cell_artifact_root / "aggregate_summary.csv")
        agg_row = next(row for row in aggregate_rows if row["method"] == "rpedr_full" and row["metric"] == "S3_test_nll")
        existing_cells[(M, topk)] = {
            "summary_rows": [row for row in summary_rows if row["method"] == "rpedr_full"],
            "source_run": exp_config["artifact_root"],
            "artifact_dir": str(cell_artifact_root.as_posix()),
            "reused_or_fresh": "fresh",
        }
        newly_run_cells.append({"M": M, "topk": topk, "artifact_dir": str(cell_artifact_root.as_posix())})

    target_cell_summary_rows: list[dict] = []
    for M, topk in sorted(existing_cells.keys()):
        cell_info = existing_cells[(M, topk)]
        if "summary_agg" in cell_info:
            agg = cell_info["summary_agg"]
            nll_mean = float(agg["nll_mean"])
            nll_std = float(agg["nll_std"])
            ppl_mean = float(agg["ppl_mean"])
            ppl_std = float(agg["ppl_std"])
            kl_mean = float(agg["kl_mean"])
            kl_std = float(agg["kl_std"])
        else:
            summary_rows = cell_info["summary_rows"]
            nll_values = [float(row["S3_test_nll"]) for row in summary_rows]
            ppl_values = [float(row["S3_test_ppl"]) for row in summary_rows]
            kl_values = [_to_float(row["S3_test_teacher_kl"]) for row in summary_rows]
            nll_mean = statistics.mean(nll_values)
            nll_std = statistics.stdev(nll_values) if len(nll_values) > 1 else 0.0
            ppl_mean = statistics.mean(ppl_values)
            ppl_std = statistics.stdev(ppl_values) if len(ppl_values) > 1 else 0.0
            valid_kl_values = [value for value in kl_values if value is not None]
            kl_mean = statistics.mean(valid_kl_values) if valid_kl_values else None
            kl_std = statistics.stdev(valid_kl_values) if len(valid_kl_values) > 1 else 0.0

        for row in cell_info["summary_rows"]:
            _add_master_row(
                master_rows,
                base=base_meta,
                method="rpedr_full",
                M=M,
                topk=topk,
                seed_or_aggregate=f"seed{row['seed']}",
                source_run=cell_info["source_run"],
                artifact_dir=cell_info["artifact_dir"],
                reused_or_fresh=cell_info["reused_or_fresh"],
                nll=float(row["S3_test_nll"]),
                ppl=float(row["S3_test_ppl"]),
                kl=_to_float(row["S3_test_teacher_kl"]),
                dense_nll=dense_nll,
                dense_ppl=dense_ppl,
                pca_nll=pca_nll,
                pca_ppl=pca_ppl,
            )

        _add_master_row(
            master_rows,
            base=base_meta,
            method="rpedr_full",
            M=M,
            topk=topk,
            seed_or_aggregate="aggregate",
            source_run=cell_info["source_run"],
            artifact_dir=cell_info["artifact_dir"],
            reused_or_fresh=cell_info["reused_or_fresh"],
            nll=None,
            ppl=None,
            kl=None,
            nll_mean=nll_mean,
            nll_std=nll_std,
            ppl_mean=ppl_mean,
            ppl_std=ppl_std,
            kl_mean=kl_mean,
            kl_std=kl_std,
            dense_nll=dense_nll,
            dense_ppl=dense_ppl,
            pca_nll=pca_nll,
            pca_ppl=pca_ppl,
        )

        target_cell_summary_rows.append({
            "M": M,
            "topk": topk,
            "source_run": cell_info["source_run"],
            "artifact_dir": cell_info["artifact_dir"],
            "reused_or_fresh": cell_info["reused_or_fresh"],
            "S3_test_nll_mean": nll_mean,
            "S3_test_nll_std": nll_std,
            "S3_test_ppl_mean": ppl_mean,
            "S3_test_ppl_std": ppl_std,
            "S3_test_teacher_kl_mean": kl_mean,
            "S3_test_teacher_kl_std": kl_std,
            "delta_vs_dense_nll": nll_mean - dense_nll,
            "delta_vs_dense_ppl": ppl_mean - dense_ppl,
            "delta_vs_pca_nll": nll_mean - pca_nll,
            "delta_vs_pca_ppl": ppl_mean - pca_ppl,
        })

    with (master_root / "all_head_layer3_master_results.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "model", "layer", "heads", "rank_ratio", "rank", "L", "M", "topk", "method", "seed_or_aggregate", "source_run", "artifact_dir", "reused_or_fresh",
            "S3_test_nll", "S3_test_ppl", "S3_test_teacher_kl", "S3_test_nll_mean", "S3_test_nll_std", "S3_test_ppl_mean", "S3_test_ppl_std", "S3_test_teacher_kl_mean", "S3_test_teacher_kl_std",
            "delta_vs_dense_nll", "delta_vs_dense_ppl", "delta_vs_pca_nll", "delta_vs_pca_ppl",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in master_rows:
            writer.writerow(row)

    save_json(
        {
            "master_rows": master_rows,
            "target_grid_summary": target_cell_summary_rows,
            "refresh_only": args.refresh_only,
            "missing_cells": [{"M": M, "topk": topk} for M, topk in missing_cells],
            "reused_cells": [{"M": M, "topk": topk, "source_run": existing_cells[(M, topk)]["source_run"]} for M, topk in sorted(existing_cells.keys()) if existing_cells[(M, topk)]["reused_or_fresh"] == "reused"],
            "newly_run_cells": newly_run_cells,
            "reference_dense": dense_row,
            "reference_pca": pca_row,
        },
        master_root / "all_head_layer3_master_results.json",
    )


if __name__ == "__main__":
    main()


