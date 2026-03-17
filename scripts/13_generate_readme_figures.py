from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
MASTER_CSV = ROOT / "results" / "all_head_layer3_master_table" / "all_head_layer3_master_results.csv"
OUTPUT_DIR = ROOT / "assets" / "readme"

TARGET_MS = [32, 64, 128, 256]
TARGET_TOPKS = [2, 4, 8, 16, 32]


def load_rows() -> list[dict[str, str]]:
    with MASTER_CSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row["seed_or_aggregate"] == "aggregate"
        and row["method"] in {"dense", "pca", "rpedr_full"}
    ]


def reference_row(rows: list[dict[str, str]], method: str) -> dict[str, str]:
    return next(row for row in rows if row["method"] == method and row["seed_or_aggregate"] == "aggregate")


def cell_lookup(rows: list[dict[str, str]]) -> dict[tuple[int, int], dict[str, str]]:
    return {
        (int(row["M"]), int(row["topk"])): row
        for row in rows
        if row["method"] == "rpedr_full" and row["M"] and row["topk"]
    }


def best_target_row(rows: list[dict[str, str]]) -> dict[str, str]:
    target_rows = [
        row
        for row in rows
        if row["method"] == "rpedr_full"
        and row["M"]
        and row["topk"]
        and int(row["M"]) in TARGET_MS
        and int(row["topk"]) in TARGET_TOPKS
    ]
    return min(target_rows, key=lambda row: float(row["S3_test_nll_mean"]))


def fmt_delta(value: float) -> str:
    return f"{value:+.3f}"


def plot_heatmap(rows: list[dict[str, str]], best_row: dict[str, str]) -> None:
    lookup = cell_lookup(rows)
    values: list[list[float]] = []
    for M in TARGET_MS:
        row_values = []
        for topk in TARGET_TOPKS:
            cell = lookup[(M, topk)]
            row_values.append(float(cell["delta_vs_pca_nll"]))
        values.append(row_values)

    flat = [value for row in values for value in row]
    vmax = max(abs(min(flat)), abs(max(flat)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(9, 5.8), constrained_layout=True)
    im = ax.imshow(values, cmap="RdYlGn_r", norm=norm)

    ax.set_xticks(range(len(TARGET_TOPKS)), [str(value) for value in TARGET_TOPKS])
    ax.set_yticks(range(len(TARGET_MS)), [str(value) for value in TARGET_MS])
    ax.set_xlabel("topk")
    ax.set_ylabel("M")
    ax.set_title("All-head layer-3 search: delta NLL vs PCA (lower is better)")

    for i, M in enumerate(TARGET_MS):
        for j, topk in enumerate(TARGET_TOPKS):
            value = values[i][j]
            color = "white" if abs(value) > vmax * 0.42 else "black"
            ax.text(j, i, fmt_delta(value), ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    best_M = int(best_row["M"])
    best_topk = int(best_row["topk"])
    best_i = TARGET_MS.index(best_M)
    best_j = TARGET_TOPKS.index(best_topk)
    ax.add_patch(Rectangle((best_j - 0.5, best_i - 0.5), 1, 1, fill=False, edgecolor="#111111", linewidth=2.5))
    ax.text(best_j, best_i - 0.36, "best", ha="center", va="bottom", fontsize=9, color="#111111")

    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label("RPEDR-full NLL - PCA NLL")

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(OUTPUT_DIR / "all_head_heatmap_delta_nll_vs_pca.svg", format="svg", dpi=200)
    plt.close(fig)


def plot_baseline_comparison(rows: list[dict[str, str]], best_row: dict[str, str]) -> None:
    dense = reference_row(rows, "dense")
    pca = reference_row(rows, "pca")
    labels = ["Dense", "PCA", "RPEDR-full\nbest"]
    values = [
        float(dense["S3_test_nll"]),
        float(pca["S3_test_nll"]),
        float(best_row["S3_test_nll_mean"]),
    ]
    colors = ["#9aa5b1", "#4c78a8", "#2d6a4f"]

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.set_ylabel("Held-out NLL")
    ax.set_title("Completed all-head layer-3 comparison")
    ax.set_ylim(min(values) - 0.01, max(values) + 0.01)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.0008, f"{value:.6f}", ha="center", va="bottom", fontsize=10)

    note = f"Best RPEDR-full cell: M={best_row['M']}, topk={best_row['topk']}"
    ax.text(0.98, 0.03, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color="#333333")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)

    fig.savefig(OUTPUT_DIR / "baseline_nll_comparison.svg", format="svg", dpi=200)
    plt.close(fig)


def plot_m256_profile(rows: list[dict[str, str]]) -> None:
    lookup = cell_lookup(rows)
    topks = [2, 4, 8, 16, 32]
    values = [float(lookup[(256, topk)]["S3_test_nll_mean"]) for topk in topks]
    dense = float(reference_row(rows, "dense")["S3_test_nll"])
    pca = float(reference_row(rows, "pca")["S3_test_nll"])

    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    ax.plot(topks, values, color="#2d6a4f", marker="o", linewidth=2.2, markersize=6)
    ax.axhline(dense, color="#9aa5b1", linestyle="--", linewidth=1.6, label=f"Dense ({dense:.6f})")
    ax.axhline(pca, color="#4c78a8", linestyle=":", linewidth=1.8, label=f"PCA ({pca:.6f})")

    best_index = values.index(min(values))
    ax.scatter([topks[best_index]], [values[best_index]], color="#111111", s=40, zorder=3)
    ax.annotate("best", (topks[best_index], values[best_index]), xytext=(8, -14), textcoords="offset points", fontsize=9)

    for x, y in zip(topks, values):
        ax.text(x, y + 0.0008, f"{y:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(topks)
    ax.set_xlabel("topk (M = 256)")
    ax.set_ylabel("Held-out NLL")
    ax.set_title("M=256 profile: moderate topk wins in the current tested slice")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False, loc="upper right")

    fig.savefig(OUTPUT_DIR / "m256_topk_profile_nll.svg", format="svg", dpi=200)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = aggregate_rows(load_rows())
    best_row = best_target_row(rows)
    plot_heatmap(rows, best_row)
    plot_baseline_comparison(rows, best_row)
    plot_m256_profile(rows)


if __name__ == "__main__":
    main()
