from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.text_data import get_text_splits
from experiments.collect_activations import collect_head_activations
from models.gpt2_adapter import GPT2Adapter
from utils.config import ensure_dir, load_yaml
from utils.io import save_json, save_pt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model/tiny_random_gpt2.yaml")
    parser.add_argument("--data-config", default="configs/data/tiny_corpus.yaml")
    parser.add_argument("--artifact-root", default="results/demo")
    args = parser.parse_args()

    model_config = load_yaml(args.model_config)
    data_config = load_yaml(args.data_config)
    calibration_texts, _ = get_text_splits(data_config)
    adapter = GPT2Adapter.from_config(model_config).to(model_config.get("device", "cpu"))

    bundle = collect_head_activations(
        adapter,
        calibration_texts,
        max_length=int(data_config["max_length"]),
        device=model_config.get("device", "cpu"),
    )

    artifact_root = ensure_dir(args.artifact_root)
    save_pt(bundle, artifact_root / "head_activations.pt")
    save_json(bundle["metadata"], artifact_root / "head_activations_meta.json")


if __name__ == "__main__":
    main()
