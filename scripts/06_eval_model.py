from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.text_data import get_text_splits
from experiments.evaluate_lm import evaluate_causal_lm
from models.gpt2_adapter import GPT2Adapter
from utils.config import ensure_dir, load_yaml, resolve_runtime_device
from utils.io import load_pt, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model/tiny_random_gpt2.yaml")
    parser.add_argument("--data-config", default="configs/data/tiny_corpus.yaml")
    parser.add_argument("--artifact-root", default="results/demo")
    parser.add_argument("--compression-spec", default=None)
    parser.add_argument("--teacher-kl", action="store_true")
    args = parser.parse_args()

    model_config = load_yaml(args.model_config)
    data_config = load_yaml(args.data_config)
    _, evaluation_texts = get_text_splits(data_config)
    device, device_meta = resolve_runtime_device(model_config.get("device"))
    print(f"[device] requested={device_meta['requested_device']} selected={device_meta['selected_device']} cuda_available={device_meta['cuda_available']}")

    dense_adapter = GPT2Adapter.from_config(model_config).to(device)
    eval_adapter = dense_adapter
    if args.compression_spec:
        spec = load_pt(args.compression_spec)
        eval_adapter = dense_adapter.apply_compression(spec).to(device)

    teacher_adapter = dense_adapter if args.teacher_kl and args.compression_spec else None
    metrics = evaluate_causal_lm(
        eval_adapter,
        evaluation_texts,
        max_length=int(data_config["max_length"]),
        device=device,
        teacher_adapter=teacher_adapter,
    )
    metrics["device"] = device_meta

    artifact_root = ensure_dir(args.artifact_root)
    suffix = "compressed" if args.compression_spec else "dense"
    save_json(metrics, artifact_root / f"eval_{suffix}.json")


if __name__ == "__main__":
    main()
