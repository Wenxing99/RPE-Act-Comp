from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.build_bases import build_random_bases
from scoring.reconstruction import reconstruction_mse
from utils.config import load_yaml
from utils.io import load_pt, save_json, save_pt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-config", default="configs/exp/baseline_demo.yaml")
    args = parser.parse_args()

    exp_config = load_yaml(args.exp_config)
    artifact_root = Path(exp_config["artifact_root"])
    bundle = load_pt(artifact_root / "head_activations.pt")
    layer_index = int(exp_config["target_layer"])
    rank = int(exp_config["rank"])
    seed = int(exp_config["seed"])
    bases = build_random_bases(bundle, layer_index=layer_index, rank=rank, seed=seed)

    layer_outputs = bundle["activations"]["output"][layer_index]
    summary = {
        str(head_index): reconstruction_mse(layer_outputs[:, head_index, :], basis)
        for head_index, basis in bases.items()
    }

    save_pt(
        {
            "method": "random",
            "layer_index": layer_index,
            "rank": rank,
            "seed": seed,
            "bases": bases,
        },
        artifact_root / "random_bases.pt",
    )
    save_json(
        {
            "method": "random",
            "layer_index": layer_index,
            "rank": rank,
            "seed": seed,
            "reconstruction_mse": summary,
        },
        artifact_root / "random_summary.json",
    )


if __name__ == "__main__":
    main()
