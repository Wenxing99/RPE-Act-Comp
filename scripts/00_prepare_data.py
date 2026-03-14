from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.text_data import get_text_splits
from utils.config import ensure_dir, load_yaml
from utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data/tiny_corpus.yaml")
    parser.add_argument("--artifact-root", default="results/demo")
    args = parser.parse_args()

    data_config = load_yaml(args.data_config)
    calibration, evaluation = get_text_splits(data_config)
    artifact_root = ensure_dir(args.artifact_root)
    save_json(
        {
            "calibration_texts": calibration,
            "evaluation_texts": evaluation,
            "max_length": int(data_config["max_length"]),
        },
        artifact_root / "prepared_data.json",
    )


if __name__ == "__main__":
    main()
