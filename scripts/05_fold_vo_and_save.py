from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models.gpt2_adapter import CompressionSpec
from utils.config import ensure_dir
from utils.io import load_pt, save_json, save_pt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis-file", required=True)
    parser.add_argument("--artifact-root", default="results/demo")
    args = parser.parse_args()

    bundle = load_pt(args.basis_file)
    spec = CompressionSpec(
        layer_index=int(bundle["layer_index"]),
        rank=int(bundle["rank"]),
        bases=bundle["bases"],
    )

    artifact_root = ensure_dir(args.artifact_root)
    save_pt(spec, artifact_root / "compression_spec.pt")
    save_json(
        {
            "method": bundle["method"],
            "layer_index": spec.layer_index,
            "rank": spec.rank,
            "num_heads": len(spec.bases),
        },
        artifact_root / "compression_spec.json",
    )


if __name__ == "__main__":
    main()
