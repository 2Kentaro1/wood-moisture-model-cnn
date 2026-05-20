from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nir_mc.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Run wood NIR CNN embedding experiment.")
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--test-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--band", default=None)
    parser.add_argument("--channels", nargs="+", default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--target-transform", choices=["none", "log1p"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    from nir_mc.experiments.cnn_embedding_experiment import run_full_experiment

    config = Config().update(
        TRAIN_PATH=args.train_path,
        TEST_PATH=args.test_path,
        OUTPUT_DIR=args.output_dir,
        band=args.band,
        channels=args.channels,
        embedding_dim=args.embedding_dim,
        target_transform=args.target_transform,
        epochs=args.epochs,
        batch_size=args.batch_size,
        DEVICE=args.device,
    )
    result = run_full_experiment(config)
    print("Experiment finished")
    print(f"setting: {result['setting']}")
    print(f"output_dir: {result['output_dir']}")
    print(result["final_compare"])


if __name__ == "__main__":
    main()
