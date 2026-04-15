from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from thesis_rag.config import load_config
from thesis_rag.pipeline import build_indexes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build embeddings and an exact FAISS index.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--chunks-path", required=True, help="Path to chunks.parquet.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = build_indexes(load_config(args.config), Path(args.chunks_path))
    print(run_dir)


if __name__ == "__main__":
    main()
