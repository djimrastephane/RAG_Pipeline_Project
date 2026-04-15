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
from thesis_rag.pipeline import retrieve_queries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dense, sparse, and hybrid retrieval.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--chunk-metadata-path", required=True, help="Path to chunks.parquet or chunk metadata parquet.")
    parser.add_argument("--faiss-index-path", required=True, help="Path to faiss.index.")
    parser.add_argument("--query-set-path", default="", help="Optional override path to query set JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    query_path = Path(args.query_set_path) if args.query_set_path else config.paths.query_set_path
    run_dir = retrieve_queries(
        config,
        Path(args.chunk_metadata_path),
        Path(args.faiss_index_path),
        query_path,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
