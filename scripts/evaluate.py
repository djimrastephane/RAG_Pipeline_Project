from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from thesis_rag.config import load_config
from thesis_rag.pipeline import evaluate_retrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate page-level retrieval outputs.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--query-set-path", default="", help="Optional override path to query set JSON.")
    parser.add_argument("--dense-hits-path", required=True, help="Path to dense_page_hits.jsonl.")
    parser.add_argument("--sparse-hits-path", required=True, help="Path to bm25_page_hits.jsonl.")
    parser.add_argument("--hybrid-hits-path", required=True, help="Path to hybrid_page_hits.jsonl.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    query_path = Path(args.query_set_path) if args.query_set_path else config.paths.query_set_path
    run_dir = evaluate_retrieval(
        config,
        query_path,
        Path(args.dense_hits_path),
        Path(args.sparse_hits_path),
        Path(args.hybrid_hits_path),
    )
    print(run_dir)


if __name__ == "__main__":
    main()
