from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rag_pdf.chunking import count_tokens, chunk_text_by_tokens, get_encoder, split_text_for_segment_aware_chunking


DEFAULT_DOCS = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]


def _chunk_count_for_text(
    text: str,
    *,
    chunk_size_tokens: int,
    overlap_tokens: int,
    segment_aware: bool,
    enc,
) -> int:
    value = str(text or "").strip()
    if not value:
        return 0
    if segment_aware:
        segments = split_text_for_segment_aware_chunking(value)
    else:
        segments = [("segment_000", value)]
    count = 0
    for _, seg_text in segments:
        count += len(
            chunk_text_by_tokens(
                seg_text,
                chunk_tokens=chunk_size_tokens,
                overlap_tokens=overlap_tokens,
                enc=enc,
            )
        )
    return count


def _doc_params(doc_dir: Path) -> tuple[int, int, bool]:
    metrics_path = doc_dir / "metrics.json"
    if not metrics_path.exists():
        return 280, 90, True
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        params = metrics.get("params", {}) if isinstance(metrics, dict) else {}
        if not isinstance(params, dict):
            params = {}
    except Exception:
        params = {}
    return (
        int(params.get("chunk_size_tokens", 280) or 280),
        int(params.get("chunk_overlap_tokens", 90) or 90),
        bool(params.get("segment_aware_chunking", True)),
    )


def compare_doc(doc_dir: Path, doc_id: str) -> dict[str, object]:
    pages_path = doc_dir / "pages.parquet"
    if not pages_path.exists():
        raise FileNotFoundError(f"Missing pages.parquet for {doc_id}")
    pages_df = pd.read_parquet(pages_path, columns=["page", "clean_text"])
    chunk_size_tokens, overlap_tokens, segment_aware = _doc_params(doc_dir)
    enc = get_encoder()

    token_tiktoken_total = 0
    token_fallback_total = 0
    chunk_tiktoken_total = 0
    chunk_fallback_total = 0
    pages_with_token_delta = 0
    pages_with_chunk_delta = 0
    max_abs_token_delta = 0
    max_abs_chunk_delta = 0

    for _, row in pages_df.iterrows():
        text = str(row.get("clean_text") or "")
        if not text.strip():
            continue
        tokens_tiktoken = count_tokens(text, enc)
        tokens_fallback = count_tokens(text, None)
        chunks_tiktoken = _chunk_count_for_text(
            text,
            chunk_size_tokens=chunk_size_tokens,
            overlap_tokens=overlap_tokens,
            segment_aware=segment_aware,
            enc=enc,
        )
        chunks_fallback = _chunk_count_for_text(
            text,
            chunk_size_tokens=chunk_size_tokens,
            overlap_tokens=overlap_tokens,
            segment_aware=segment_aware,
            enc=None,
        )

        token_tiktoken_total += int(tokens_tiktoken)
        token_fallback_total += int(tokens_fallback)
        chunk_tiktoken_total += int(chunks_tiktoken)
        chunk_fallback_total += int(chunks_fallback)

        token_delta = int(tokens_fallback) - int(tokens_tiktoken)
        chunk_delta = int(chunks_fallback) - int(chunks_tiktoken)
        if token_delta != 0:
            pages_with_token_delta += 1
        if chunk_delta != 0:
            pages_with_chunk_delta += 1
        max_abs_token_delta = max(max_abs_token_delta, abs(token_delta))
        max_abs_chunk_delta = max(max_abs_chunk_delta, abs(chunk_delta))

    token_delta_total = token_fallback_total - token_tiktoken_total
    chunk_delta_total = chunk_fallback_total - chunk_tiktoken_total
    token_delta_pct = (
        (token_delta_total / token_tiktoken_total) * 100.0 if token_tiktoken_total else 0.0
    )
    chunk_delta_pct = (
        (chunk_delta_total / chunk_tiktoken_total) * 100.0 if chunk_tiktoken_total else 0.0
    )

    return {
        "doc_id": doc_id,
        "pages": int(len(pages_df)),
        "chunk_size_tokens": chunk_size_tokens,
        "chunk_overlap_tokens": overlap_tokens,
        "segment_aware": segment_aware,
        "page_tokens_tiktoken": int(token_tiktoken_total),
        "page_tokens_fallback": int(token_fallback_total),
        "page_tokens_delta": int(token_delta_total),
        "page_tokens_delta_pct": round(float(token_delta_pct), 2),
        "text_chunks_tiktoken": int(chunk_tiktoken_total),
        "text_chunks_fallback": int(chunk_fallback_total),
        "text_chunks_delta": int(chunk_delta_total),
        "text_chunks_delta_pct": round(float(chunk_delta_pct), 2),
        "pages_with_token_delta": int(pages_with_token_delta),
        "pages_with_chunk_delta": int(pages_with_chunk_delta),
        "max_abs_page_token_delta": int(max_abs_token_delta),
        "max_abs_page_chunk_delta": int(max_abs_chunk_delta),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare tiktoken vs fallback token/chunk counts.")
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data_processed"))
    parser.add_argument("--out-csv", default=str(REPO_ROOT / "results" / "tiktoken_vs_fallback_2020_2025.csv"))
    parser.add_argument("--docs", nargs="*", default=DEFAULT_DOCS)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    rows: list[dict[str, object]] = []
    for doc_id in args.docs:
        rows.append(compare_doc(data_root / doc_id, doc_id))

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(out_df.to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
