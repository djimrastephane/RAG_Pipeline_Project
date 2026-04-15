from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from rag_pdf.services.search_service import SearchService


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export RAGAS-ready JSONL from eval_set queries.")
    p.add_argument("--data-root", default="data_processed", help="Root directory containing per-doc folders.")
    p.add_argument(
        "--docs",
        default="Grampian-2020-2021,Grampian-2021-2022,Grampian-2022-2023,Grampian-2023-2024,Grampian-2024-2025",
        help="Comma-separated doc folder names.",
    )
    p.add_argument("--model-path", default="models/all-MiniLM-L6-v2", help="Sentence-transformer model path.")
    p.add_argument("--k", type=int, default=5, help="Retrieved contexts per query.")
    p.add_argument("--out-jsonl", default="results/ragas/ragas_input_250q.jsonl", help="Output JSONL path.")
    p.add_argument("--out-csv", default="results/ragas/ragas_input_250q.csv", help="Output CSV path.")
    p.add_argument(
        "--include-generated-answer",
        action="store_true",
        help="Use LLM generated answer. If false, uses deterministic predicted_answer fallback.",
    )
    p.add_argument(
        "--gen-timeout-seconds",
        type=float,
        default=20.0,
        help="Generation timeout when --include-generated-answer is set.",
    )
    p.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Optional cap for quick tests (0 = all queries).",
    )
    p.add_argument(
        "--queries-per-doc",
        type=int,
        default=0,
        help="Optional per-document query cap (0 = all queries in each doc).",
    )
    return p.parse_args()


def _load_queries(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        q = obj.get("queries")
        if isinstance(q, list):
            return [x for x in q if isinstance(x, dict)]
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    return []


def _contexts_from_results(results: list[dict[str, Any]]) -> list[str]:
    contexts: list[str] = []
    for r in results:
        txt = str(r.get("chunk_text") or "").strip()
        if not txt:
            txt = str(r.get("snippet") or "").strip()
        if txt:
            contexts.append(txt)
    return contexts


def main() -> None:
    args = parse_args()
    repo_root = Path(".").resolve()
    data_root = Path(args.data_root)
    docs = [d.strip() for d in str(args.docs).split(",") if d.strip()]

    out_jsonl = Path(args.out_jsonl)
    out_csv = Path(args.out_csv)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    svc = SearchService(repo_root=repo_root, model_path=Path(args.model_path))
    svc.gen_timeout_seconds = float(args.gen_timeout_seconds)

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    done = 0

    for doc_id in docs:
        data_dir = data_root / doc_id
        eval_path = data_dir / "eval_set.json"
        if not eval_path.exists():
            errors.append({"doc_id": doc_id, "query_id": None, "error": f"missing {eval_path}"})
            continue

        queries = _load_queries(eval_path)
        per_doc_done = 0
        for q in queries:
            if args.max_queries and done >= int(args.max_queries):
                break
            if args.queries_per_doc and per_doc_done >= int(args.queries_per_doc):
                break
            question = str(q.get("question") or "").strip()
            query_id = str(q.get("query_id") or "").strip()
            if not question:
                continue
            done += 1
            per_doc_done += 1
            try:
                out = svc.search(
                    data_dir=data_dir,
                    question=question,
                    k=int(args.k),
                    query_id=query_id or None,
                    include_generated_answer=bool(args.include_generated_answer),
                )
                results = out.get("results") if isinstance(out, dict) else []
                contexts = _contexts_from_results(results if isinstance(results, list) else [])
                generated_answer = out.get("generated_answer") if isinstance(out, dict) else None
                predicted_answer = out.get("predicted_answer") if isinstance(out, dict) else None
                answer = str(generated_answer or predicted_answer or "").strip()
                row = {
                    "question": question,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": str(q.get("expected_answer") or "").strip(),
                    "ground_truths": [str(q.get("expected_answer") or "").strip()],
                    "query_id": query_id,
                    "doc_id": doc_id,
                    "difficulty": q.get("difficulty"),
                    "answer_type": q.get("answer_type"),
                    "expected_pages": q.get("expected_pages") or [],
                    "generation_status": (out.get("generation_status") if isinstance(out, dict) else None),
                }
                rows.append(row)
            except Exception as exc:
                errors.append({"doc_id": doc_id, "query_id": query_id, "error": str(exc)})

        if args.max_queries and done >= int(args.max_queries):
            break

    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out_csv, index=False)
    else:
        pd.DataFrame(columns=["question", "answer", "contexts", "ground_truth"]).to_csv(out_csv, index=False)

    report = {
        "rows": len(rows),
        "errors": len(errors),
        "include_generated_answer": bool(args.include_generated_answer),
        "docs": docs,
        "k": int(args.k),
        "out_jsonl": str(out_jsonl),
        "out_csv": str(out_csv),
    }
    report_path = out_jsonl.parent / "ragas_export_report.json"
    report_path.write_text(json.dumps({"report": report, "errors": errors[:50]}, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    if errors:
        print(f"First error: {errors[0]}")


if __name__ == "__main__":
    main()
