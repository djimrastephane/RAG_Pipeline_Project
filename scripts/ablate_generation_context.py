from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rag_pdf.services.search_service import SearchService
from scripts.retrieval_eval import score_answer_correctness


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablate generation context parameters (chunks/chars) with local LLM.")
    p.add_argument("--data-root", default="data_processed")
    p.add_argument(
        "--docs",
        default="Grampian-2020-2021,Grampian-2021-2022,Grampian-2022-2023,Grampian-2023-2024,Grampian-2024-2025",
    )
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--queries-per-doc", type=int, default=3, help="Total sampled queries per doc.")
    p.add_argument("--out-dir", default="results/generation_context_ablation_2026-03-02")
    p.add_argument("--model-path", default="models/all-MiniLM-L6-v2")
    p.add_argument("--gen-timeout-seconds", type=float, default=20.0)
    return p.parse_args()


def _load_queries(eval_path: Path) -> list[dict[str, Any]]:
    obj = json.loads(eval_path.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        q = obj.get("queries")
        if isinstance(q, list):
            return [x for x in q if isinstance(x, dict)]
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    return []


def _sample_queries(queries: list[dict[str, Any]], n_total: int, seed: int) -> list[dict[str, Any]]:
    if not queries or n_total <= 0:
        return []
    rng = random.Random(seed)
    # Try stratified by difficulty first.
    by_diff: dict[str, list[dict[str, Any]]] = {"LEX": [], "MOD": [], "STR": [], "OTHER": []}
    for q in queries:
        d = str(q.get("difficulty", "OTHER")).upper()
        if d not in by_diff:
            d = "OTHER"
        by_diff[d].append(q)

    picked: list[dict[str, Any]] = []
    order = ["LEX", "MOD", "STR"]
    # First pass: one per major bucket if available.
    for d in order:
        if len(picked) >= n_total:
            break
        if by_diff[d]:
            picked.append(rng.choice(by_diff[d]))

    # Fill the rest from remaining pool without duplicate query_id when possible.
    used = {str(x.get("query_id", "")) for x in picked}
    pool = queries[:]
    rng.shuffle(pool)
    for q in pool:
        if len(picked) >= n_total:
            break
        qid = str(q.get("query_id", ""))
        if qid and qid in used:
            continue
        picked.append(q)
        if qid:
            used.add(qid)

    return picked[:n_total]


def _pct(arr: list[float], p: float) -> float:
    if not arr:
        return float("nan")
    return float(np.percentile(np.asarray(arr, dtype=np.float64), p))


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    docs = [d.strip() for d in args.docs.split(",") if d.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3x3 grid requested
    chunk_grid = [3, 5, 8]
    char_grid = [6000, 9000, 12000]
    configs = [(c, ch) for c in chunk_grid for ch in char_grid]

    svc = SearchService(repo_root=Path(".").resolve(), model_path=Path(args.model_path))
    svc.gen_timeout_seconds = float(args.gen_timeout_seconds)

    sampled_by_doc: dict[str, list[dict[str, Any]]] = {}
    for i, d in enumerate(docs):
        eval_path = data_root / d / "eval_set.json"
        q = _load_queries(eval_path)
        sampled_by_doc[d] = _sample_queries(q, n_total=int(args.queries_per_doc), seed=int(args.seed) + i)

    sample_rows = []
    for d, qs in sampled_by_doc.items():
        for q in qs:
            sample_rows.append(
                {
                    "doc_id": d,
                    "query_id": q.get("query_id"),
                    "difficulty": q.get("difficulty"),
                    "question": q.get("question"),
                    "answer_type": q.get("answer_type"),
                    "expected_answer": q.get("expected_answer"),
                }
            )
    pd.DataFrame(sample_rows).to_csv(out_dir / "sampled_queries.csv", index=False)

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    total_calls = len(configs) * sum(len(v) for v in sampled_by_doc.values())
    call_no = 0

    for max_chunks, max_chars in configs:
        svc.gen_max_context_chunks = int(max_chunks)
        svc.gen_max_context_chars = int(max_chars)

        cfg_rows: list[dict[str, Any]] = []
        for d in docs:
            data_dir = data_root / d
            for q in sampled_by_doc[d]:
                call_no += 1
                out = svc.search(
                    data_dir=data_dir,
                    question=str(q.get("question", "")),
                    k=int(args.k),
                    query_id=str(q.get("query_id", "")) or None,
                    include_generated_answer=True,
                )
                gen_answer = out.get("generated_answer")
                expected_answer = q.get("expected_answer")
                answer_type = str(q.get("answer_type", "unknown"))
                is_correct, answer_status = score_answer_correctness(
                    expected_answer=expected_answer,
                    answer_type=answer_type,
                    extracted_answer=(str(gen_answer) if gen_answer is not None else None),
                )
                gdbg = out.get("generation_debug") or {}
                row = {
                    "max_context_chunks": int(max_chunks),
                    "max_context_chars": int(max_chars),
                    "doc_id": d,
                    "query_id": q.get("query_id"),
                    "difficulty": q.get("difficulty"),
                    "answer_type": answer_type,
                    "generation_status": out.get("generation_status"),
                    "generation_model": gdbg.get("model"),
                    "latency_ms": float(gdbg.get("latency_ms", 0.0) or 0.0),
                    "context_chunks_used": int(gdbg.get("context_chunks_used", 0) or 0),
                    "context_chars_used": int(gdbg.get("context_chars_used", 0) or 0),
                    "context_truncated": bool(gdbg.get("context_truncated", False)),
                    "citations_valid": int(gdbg.get("citations_valid", 0) or 0),
                    "answer_correct": (None if is_correct is None else bool(is_correct)),
                    "answer_status": answer_status,
                    "generated_answer_empty": (gen_answer is None or str(gen_answer).strip() == ""),
                }
                cfg_rows.append(row)
                detail_rows.append(row)

        cdf = pd.DataFrame(cfg_rows)
        scored = cdf[cdf["answer_correct"].notna()]
        acc = float(scored["answer_correct"].mean()) if len(scored) else float("nan")
        ok_rate = float((cdf["generation_status"].astype(str) == "ok").mean()) if len(cdf) else float("nan")
        insuff_rate = float((cdf["generation_status"].astype(str) == "insufficient_evidence").mean()) if len(cdf) else float("nan")
        trunc_rate = float(cdf["context_truncated"].mean()) if len(cdf) else float("nan")
        lat = cdf["latency_ms"].tolist()

        summary_rows.append(
            {
                "max_context_chunks": int(max_chunks),
                "max_context_chars": int(max_chars),
                "n_queries": int(len(cdf)),
                "answer_accuracy": acc,
                "generation_ok_rate": ok_rate,
                "insufficient_evidence_rate": insuff_rate,
                "context_truncated_rate": trunc_rate,
                "latency_mean_ms": float(np.mean(lat)) if lat else float("nan"),
                "latency_p50_ms": _pct(lat, 50),
                "latency_p95_ms": _pct(lat, 95),
                "context_chunks_used_mean": float(cdf["context_chunks_used"].mean()) if len(cdf) else float("nan"),
                "context_chars_used_mean": float(cdf["context_chars_used"].mean()) if len(cdf) else float("nan"),
            }
        )
        print(f"[{max_chunks} chunks, {max_chars} chars] done {len(cdf)} queries ({call_no}/{total_calls})")

    detail_df = pd.DataFrame(detail_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values(["answer_accuracy", "latency_p50_ms"], ascending=[False, True])

    detail_df.to_csv(out_dir / "generation_context_ablation_detail.csv", index=False)
    summary_df.to_csv(out_dir / "generation_context_ablation_summary.csv", index=False)

    best = summary_df.iloc[0].to_dict() if len(summary_df) else {}
    report = {
        "docs": docs,
        "queries_per_doc": int(args.queries_per_doc),
        "k": int(args.k),
        "seed": int(args.seed),
        "total_evals": int(len(detail_df)),
        "best_config": best,
    }
    (out_dir / "generation_context_ablation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# Generation Context Ablation",
        "",
        f"- docs: `{len(docs)}`",
        f"- queries_per_doc: `{args.queries_per_doc}`",
        f"- k: `{args.k}`",
        f"- total evaluations: `{len(detail_df)}`",
        "",
        "## Ranked configs",
        "",
        summary_df.to_markdown(index=False),
    ]
    (out_dir / "generation_context_ablation_summary.md").write_text("\n".join(md), encoding="utf-8")

    print("Wrote:", out_dir / "generation_context_ablation_detail.csv")
    print("Wrote:", out_dir / "generation_context_ablation_summary.csv")
    print("Wrote:", out_dir / "generation_context_ablation_summary.md")
    print("Wrote:", out_dir / "generation_context_ablation_report.json")


if __name__ == "__main__":
    main()
