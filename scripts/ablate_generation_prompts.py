from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

import _matplotlib_env
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
scripts_path = repo_root / "scripts"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if scripts_path.exists() and str(scripts_path) not in sys.path:
    sys.path.insert(0, str(scripts_path))

from generation.constrained_extraction import run_constrained_extraction
from rag_pdf.services.search_service import SearchService
from retrieval_eval import score_answer_correctness


DEFAULT_DOCS = (
    "Grampian-2020-2021,"
    "Grampian-2021-2022,"
    "Grampian-2022-2023,"
    "Grampian-2023-2024,"
    "Grampian-2024-2025"
)

SUPPORTED_ARMS = (
    "baseline",
    "grounded_reasoning",
    "quote_then_answer",
    "constrained_extraction",
)

ARM_DISPLAY_ORDER = [
    "baseline",
    "grounded_reasoning",
    "quote_then_answer",
    "constrained_extraction",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ablate prompt variants under fixed retrieval and grounded-answer constraints."
    )
    p.add_argument("--data-root", default="data_processed")
    p.add_argument("--docs", default=DEFAULT_DOCS)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--queries-per-doc", type=int, default=3)
    p.add_argument("--out-dir", default="results/generation_prompt_ablation_2026-03-13")
    p.add_argument("--model-path", default="models/all-MiniLM-L6-v2")
    p.add_argument("--gen-timeout-seconds", type=float, default=20.0)
    p.add_argument("--max-context-chunks", type=int, default=5)
    p.add_argument("--max-context-chars", type=int, default=9000)
    p.add_argument("--max-chunk-chars", type=int, default=2200)
    p.add_argument(
        "--arms",
        default="baseline,grounded_reasoning,quote_then_answer,constrained_extraction",
        help=f"Comma-separated prompt arms. Supported: {', '.join(SUPPORTED_ARMS)}",
    )
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
    by_diff: dict[str, list[dict[str, Any]]] = {"LEX": [], "MOD": [], "STR": [], "OTHER": []}
    for q in queries:
        d = str(q.get("difficulty", "OTHER")).upper()
        if d not in by_diff:
            d = "OTHER"
        by_diff[d].append(q)

    picked: list[dict[str, Any]] = []
    for d in ("LEX", "MOD", "STR"):
        if len(picked) >= n_total:
            break
        if by_diff[d]:
            picked.append(rng.choice(by_diff[d]))

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


def _parse_arms(raw: str) -> list[str]:
    arms = [a.strip() for a in str(raw).split(",") if a.strip()]
    bad = [a for a in arms if a not in SUPPORTED_ARMS]
    if bad:
        raise ValueError(f"Unsupported arms: {bad}. Supported: {list(SUPPORTED_ARMS)}")
    if not arms:
        raise ValueError("No ablation arms selected.")
    return arms


def _build_context_text(results: list[dict[str, Any]], max_context_chunks: int, max_context_chars: int, max_chunk_chars: int) -> tuple[str, dict[str, Any]]:
    blocks: list[str] = []
    total_chars = 0
    used_chunks = 0
    truncated_chunks = 0
    context_truncated = False
    for r in results:
        if used_chunks >= int(max_context_chunks):
            context_truncated = True
            break
        chunk_id = str(r.get("chunk_id") or "").strip()
        pages = [int(x) for x in (r.get("pages") or []) if str(x).strip().isdigit()]
        page_label = ",".join(str(p) for p in pages) if pages else "NA"
        text = str(r.get("chunk_text") or "").strip()
        if not text:
            continue
        if len(text) > int(max_chunk_chars):
            text = text[: int(max_chunk_chars)].rstrip() + " ..."
            truncated_chunks += 1
        candidate_block = f"[chunk_id={chunk_id} pages={page_label}]\n{text}"
        if total_chars + len(candidate_block) > int(max_context_chars):
            context_truncated = True
            break
        blocks.append(candidate_block)
        total_chars += len(candidate_block)
        used_chunks += 1
    context = "\n\n".join(blocks).strip() or "[no context]"
    return context, {
        "context_chunks_used": int(used_chunks),
        "context_chars_used": int(total_chars),
        "context_chunk_char_limit": int(max_chunk_chars),
        "context_max_chunks": int(max_context_chunks),
        "context_max_chars": int(max_context_chars),
        "context_truncated": bool(context_truncated),
        "chunk_text_truncations": int(truncated_chunks),
    }


def _build_reasoning_prompt(question: str, context: str) -> str:
    return (
        "You are a retrieval-grounded assistant.\n"
        "Use only the provided CONTEXT.\n"
        "Reason from the evidence in CONTEXT before answering, but do not reveal your reasoning.\n"
        "If the answer is not explicitly supported, reply exactly: "
        "\"Insufficient evidence in retrieved context.\"\n"
        "Keep answer concise and factual.\n"
        "Do not invent chunk_id or page values.\n"
        "Return JSON only (no markdown/code fences) with this exact shape:\n"
        "{\"answer\":\"...\",\"citations\":[{\"chunk_id\":\"...\",\"page\":21}]}\n"
        "When answer is unsupported, return:\n"
        "{\"answer\":\"Insufficient evidence in retrieved context.\",\"citations\":[]}\n\n"
        f"QUESTION:\n{str(question).strip()}\n\n"
        f"CONTEXT:\n{context}\n\n"
        "INTERNAL TASK:\n"
        "1. Identify the smallest set of context lines that directly answer the question.\n"
        "2. Check that the answer is explicitly stated, not inferred.\n"
        "3. Return the answer and only the supporting citations.\n\n"
        "ANSWER:"
    )


def _build_quote_then_answer_prompt(question: str, context: str) -> str:
    return (
        "You are a retrieval-grounded assistant.\n"
        "Use only the provided CONTEXT.\n"
        "First locate the direct supporting evidence in CONTEXT, then answer from that evidence.\n"
        "Do not reveal hidden reasoning.\n"
        "If the answer is not explicitly supported, reply exactly: "
        "\"Insufficient evidence in retrieved context.\"\n"
        "Do not invent chunk_id or page values.\n"
        "Return JSON only (no markdown/code fences) with this exact shape:\n"
        "{\"answer\":\"...\",\"evidence_quote\":\"...\",\"citations\":[{\"chunk_id\":\"...\",\"page\":21}]}\n"
        "Rules:\n"
        "1. evidence_quote should be a short verbatim quote from CONTEXT supporting the answer.\n"
        "2. answer must stay concise and factual.\n"
        "3. When unsupported, return:\n"
        "{\"answer\":\"Insufficient evidence in retrieved context.\",\"evidence_quote\":\"\",\"citations\":[]}\n\n"
        f"QUESTION:\n{str(question).strip()}\n\n"
        f"CONTEXT:\n{context}\n\n"
        "ANSWER:"
    )


def _pct(arr: list[float], p: float) -> float:
    if not arr:
        return float("nan")
    return float(np.percentile(np.asarray(arr, dtype=np.float64), p))


def _parse_json_object(text: str) -> Optional[dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
    return None


def _write_charts(summary_df: pd.DataFrame, detail_df: pd.DataFrame, out_dir: Path) -> list[str]:
    chart_dir = out_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[str] = []
    n_queries = int(detail_df["query_id"].nunique()) if "query_id" in detail_df.columns and len(detail_df) else 0
    caption = f"Evaluation performed on n = {n_queries} queries."
    present_order = [arm for arm in ARM_DISPLAY_ORDER if arm in set(summary_df.get("arm", pd.Series(dtype=str)).astype(str))]

    if not summary_df.empty:
        plot_df = summary_df.copy()
        if present_order:
            plot_df["arm"] = pd.Categorical(plot_df["arm"], categories=present_order, ordered=True)
            plot_df = plot_df.sort_values("arm").reset_index(drop=True)
        plot_df["arm_label"] = plot_df["arm"].astype(str).str.replace("_", "\n", regex=False)

        # Chart 1: key quality metrics per arm.
        metrics = [
            ("answer_accuracy", "Answer accuracy"),
            ("generation_ok_rate", "Generation ok"),
            ("citation_valid_rate", "Valid citation"),
        ]
        x = np.arange(len(plot_df))
        width = 0.22
        fig, ax = plt.subplots(figsize=(9.2, 5.0), constrained_layout=True)
        palette = ["#33658a", "#55a630", "#b56576"]
        for i, (col, label) in enumerate(metrics):
            vals = plot_df[col].astype(float).to_numpy()
            ax.bar(x + (i - 1) * width, vals, width=width, label=label, color=palette[i])
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["arm_label"].tolist())
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Rate")
        ax.set_title(f"Prompt Ablation: Quality Metrics by Arm\n{caption}")
        ax.legend(frameon=False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        p1 = chart_dir / "prompt_ablation_quality_metrics.png"
        fig.savefig(p1, dpi=180)
        plt.close(fig)
        out_paths.append(str(p1))

        # Chart 2: latency vs accuracy tradeoff.
        fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
        colors = ["#1d3557", "#457b9d", "#2a9d8f", "#e76f51", "#8d99ae", "#bc6c25"]
        for i, (_, row) in enumerate(plot_df.iterrows()):
            ax.scatter(
                float(row["latency_mean_ms"]),
                float(row["answer_accuracy"]),
                s=120,
                color=colors[i % len(colors)],
                alpha=0.9,
            )
            ax.text(
                float(row["latency_mean_ms"]) + 20.0,
                float(row["answer_accuracy"]) + 0.01,
                str(row["arm"]),
                fontsize=9,
            )
        ax.set_xlabel("Mean latency (ms)")
        ax.set_ylabel("Answer accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Prompt Ablation: Accuracy vs Latency\n{caption}")
        ax.grid(linestyle="--", alpha=0.3)
        p2 = chart_dir / "prompt_ablation_accuracy_vs_latency.png"
        fig.savefig(p2, dpi=180)
        plt.close(fig)
        out_paths.append(str(p2))

    if not detail_df.empty:
        # Chart 3: status mix by arm.
        status_df = (
            detail_df.groupby(["arm", "generation_status"], dropna=False)
            .size()
            .rename("count")
            .reset_index()
        )
        pivot = status_df.pivot(index="arm", columns="generation_status", values="count").fillna(0.0)
        ordered_index = [arm for arm in ARM_DISPLAY_ORDER if arm in pivot.index]
        if ordered_index:
            pivot = pivot.reindex(ordered_index)
        row_sums = pivot.sum(axis=1).replace(0, 1.0)
        pct_df = pivot.div(row_sums, axis=0)
        fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
        bottom = np.zeros(len(pct_df.index))
        status_colors = {
            "ok": "#1b7f3a",
            "insufficient_evidence": "#f4a261",
            "guard_violation": "#e76f51",
            "error": "#c1121f",
            "unavailable": "#6c757d",
            "empty_response": "#8d99ae",
        }
        x = np.arange(len(pct_df.index))
        status_order = [
            "ok",
            "insufficient_evidence",
            "guard_violation",
            "error",
            "unavailable",
            "empty_response",
        ]
        ordered_cols = [col for col in status_order if col in pct_df.columns] + [
            col for col in pct_df.columns if col not in status_order
        ]
        for col in ordered_cols:
            vals = pct_df[col].to_numpy(dtype=float)
            ax.bar(
                x,
                vals,
                bottom=bottom,
                label=str(col),
                color=status_colors.get(str(col), None),
                edgecolor="#ffffff",
                linewidth=0.6,
            )
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in pct_df.index], rotation=15, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Share of queries")
        ax.set_title(f"Prompt Ablation: Generation Status Mix\n{caption}")
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8)
        p3 = chart_dir / "prompt_ablation_status_mix.png"
        fig.savefig(p3, dpi=180)
        plt.close(fig)
        out_paths.append(str(p3))

    return out_paths


def _run_prompt_arm_detailed(
    svc: SearchService,
    arm: str,
    question: str,
    results: list[dict[str, Any]],
    max_context_chunks: int,
    max_context_chars: int,
    max_chunk_chars: int,
    timeout_seconds: float,
) -> tuple[Optional[str], dict[str, Any]]:
    context, ctx_stats = _build_context_text(
        results=results,
        max_context_chunks=max_context_chunks,
        max_context_chars=max_context_chars,
        max_chunk_chars=max_chunk_chars,
    )

    if arm == "baseline":
        prompt, _ = svc._build_local_generation_prompt(
            question=question,
            results=results,
            max_context_chunks=max_context_chunks,
            max_context_chars=max_context_chars,
            max_chunk_chars=max_chunk_chars,
        )
    elif arm == "grounded_reasoning":
        prompt = _build_reasoning_prompt(question=question, context=context)
    elif arm == "quote_then_answer":
        prompt = _build_quote_then_answer_prompt(question=question, context=context)
    elif arm == "constrained_extraction":
        t0 = time.perf_counter()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        try:
            out = run_constrained_extraction(
                llm_client=svc.local_llm,
                question=question,
                context=context,
                temperature=0.0,
                top_p=1.0,
                max_tokens=200,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            answer = str(out.get("answer") or "").strip() or None
            violations = list(out.get("violations") or [])
            status = "ok" if not violations and answer else ("insufficient_evidence" if not answer else "guard_violation")
            return {
                "answer": answer,
                "evidence_quote": str(out.get("evidence_span") or "").strip() or None,
                "raw_output": None,
                "parsed_payload": None,
                "context_text": context,
                "debug": {
                "provider": "local_ollama",
                "status": status,
                "model": getattr(svc.local_llm, "model", ""),
                "error": None if not violations else "|".join(violations),
                "prompt_chars": 0,
                "latency_ms": float(round(latency_ms, 3)),
                "timeout_seconds": float(timeout_seconds),
                "parse_mode": "constrained_extraction",
                "citations_parsed": 0,
                "citations_valid": 0,
                "citations_rejected": 0,
                "quote_present": bool(out.get("evidence_span")),
                **ctx_stats,
                },
            }
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "answer": None,
                "evidence_quote": None,
                "raw_output": None,
                "parsed_payload": None,
                "context_text": context,
                "debug": {
                "provider": "local_ollama",
                "status": "error",
                "model": getattr(svc.local_llm, "model", ""),
                "error": f"{type(exc).__name__}: {exc}",
                "prompt_chars": 0,
                "latency_ms": float(round(latency_ms, 3)),
                "timeout_seconds": float(timeout_seconds),
                "parse_mode": "constrained_extraction_error",
                "citations_parsed": 0,
                "citations_valid": 0,
                "citations_rejected": 0,
                "quote_present": False,
                **ctx_stats,
                },
            }
    else:
        raise ValueError(f"Unsupported arm: {arm}")

    t0 = time.perf_counter()
    out = svc.local_llm.generate(prompt, timeout_seconds=float(timeout_seconds))
    latency_ms = (time.perf_counter() - t0) * 1000.0
    raw_generated_answer = str(out.answer or "")
    parsed_answer, raw_citations_from_json, parse_mode = svc._parse_generation_json_payload(raw_generated_answer)
    raw_citations = list(raw_citations_from_json)
    if not raw_citations:
        raw_citations = svc._extract_citations_from_answer(raw_generated_answer)
    valid_citations, rejected_citations = svc._validate_citations(raw_citations, results)
    text_answer = str(parsed_answer or "").strip()
    raw_status = str(out.status or "").strip().lower()

    if raw_status != "ok" or not text_answer:
        final_answer = None
        status = raw_status or "error"
    elif text_answer.lower() == "insufficient evidence in retrieved context.":
        final_answer = None
        status = "insufficient_evidence"
    elif arm != "constrained_extraction" and not valid_citations:
        final_answer = None
        status = "insufficient_evidence"
    else:
        final_answer = text_answer
        status = "ok"

    parsed_payload = _parse_json_object(raw_generated_answer)
    evidence_quote = None
    if isinstance(parsed_payload, dict):
        eq = parsed_payload.get("evidence_quote")
        evidence_quote = str(eq).strip() if eq is not None and str(eq).strip() else None

    return {
        "answer": final_answer,
        "evidence_quote": evidence_quote,
        "raw_output": raw_generated_answer,
        "parsed_payload": parsed_payload,
        "context_text": context,
        "debug": {
        "provider": "local_ollama",
        "status": status,
        "model": out.model,
        "error": out.error,
        "prompt_chars": int(out.prompt_chars),
        "latency_ms": float(round(latency_ms, 3)),
        "timeout_seconds": float(timeout_seconds),
        "parse_mode": parse_mode,
        "citations_parsed": int(len(raw_citations)),
        "citations_valid": int(len(valid_citations)),
        "citations_rejected": int(rejected_citations),
        "quote_present": bool(evidence_quote),
        **ctx_stats,
        },
    }


def _run_prompt_arm(
    svc: SearchService,
    arm: str,
    question: str,
    results: list[dict[str, Any]],
    max_context_chunks: int,
    max_context_chars: int,
    max_chunk_chars: int,
    timeout_seconds: float,
) -> tuple[Optional[str], dict[str, Any]]:
    out = _run_prompt_arm_detailed(
        svc=svc,
        arm=arm,
        question=question,
        results=results,
        max_context_chunks=max_context_chunks,
        max_context_chars=max_context_chars,
        max_chunk_chars=max_chunk_chars,
        timeout_seconds=timeout_seconds,
    )
    return out.get("answer"), dict(out.get("debug") or {})


def main() -> None:
    args = parse_args()
    docs = [d.strip() for d in str(args.docs).split(",") if d.strip()]
    arms = _parse_arms(args.arms)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    svc = SearchService(repo_root=Path(".").resolve(), model_path=Path(args.model_path))
    svc.gen_timeout_seconds = float(args.gen_timeout_seconds)

    sampled_by_doc: dict[str, list[dict[str, Any]]] = {}
    for i, d in enumerate(docs):
        eval_path = data_root / d / "eval_set.json"
        queries = _load_queries(eval_path)
        sampled_by_doc[d] = _sample_queries(queries, n_total=int(args.queries_per_doc), seed=int(args.seed) + i)

    sample_rows: list[dict[str, Any]] = []
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
    total_queries = sum(len(v) for v in sampled_by_doc.values())
    total_calls = total_queries * len(arms)
    call_no = 0

    for arm in arms:
        arm_rows: list[dict[str, Any]] = []
        for d in docs:
            data_dir = data_root / d
            for q in sampled_by_doc[d]:
                call_no += 1
                retrieval_out = svc.search(
                    data_dir=data_dir,
                    question=str(q.get("question", "")),
                    k=int(args.k),
                    query_id=str(q.get("query_id", "")) or None,
                    include_generated_answer=False,
                )
                results = list(retrieval_out.get("results") or [])
                generated_answer, gdbg = _run_prompt_arm(
                    svc=svc,
                    arm=arm,
                    question=str(q.get("question", "")),
                    results=results,
                    max_context_chunks=int(args.max_context_chunks),
                    max_context_chars=int(args.max_context_chars),
                    max_chunk_chars=int(args.max_chunk_chars),
                    timeout_seconds=float(args.gen_timeout_seconds),
                )
                expected_answer = q.get("expected_answer")
                answer_type = str(q.get("answer_type", "unknown"))
                is_correct, answer_status = score_answer_correctness(
                    expected_answer=expected_answer,
                    answer_type=answer_type,
                    extracted_answer=(str(generated_answer) if generated_answer is not None else None),
                )
                row = {
                    "arm": arm,
                    "doc_id": d,
                    "query_id": q.get("query_id"),
                    "difficulty": q.get("difficulty"),
                    "answer_type": answer_type,
                    "generation_status": gdbg.get("status"),
                    "generation_model": gdbg.get("model"),
                    "latency_ms": float(gdbg.get("latency_ms", 0.0) or 0.0),
                    "prompt_chars": int(gdbg.get("prompt_chars", 0) or 0),
                    "context_chunks_used": int(gdbg.get("context_chunks_used", 0) or 0),
                    "context_chars_used": int(gdbg.get("context_chars_used", 0) or 0),
                    "context_truncated": bool(gdbg.get("context_truncated", False)),
                    "citations_parsed": int(gdbg.get("citations_parsed", 0) or 0),
                    "citations_valid": int(gdbg.get("citations_valid", 0) or 0),
                    "citations_rejected": int(gdbg.get("citations_rejected", 0) or 0),
                    "parse_mode": gdbg.get("parse_mode"),
                    "quote_present": bool(gdbg.get("quote_present", False)),
                    "answer_correct": (None if is_correct is None else bool(is_correct)),
                    "answer_status": answer_status,
                    "generated_answer_empty": (generated_answer is None or str(generated_answer).strip() == ""),
                }
                arm_rows.append(row)
                detail_rows.append(row)
        cdf = pd.DataFrame(arm_rows)
        scored = cdf[cdf["answer_correct"].notna()]
        lat = cdf["latency_ms"].tolist()
        summary_rows.append(
            {
                "arm": arm,
                "n_queries": int(len(cdf)),
                "answer_accuracy": float(scored["answer_correct"].mean()) if len(scored) else float("nan"),
                "generation_ok_rate": float((cdf["generation_status"].astype(str) == "ok").mean()) if len(cdf) else float("nan"),
                "insufficient_evidence_rate": float((cdf["generation_status"].astype(str) == "insufficient_evidence").mean()) if len(cdf) else float("nan"),
                "context_truncated_rate": float(cdf["context_truncated"].mean()) if len(cdf) else float("nan"),
                "citation_valid_rate": float((cdf["citations_valid"] > 0).mean()) if len(cdf) else float("nan"),
                "json_parse_success_rate": float(cdf["parse_mode"].astype(str).ne("empty").mean()) if len(cdf) else float("nan"),
                "latency_mean_ms": float(np.mean(lat)) if lat else float("nan"),
                "latency_p50_ms": _pct(lat, 50),
                "latency_p95_ms": _pct(lat, 95),
                "prompt_chars_mean": float(cdf["prompt_chars"].mean()) if len(cdf) else float("nan"),
            }
        )
        print(f"[{arm}] done {len(cdf)} queries ({call_no}/{total_calls})")

    detail_df = pd.DataFrame(detail_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["answer_accuracy", "citation_valid_rate", "latency_p50_ms"],
        ascending=[False, False, True],
    )
    if not summary_df.empty:
        ordered_summary = [arm for arm in ARM_DISPLAY_ORDER if arm in set(summary_df["arm"].astype(str))]
        summary_df["arm"] = pd.Categorical(summary_df["arm"], categories=ordered_summary, ordered=True)
        summary_df = summary_df.sort_values("arm").reset_index(drop=True)
    if not detail_df.empty:
        ordered_detail = [arm for arm in ARM_DISPLAY_ORDER if arm in set(detail_df["arm"].astype(str))]
        detail_df["arm"] = pd.Categorical(detail_df["arm"], categories=ordered_detail, ordered=True)
        detail_df = detail_df.sort_values(["arm", "doc_id", "query_id"]).reset_index(drop=True)

    detail_df.to_csv(out_dir / "generation_prompt_ablation_detail.csv", index=False)
    summary_df.to_csv(out_dir / "generation_prompt_ablation_summary.csv", index=False)
    chart_paths = _write_charts(summary_df=summary_df, detail_df=detail_df, out_dir=out_dir)

    best = summary_df.iloc[0].to_dict() if len(summary_df) else {}
    report = {
        "docs": docs,
        "queries_per_doc": int(args.queries_per_doc),
        "k": int(args.k),
        "seed": int(args.seed),
        "arms": arms,
        "total_evals": int(len(detail_df)),
        "best_arm": best,
        "chart_paths": chart_paths,
    }
    (out_dir / "generation_prompt_ablation_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    status_by_arm = (
        detail_df.groupby(["arm", "generation_status"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["arm", "count"], ascending=[True, False])
    )
    if len(status_by_arm):
        status_by_arm["pct"] = status_by_arm.groupby("arm")["count"].transform(lambda s: s / max(1, s.sum()))

    difficulty_summary = (
        detail_df.groupby(["arm", "difficulty"], dropna=False)
        .agg(
            n=("query_id", "count"),
            answer_accuracy=("answer_correct", "mean"),
            ok_rate=("generation_status", lambda s: float((s.astype(str) == "ok").mean())),
        )
        .reset_index()
        .sort_values(["arm", "difficulty"])
    )

    md = [
        "# Generation Prompt Ablation",
        "",
        "- Design: fixed retrieval, grounded-answer prompt variants only.",
        f"- docs: `{len(docs)}`",
        f"- queries_per_doc: `{args.queries_per_doc}`",
        f"- k: `{args.k}`",
        f"- arms: `{', '.join(arms)}`",
        f"- total evaluations: `{len(detail_df)}`",
        f"- Evaluation performed on `n = {int(detail_df['query_id'].nunique()) if len(detail_df) else 0}` queries.",
        "",
        "## Recommendation",
        "",
        (
            f"- Current top arm by ranking rule: `{best.get('arm', '')}` "
            f"(answer_accuracy={float(best.get('answer_accuracy', float('nan'))):.3f}, "
            f"citation_valid_rate={float(best.get('citation_valid_rate', float('nan'))):.3f}, "
            f"latency_p50_ms={float(best.get('latency_p50_ms', float('nan'))):.1f})."
            if best
            else "- No best arm available."
        ),
        "",
        "## Ranked arms",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## Generation Status By Arm",
        "",
        (status_by_arm.to_markdown(index=False) if len(status_by_arm) else "_No rows_"),
        "",
        "## Difficulty Breakdown",
        "",
        (difficulty_summary.to_markdown(index=False) if len(difficulty_summary) else "_No rows_"),
        "",
        "## Charts",
        "",
        f"- Evaluation performed on n = {int(detail_df['query_id'].nunique()) if len(detail_df) else 0} queries.",
        "",
        *([f"- `{p}`" for p in chart_paths] if chart_paths else ["- No charts generated."]),
    ]
    (out_dir / "generation_prompt_ablation_summary.md").write_text("\n".join(md), encoding="utf-8")

    print("Wrote:", out_dir / "generation_prompt_ablation_detail.csv")
    print("Wrote:", out_dir / "generation_prompt_ablation_summary.csv")
    print("Wrote:", out_dir / "generation_prompt_ablation_summary.md")
    print("Wrote:", out_dir / "generation_prompt_ablation_report.json")
    for chart_path in chart_paths:
        print("Wrote:", chart_path)


if __name__ == "__main__":
    main()
