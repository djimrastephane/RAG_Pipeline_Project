from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import _matplotlib_env
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

from generation.constrained_extraction import run_constrained_extraction
from rag_pdf.services.local_llm_service import LocalLLMService

RRF_K = 20
RRF_DENSE_WEIGHT = 0.5
RRF_BM25_WEIGHT = 2.0
MAX_K_SEARCH = 100


@dataclass
class DocArtifacts:
    meta: pd.DataFrame
    embeddings: np.ndarray
    chunk_text_by_id: dict[str, str]
    bm25: "BM25Index"
    eval_map: dict[str, dict[str, Any]]


class BM25Index:
    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.n_docs = len(docs_tokens)
        self.doc_len = [len(toks) for toks in docs_tokens]
        self.avgdl = float(sum(self.doc_len)) / max(1.0, float(self.n_docs))
        self.term_freqs: list[Counter[str]] = [Counter(toks) for toks in docs_tokens]
        self.doc_freq: Counter[str] = Counter()
        for tf in self.term_freqs:
            for term in tf.keys():
                self.doc_freq[term] += 1
        self.idf: dict[str, float] = {}
        for term, df in self.doc_freq.items():
            self.idf[term] = math.log(1.0 + ((self.n_docs - df + 0.5) / (df + 0.5)))

    def score_query(self, query_tokens: list[str]) -> list[float]:
        if self.n_docs == 0:
            return []
        q_terms = Counter(query_tokens)
        scores = [0.0] * self.n_docs
        for term in q_terms.keys():
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i, tf in enumerate(self.term_freqs):
                f = tf.get(term, 0)
                if f <= 0:
                    continue
                dl = self.doc_len[i]
                denom = f + self.k1 * (1.0 - self.b + self.b * (dl / max(self.avgdl, 1e-9)))
                scores[i] += idf * (f * (self.k1 + 1.0) / max(denom, 1e-12))
        return scores


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict evidence-constrained extraction eval.")
    p.add_argument(
        "--mode",
        default="constrained_extraction",
        choices=["legacy", "constrained_extraction"],
        help="Generation mode. constrained_extraction uses src/generation/constrained_extraction.py",
    )
    p.add_argument("--sample-csv", default="results/context_chunks_3_vs_5_stats_2026-03-02/sampled_50_queries.csv")
    p.add_argument("--data-root", default="data_processed")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--max-context-chunks", type=int, default=3)
    p.add_argument("--max-context-chars", type=int, default=6000)
    p.add_argument("--max-chunk-chars", type=int, default=2200)
    p.add_argument("--model-path", default="models/all-MiniLM-L6-v2")
    p.add_argument("--gen-timeout-seconds", type=float, default=30.0)
    p.add_argument("--gen-max-tokens", type=int, default=200)
    p.add_argument("--out-dir", default="results/strict_evidence_extraction_2026-03-02")
    return p.parse_args()


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9\-]{1,}", str(text or "").lower())


def rrf_fuse(
    dense_ranked: list[int],
    bm25_ranked: list[int],
    rrf_k: int = RRF_K,
    dense_weight: float = RRF_DENSE_WEIGHT,
    bm25_weight: float = RRF_BM25_WEIGHT,
) -> tuple[list[int], dict[int, float]]:
    scores: dict[int, float] = {}
    for rank, idx in enumerate(dense_ranked, start=1):
        scores[idx] = scores.get(idx, 0.0) + (dense_weight / float(rrf_k + rank))
    for rank, idx in enumerate(bm25_ranked, start=1):
        scores[idx] = scores.get(idx, 0.0) + (bm25_weight / float(rrf_k + rank))
    ranked = [idx for idx, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
    return ranked, scores


def _read_eval_items(eval_path: Path) -> dict[str, dict[str, Any]]:
    if not eval_path.exists():
        return {}
    raw = json.loads(eval_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        rows = raw.get("queries") if isinstance(raw.get("queries"), list) else []
    elif isinstance(raw, list):
        rows = raw
    else:
        rows = []
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        qid = str(r.get("query_id") or "").strip()
        if qid:
            out[qid] = r
    return out


def _to_pages_list(v: Any) -> list[int]:
    if v is None:
        return []
    if isinstance(v, list):
        out: list[int] = []
        for x in v:
            try:
                out.append(int(x))
            except Exception:
                if isinstance(x, dict) and "element" in x:
                    try:
                        out.append(int(x["element"]))
                    except Exception:
                        pass
        return out
    try:
        return [int(v)]
    except Exception:
        return []


def _load_doc_artifacts(data_dir: Path) -> DocArtifacts:
    embeddings = np.load(data_dir / "embeddings.npy").astype("float32")
    embeddings = l2_normalize(embeddings).astype("float32")
    meta = pd.read_parquet(data_dir / "chunk_meta.parquet").reset_index(drop=True)
    chunks = pd.read_parquet(data_dir / "chunks.parquet")

    chunk_text_by_id: dict[str, str] = {}
    if "chunk_id_global" in chunks.columns:
        for _, row in chunks.iterrows():
            cid = str(row.get("chunk_id_global") or "").strip()
            if cid:
                chunk_text_by_id[cid] = str(row.get("chunk_text") or "")
    if "chunk_id" in chunks.columns:
        for _, row in chunks.iterrows():
            cid = str(row.get("chunk_id") or "").strip()
            if cid and cid not in chunk_text_by_id:
                chunk_text_by_id[cid] = str(row.get("chunk_text") or "")

    corpus_texts: list[str] = []
    for _, row in meta.iterrows():
        cid = str(row.get("chunk_id_global") or row.get("chunk_id") or "").strip()
        corpus_texts.append(chunk_text_by_id.get(cid, ""))
    bm25 = BM25Index([tokenize(t) for t in corpus_texts], k1=1.5, b=0.75)

    eval_map = _read_eval_items(data_dir / "eval_set.json")
    return DocArtifacts(
        meta=meta,
        embeddings=embeddings,
        chunk_text_by_id=chunk_text_by_id,
        bm25=bm25,
        eval_map=eval_map,
    )


def _retrieve_hybrid(
    question: str,
    doc: DocArtifacts,
    k: int,
) -> tuple[list[dict[str, Any]], bool]:
    q_tokens = tokenize(question)
    q_emb = model.encode([question], convert_to_numpy=True, normalize_embeddings=False).astype("float32")
    q_emb = l2_normalize(q_emb).astype("float32")

    dense_scores = np.dot(doc.embeddings, q_emb[0])
    k_search = min(max(int(k), MAX_K_SEARCH), len(doc.meta))
    if k_search < len(dense_scores):
        top_idx = np.argpartition(-dense_scores, k_search - 1)[:k_search]
        dense_ranked = top_idx[np.argsort(-dense_scores[top_idx])].tolist()
    else:
        dense_ranked = np.argsort(-dense_scores).tolist()

    bm25_scores = doc.bm25.score_query(q_tokens)
    bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k_search]

    fused_ranked, fused_scores = rrf_fuse(dense_ranked=dense_ranked, bm25_ranked=bm25_ranked)
    idx_list = fused_ranked[: max(1, int(k))]

    results: list[dict[str, Any]] = []
    for rank, idx in enumerate(idx_list, start=1):
        row = doc.meta.iloc[int(idx)]
        cid = str(row.get("chunk_id_global") or row.get("chunk_id") or "").strip()
        pages = _to_pages_list(row.get("pages"))
        if not pages:
            ps = row.get("page_start")
            pe = row.get("page_end")
            try:
                if pd.notna(ps):
                    pages.append(int(ps))
            except Exception:
                pass
            try:
                if pd.notna(pe):
                    pe_i = int(pe)
                    if pe_i not in pages:
                        pages.append(pe_i)
            except Exception:
                pass
        results.append(
            {
                "rank": rank,
                "chunk_id": cid,
                "pages": pages,
                "chunk_text": doc.chunk_text_by_id.get(cid, ""),
                "rrf_score": float(fused_scores.get(int(idx), 0.0)),
                "dense_raw_score": float(dense_scores[int(idx)]),
                "bm25_raw_score": float(bm25_scores[int(idx)] if int(idx) < len(bm25_scores) else 0.0),
            }
        )

    return results, bool(idx_list)


def _build_prompt(question: str, results: list[dict[str, Any]], max_chunks: int, max_chars: int, max_chunk_chars: int) -> tuple[str, dict[str, Any]]:
    blocks: list[str] = []
    total_chars = 0
    used_chunks = 0
    context_truncated = False

    for r in results:
        if used_chunks >= max_chunks:
            context_truncated = True
            break
        chunk_id = str(r.get("chunk_id") or "")
        pages = [int(x) for x in (r.get("pages") or []) if str(x).isdigit()]
        page_label = ",".join(str(p) for p in pages) if pages else "NA"
        text = str(r.get("chunk_text") or "").strip()
        if not text:
            continue
        if len(text) > max_chunk_chars:
            text = text[:max_chunk_chars].rstrip() + " ..."
        candidate = f"[chunk_id={chunk_id} pages={page_label}]\\n{text}"
        if total_chars + len(candidate) > max_chars:
            context_truncated = True
            break
        blocks.append(candidate)
        total_chars += len(candidate)
        used_chunks += 1

    context = "\\n\\n".join(blocks).strip() or "[no context]"
    prompt = (
        "You are a strict evidence extraction assistant.\\n"
        "Use only the provided CONTEXT.\\n"
        "Return JSON only (no markdown/code fences) with this exact schema:\\n"
        '{"answer":"...","evidence_quote":"..."}\\n'
        "Rules:\\n"
        "1) evidence_quote must be an exact verbatim snippet copied from CONTEXT.\\n"
        "2) If unsupported, return exactly:\\n"
        '{"answer":"Insufficient evidence in retrieved context.","evidence_quote":""}\\n\\n'
        f"QUESTION:\\n{question}\\n\\n"
        f"CONTEXT:\\n{context}\\n\\n"
        "OUTPUT JSON:"
    )
    return prompt, {
        "context_chunks_used": int(used_chunks),
        "context_chars_used": int(total_chars),
        "context_truncated": bool(context_truncated),
        "context_text": context,
    }


def _parse_json_obj(text: str) -> Optional[dict[str, Any]]:
    s = str(text or "").strip()
    if not s:
        return None
    try:
        x = json.loads(s)
        if isinstance(x, dict):
            return x
    except Exception:
        pass
    m = re.search(r"```(?:json)?\\s*([\\s\\S]*?)\\s*```", s, flags=re.IGNORECASE)
    if m:
        try:
            x = json.loads(m.group(1).strip())
            if isinstance(x, dict):
                return x
        except Exception:
            pass
    l = s.find("{")
    r = s.rfind("}")
    if l >= 0 and r > l:
        try:
            x = json.loads(s[l : r + 1])
            if isinstance(x, dict):
                return x
        except Exception:
            pass
    return None


def _norm_text(s: str) -> str:
    s = str(s or "").lower()
    s = s.replace("£", " ")
    s = re.sub(r"[^a-z0-9%\.\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_num_tokens(s: str) -> set[str]:
    raw = re.findall(r"\d+(?:[\.,]\d+)*(?:\s*%)?", str(s or ""))
    out: set[str] = set()
    for t in raw:
        x = t.replace(" ", "").replace(",", "")
        if x:
            out.add(x)
    return out


def _score_match(expected: Any, extracted: str, answer_type: str) -> tuple[bool, str]:
    if expected is None:
        return False, "not_scored"
    exp = str(expected).strip()
    got = str(extracted or "").strip()
    if not exp:
        return False, "not_scored"
    if not got:
        return False, "incorrect"

    exp_n = _norm_text(exp)
    got_n = _norm_text(got)

    if str(answer_type).lower() == "number":
        exp_nums = _extract_num_tokens(exp)
        got_nums = _extract_num_tokens(got)
        if exp_nums and got_nums and (exp_nums & got_nums):
            if exp_n in got_n or got_n in exp_n:
                return True, "correct"
            return True, "partial"
        return False, "incorrect"

    if exp_n in got_n or got_n in exp_n:
        return True, "correct"

    exp_terms = set(t for t in exp_n.split() if len(t) >= 4)
    got_terms = set(t for t in got_n.split() if len(t) >= 4)
    overlap = len(exp_terms & got_terms)
    if exp_terms and (overlap / len(exp_terms)) >= 0.5:
        return True, "partial"
    return False, "incorrect"


def _contains_casefold(haystack: str, needle: str) -> bool:
    h = str(haystack or "").casefold()
    n = str(needle or "").casefold()
    return bool(n and n in h)


def _df_to_md_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_No rows_"
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals: list[str] = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _write_charts(detail: pd.DataFrame, out_dir: Path) -> list[str]:
    out_paths: list[str] = []
    chart_dir = out_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    # Chart 1: failure-mode distribution by difficulty.
    diff_order = [d for d in ["LEX", "MOD", "STR"] if d in set(detail["difficulty"].astype(str))]
    if not diff_order:
        diff_order = sorted(detail["difficulty"].astype(str).unique().tolist())
    fm = (
        detail.groupby(["difficulty", "failure_mode"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    pivot = fm.pivot(index="difficulty", columns="failure_mode", values="count").fillna(0.0)
    if diff_order:
        pivot = pivot.reindex(diff_order)
    row_sums = pivot.sum(axis=1).replace(0, 1.0)
    pct = pivot.div(row_sums, axis=0)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    bottom = np.zeros(len(pct))
    colors = {
        "strict_correct": "#2ca02c",
        "saw_gold_in_quote_but_answer_incorrect": "#ff7f0e",
        "answer_correct_but_quote_not_supporting_gold": "#1f77b4",
        "did_not_identify_gold": "#d62728",
    }
    x = np.arange(len(pct.index))
    for col in pct.columns:
        vals = pct[col].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, label=str(col), color=colors.get(str(col), None))
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in pct.index])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Share of queries")
    ax.set_title("Strict Evidence Failure Modes by Difficulty")
    ax.legend(title="Failure mode", fontsize=8)
    p1 = chart_dir / "chart_failure_mode_by_difficulty_stacked.png"
    fig.savefig(p1, dpi=180)
    plt.close(fig)
    out_paths.append(str(p1))

    # Chart 2: main rates summary.
    metrics = {
        "Answer Acc": float(detail["answer_correct"].mean()) if len(detail) else 0.0,
        "Quote Support": float(detail["quote_supports_gold"].mean()) if len(detail) else 0.0,
        "Strict Acc": float(detail["strict_correct"].mean()) if len(detail) else 0.0,
        "Retrieval Hit@k": float(detail["retrieval_hit_at_k"].mean()) if len(detail) else 0.0,
    }
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    names = list(metrics.keys())
    vals = [metrics[k] for k in names]
    bars = ax.bar(names, vals, color=["#4c78a8", "#f58518", "#54a24b", "#9c755f"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Rate")
    ax.set_title("Strict Evidence Evaluation Summary Rates")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2.0, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    p2 = chart_dir / "chart_strict_evidence_summary_rates.png"
    fig.savefig(p2, dpi=180)
    plt.close(fig)
    out_paths.append(str(p2))

    return out_paths


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_df = pd.read_csv(args.sample_csv)

    global model
    model = SentenceTransformer(str(args.model_path))
    llm = LocalLLMService()

    doc_cache: dict[str, DocArtifacts] = {}
    rows: list[dict[str, Any]] = []

    total = len(sample_df)
    for i, rec in sample_df.iterrows():
        doc_id = str(rec.get("doc_id") or "").strip()
        qid = str(rec.get("query_id") or "").strip()
        question = str(rec.get("question") or "").strip()
        expected_answer = rec.get("expected_answer")
        answer_type = str(rec.get("answer_type") or "unknown")
        difficulty = str(rec.get("difficulty") or "")

        if doc_id not in doc_cache:
            doc_cache[doc_id] = _load_doc_artifacts(Path(args.data_root) / doc_id)
        doc = doc_cache[doc_id]

        results, _ = _retrieve_hybrid(question=question, doc=doc, k=int(args.k))
        prompt, ctx = _build_prompt(
            question=question,
            results=results,
            max_chunks=int(args.max_context_chunks),
            max_chars=int(args.max_context_chars),
            max_chunk_chars=int(args.max_chunk_chars),
        )

        if args.mode == "constrained_extraction":
            try:
                out = run_constrained_extraction(
                    llm_client=llm,
                    question=question,
                    context=str(ctx.get("context_text", "")),
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=int(args.gen_max_tokens),
                )
                answer = str(out.get("answer") or "").strip()
                quote = str(out.get("evidence_span") or "").strip()
                violations = list(out.get("violations") or [])
                json_ok = True
                llm_status = "ok" if not violations else "guard_violation"
            except Exception:
                answer = ""
                quote = ""
                violations = ["parse_or_generation_error"]
                json_ok = False
                llm_status = "error"
        else:
            gen = llm.generate(prompt=prompt, timeout_seconds=float(args.gen_timeout_seconds))
            parsed = _parse_json_obj(gen.answer or "")
            json_ok = parsed is not None
            answer = str((parsed or {}).get("answer") or "").strip()
            quote = str((parsed or {}).get("evidence_quote") or "").strip()
            violations = []
            llm_status = str(gen.status)

        answer_ok, answer_status = _score_match(expected_answer, answer, answer_type)
        quote_ok, quote_status = _score_match(expected_answer, quote, answer_type)

        strict_ok = bool(answer_ok) and bool(quote_ok)
        if quote_ok and not answer_ok:
            failure_mode = "saw_gold_in_quote_but_answer_incorrect"
        elif answer_ok and not quote_ok:
            failure_mode = "answer_correct_but_quote_not_supporting_gold"
        elif not quote_ok:
            failure_mode = "did_not_identify_gold"
        else:
            failure_mode = "strict_correct"

        expected_pages = []
        if qid in doc.eval_map:
            expected_pages = [int(x) for x in doc.eval_map[qid].get("expected_pages", []) if str(x).isdigit()]
        retrieved_pages = sorted({p for r in results for p in (r.get("pages") or []) if isinstance(p, int)})
        retrieval_hit = bool(expected_pages and any(p in expected_pages for p in retrieved_pages))

        rows.append(
            {
                "doc_id": doc_id,
                "query_id": qid,
                "difficulty": difficulty,
                "answer_type": answer_type,
                "question": question,
                "expected_answer": expected_answer,
                "llm_status": llm_status,
                "json_parse_ok": bool(json_ok),
                "answer": answer,
                "evidence_quote": quote,
                "guard_violations": "|".join(violations),
                "answer_correct": bool(answer_ok),
                "answer_status": answer_status,
                "quote_supports_gold": bool(quote_ok),
                "quote_status": quote_status,
                "strict_correct": bool(strict_ok),
                "failure_mode": failure_mode,
                "evidence_is_verbatim_in_context": bool(_contains_casefold(ctx.get("context_text", ""), quote)) if quote else False,
                "retrieval_hit_at_k": bool(retrieval_hit),
                "context_chunks_used": int(ctx.get("context_chunks_used", 0)),
                "context_chars_used": int(ctx.get("context_chars_used", 0)),
                "context_truncated": bool(ctx.get("context_truncated", False)),
            }
        )

        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"Processed {i + 1}/{total}")

    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "strict_evidence_detail.csv", index=False)

    failure = (
        detail.groupby("failure_mode", dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
    )
    failure["pct"] = failure["count"] / max(1, len(detail))
    failure.to_csv(out_dir / "strict_evidence_failure_modes.csv", index=False)

    by_diff = (
        detail.groupby("difficulty", dropna=False)
        .agg(
            n=("query_id", "count"),
            answer_accuracy=("answer_correct", "mean"),
            quote_support_rate=("quote_supports_gold", "mean"),
            strict_accuracy=("strict_correct", "mean"),
            retrieval_hit_at_k=("retrieval_hit_at_k", "mean"),
        )
        .reset_index()
        .sort_values("difficulty")
    )
    by_diff.to_csv(out_dir / "strict_evidence_by_difficulty.csv", index=False)

    summary = {
        "n_queries": int(len(detail)),
        "answer_accuracy": float(detail["answer_correct"].mean()) if len(detail) else None,
        "quote_support_rate": float(detail["quote_supports_gold"].mean()) if len(detail) else None,
        "strict_accuracy": float(detail["strict_correct"].mean()) if len(detail) else None,
        "json_parse_ok_rate": float(detail["json_parse_ok"].mean()) if len(detail) else None,
        "evidence_verbatim_rate": float(detail["evidence_is_verbatim_in_context"].mean()) if len(detail) else None,
        "retrieval_hit_at_k_rate": float(detail["retrieval_hit_at_k"].mean()) if len(detail) else None,
        "failure_mode_counts": failure.set_index("failure_mode")["count"].to_dict(),
    }
    chart_paths = _write_charts(detail=detail, out_dir=out_dir)
    summary["chart_paths"] = chart_paths
    (out_dir / "strict_evidence_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Strict Evidence-Constrained Extraction",
        "",
        f"- sample_csv: `{args.sample_csv}`",
        f"- n_queries: `{summary['n_queries']}`",
        f"- answer_accuracy: `{summary['answer_accuracy']}`",
        f"- quote_support_rate: `{summary['quote_support_rate']}`",
        f"- strict_accuracy: `{summary['strict_accuracy']}`",
        f"- json_parse_ok_rate: `{summary['json_parse_ok_rate']}`",
        f"- evidence_verbatim_rate: `{summary['evidence_verbatim_rate']}`",
        "",
        "## Failure Modes",
        "",
        _df_to_md_table(failure),
        "",
        "## By Difficulty",
        "",
        _df_to_md_table(by_diff),
    ]
    (out_dir / "strict_evidence_summary.md").write_text("\n".join(md), encoding="utf-8")

    print("Wrote:", out_dir / "strict_evidence_detail.csv")
    print("Wrote:", out_dir / "strict_evidence_failure_modes.csv")
    print("Wrote:", out_dir / "strict_evidence_by_difficulty.csv")
    print("Wrote:", out_dir / "strict_evidence_summary.json")
    print("Wrote:", out_dir / "strict_evidence_summary.md")
    for p in chart_paths:
        print("Wrote:", p)


if __name__ == "__main__":
    main()
