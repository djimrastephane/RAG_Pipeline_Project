from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from generation.constrained_extraction import run_constrained_extraction
from rag_pdf.services.local_llm_service import LocalLLMService


ARMS = ["baseline", "row_preserving", "two_stage", "row_blocks"]
RRF_K = 20
RRF_DENSE_WEIGHT = 0.5
RRF_BM25_WEIGHT = 2.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run paired table-chunking ablation (A/B/C) on sampled 50 queries.")
    p.add_argument(
        "--sample-csv",
        default="results/context_chunks_3_vs_5_stats_2026-03-02/sampled_50_queries.csv",
    )
    p.add_argument("--source-data-root", default="data_processed", help="Existing root with eval_set/sections for GT tagging.")
    p.add_argument("--out-root", default="runs/table_chunking_ablation")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-path", default="models/all-MiniLM-L6-v2")
    p.add_argument("--max-tokens", type=int, default=120)
    p.add_argument("--bootstrap-iters", type=int, default=1000)
    p.add_argument("--reuse-existing", action="store_true", default=True)
    return p.parse_args()


def _resolve_pdf_path(doc_id: str) -> Path:
    data_dir = repo_root / "Data"
    matches = list(data_dir.rglob(f"{doc_id}.pdf"))
    if not matches:
        raise FileNotFoundError(f"Could not find PDF for doc_id={doc_id} under {data_dir}")
    return matches[0]


def _run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, cwd=str(repo_root), env=env)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9\\-]{1,}", str(text or "").lower())


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)


def _rrf_fuse(dense_ranked: list[int], bm25_ranked: list[int]) -> list[int]:
    scores: dict[int, float] = {}
    for rank, idx in enumerate(dense_ranked, start=1):
        scores[idx] = scores.get(idx, 0.0) + (RRF_DENSE_WEIGHT / float(RRF_K + rank))
    for rank, idx in enumerate(bm25_ranked, start=1):
        scores[idx] = scores.get(idx, 0.0) + (RRF_BM25_WEIGHT / float(RRF_K + rank))
    return [i for i, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


class BM25Index:
    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        from collections import Counter

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
        from collections import Counter

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


def _load_doc_retrieval_cache(data_root: Path, doc_id: str, model: SentenceTransformer) -> dict[str, Any]:
    chunks = pd.read_parquet(data_root / doc_id / "chunks.parquet").reset_index(drop=True)
    texts = chunks["chunk_text"].fillna("").astype(str).tolist()
    embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False).astype("float32")
    embs = _l2_normalize(embs).astype("float32")
    bm25 = BM25Index([_tokenize(t) for t in texts], k1=1.5, b=0.75)
    return {"chunks": chunks, "embs": embs, "bm25": bm25}


def _retrieve_topk(cache: dict[str, Any], question: str, k: int, model: SentenceTransformer) -> list[dict[str, Any]]:
    chunks: pd.DataFrame = cache["chunks"]
    embs: np.ndarray = cache["embs"]
    bm25: BM25Index = cache["bm25"]

    q_emb = model.encode([question], convert_to_numpy=True, normalize_embeddings=False).astype("float32")
    q_emb = _l2_normalize(q_emb).astype("float32")
    dense_scores = np.dot(embs, q_emb[0])
    dense_ranked = np.argsort(-dense_scores).tolist()
    bm25_scores = bm25.score_query(_tokenize(question))
    bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
    fused = _rrf_fuse(dense_ranked=dense_ranked, bm25_ranked=bm25_ranked)[: int(k)]

    out: list[dict[str, Any]] = []
    for rank, idx in enumerate(fused, start=1):
        row = chunks.iloc[int(idx)]
        pages_raw = row.get("pages")
        pages: list[int] = []
        if isinstance(pages_raw, list):
            for p in pages_raw:
                try:
                    pages.append(int(p))
                except Exception:
                    pass
        if not pages:
            for col in ("page_start", "page_end"):
                try:
                    pv = int(row.get(col))
                    if pv not in pages:
                        pages.append(pv)
                except Exception:
                    pass
        out.append(
            {
                "rank": rank,
                "chunk_id": str(row.get("chunk_id_global") or row.get("chunk_id") or ""),
                "pages": pages,
                "chunk_text": str(row.get("chunk_text") or ""),
            }
        )
    return out


def _load_eval_map(source_data_root: Path, docs: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for doc in docs:
        p = source_data_root / doc / "eval_set.json"
        if not p.exists():
            out[doc] = {}
            continue
        raw = json.loads(p.read_text(encoding="utf-8"))
        rows = raw.get("queries", []) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
        m: dict[str, dict[str, Any]] = {}
        for r in rows:
            if not isinstance(r, dict):
                continue
            qid = str(r.get("query_id") or "").strip()
            if qid:
                m[qid] = r
        out[doc] = m
    return out


def _financial_pages_map(source_data_root: Path, docs: list[str]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for doc in docs:
        p = source_data_root / doc / "sections.parquet"
        pages: set[int] = set()
        if p.exists():
            df = pd.read_parquet(p)
            for _, row in df.iterrows():
                sec = str(row.get("section_title") or "").strip().lower()
                if "financial statements" not in sec:
                    continue
                try:
                    ps = int(row.get("page_start"))
                    pe = int(row.get("page_end"))
                except Exception:
                    continue
                for x in range(ps, pe + 1):
                    pages.add(x)
        out[doc] = pages
    return out


def _norm_text(s: str) -> str:
    s = str(s or "").lower()
    s = s.replace("£", " ")
    s = __import__("re").sub(r"[^a-z0-9%\.\s]", " ", s)
    s = __import__("re").sub(r"\s+", " ", s).strip()
    return s


def _extract_num_tokens(s: str) -> set[str]:
    import re

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
    if not exp or not got:
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
    if exp_terms and overlap / len(exp_terms) >= 0.5:
        return True, "partial"
    return False, "incorrect"


def _build_context_from_results(results: list[dict[str, Any]], max_chunks: int = 3) -> str:
    blocks: list[str] = []
    for r in results[:max_chunks]:
        cid = str(r.get("chunk_id") or "")
        pages = [int(x) for x in (r.get("pages") or []) if str(x).isdigit()]
        page_label = ",".join(str(p) for p in pages) if pages else "NA"
        text = str(r.get("chunk_text") or "").strip()
        if not text:
            continue
        blocks.append(f"[chunk_id={cid} pages={page_label}]\n{text}")
    return "\n\n".join(blocks).strip()


def _cited_pages_from_span(evidence_span: str | None, top_results: list[dict[str, Any]]) -> list[int]:
    if not evidence_span:
        return []
    out: set[int] = set()
    for r in top_results:
        txt = str(r.get("chunk_text") or "")
        if evidence_span in txt:
            for p in r.get("pages") or []:
                try:
                    out.add(int(p))
                except Exception:
                    continue
    return sorted(out)


def _hash_retrieval_cfg(cfg: dict[str, Any]) -> str:
    s = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _metric_summary(df: pd.DataFrame) -> dict[str, Any]:
    n = len(df)
    hit = float(df["hit3"].mean()) if n else 0.0
    strict = float(df["strict_correct"].mean()) if n else 0.0
    quote = float(df["quote_support"].mean()) if n else 0.0
    hit_df = df[df["hit3"] == 1]
    p_strict_hit = float(hit_df["strict_correct"].mean()) if len(hit_df) else None
    fail = df["failure_mode"].value_counts(dropna=False).to_dict()
    return {
        "n": int(n),
        "hit_at_3": hit,
        "strict_accuracy": strict,
        "quote_support_rate": quote,
        "p_strict_given_hit3": p_strict_hit,
        "failure_mode_counts": fail,
    }


def _mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    b_only = int(np.sum((a == 1) & (b == 0)))
    c_only = int(np.sum((a == 0) & (b == 1)))
    n = b_only + c_only
    if n == 0:
        return {"b_only": b_only, "c_only": c_only, "n_discordant": 0, "p_value": 1.0}
    k = min(b_only, c_only)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / float(2 ** n)
    p = min(1.0, 2.0 * tail)
    return {"b_only": b_only, "c_only": c_only, "n_discordant": int(n), "p_value": float(p)}


def _bootstrap_deltas(df_a: pd.DataFrame, df_b: pd.DataFrame, iters: int, seed: int) -> dict[str, Any]:
    assert len(df_a) == len(df_b)
    rng = np.random.default_rng(seed)
    n = len(df_a)
    d_strict: list[float] = []
    d_psh: list[float] = []

    a_strict = df_a["strict_correct"].to_numpy(dtype=float)
    b_strict = df_b["strict_correct"].to_numpy(dtype=float)
    a_hit = df_a["hit3"].to_numpy(dtype=int)
    b_hit = df_b["hit3"].to_numpy(dtype=int)

    for _ in range(int(iters)):
        idx = rng.integers(0, n, size=n)
        as_mean = float(np.mean(a_strict[idx]))
        bs_mean = float(np.mean(b_strict[idx]))
        d_strict.append(bs_mean - as_mean)

        a_mask = a_hit[idx] == 1
        b_mask = b_hit[idx] == 1
        a_psh = float(np.mean(a_strict[idx][a_mask])) if np.any(a_mask) else np.nan
        b_psh = float(np.mean(b_strict[idx][b_mask])) if np.any(b_mask) else np.nan
        if not np.isnan(a_psh) and not np.isnan(b_psh):
            d_psh.append(b_psh - a_psh)

    def _ci(arr: list[float]) -> dict[str, Any]:
        if not arr:
            return {"mean": None, "ci95_low": None, "ci95_high": None}
        x = np.asarray(arr, dtype=float)
        return {
            "mean": float(np.mean(x)),
            "ci95_low": float(np.percentile(x, 2.5)),
            "ci95_high": float(np.percentile(x, 97.5)),
        }

    return {
        "delta_strict_accuracy": _ci(d_strict),
        "delta_p_strict_given_hit3": _ci(d_psh),
    }


def _run_arm(
    arm: str,
    *,
    sample_df: pd.DataFrame,
    docs: list[str],
    out_root: Path,
    source_data_root: Path,
    k: int,
    model_path: Path,
    max_tokens: int,
    financial_pages: dict[str, set[int]],
    eval_map: dict[str, dict[str, dict[str, Any]]],
    reuse_existing: bool,
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    arm_root = out_root / arm
    data_root = arm_root / "data_processed"
    arm_root.mkdir(parents=True, exist_ok=True)

    if not reuse_existing or not all((data_root / d / "chunks.parquet").exists() for d in docs):
        for doc in docs:
            pdf = _resolve_pdf_path(doc)
            _run_cmd(
                [
                    sys.executable,
                    "scripts/preprocess_hybrid.py",
                    "--pdf-path",
                    str(pdf),
                    "--out-root",
                    str(data_root),
                    "--table-chunking",
                    arm,
                ]
            )
    model = SentenceTransformer(str(model_path))
    doc_cache: dict[str, dict[str, Any]] = {}
    llm = LocalLLMService()
    llm.temperature = 0.0
    llm.top_p = 1.0
    llm.max_tokens = int(max_tokens)

    rows: list[dict[str, Any]] = []
    retrieval_cfg = {
        "retrieval_mode": "hybrid_rrf_dense_bm25",
        "rrf_k": int(RRF_K),
        "dense_weight": float(RRF_DENSE_WEIGHT),
        "bm25_weight": float(RRF_BM25_WEIGHT),
        "k": int(k),
    }
    retrieval_hash = _hash_retrieval_cfg(retrieval_cfg)

    total_queries = len(sample_df)
    for idx, (_, r) in enumerate(sample_df.iterrows(), start=1):
        doc_id = str(r["doc_id"])
        query_id = str(r["query_id"])
        question = str(r["question"])
        difficulty = str(r.get("difficulty") or "")
        answer_type = str(r.get("answer_type") or "unknown")

        gt = eval_map.get(doc_id, {}).get(query_id, {})
        gold_pages = [int(x) for x in gt.get("expected_pages", []) if str(x).isdigit()]
        expected_answer = gt.get("expected_answer", r.get("expected_answer"))

        if doc_id not in doc_cache:
            doc_cache[doc_id] = _load_doc_retrieval_cache(data_root=data_root, doc_id=doc_id, model=model)
        results = _retrieve_topk(cache=doc_cache[doc_id], question=question, k=int(k), model=model)
        top3 = results[:3]
        predicted_pages = sorted({int(p) for rr in top3 for p in (rr.get("pages") or []) if str(p).isdigit()})
        hit3 = int(bool(gold_pages and any(p in gold_pages for p in predicted_pages)))

        context = _build_context_from_results(top3, max_chunks=3)
        t0 = time.perf_counter()
        try:
            gen = run_constrained_extraction(
                llm_client=llm,
                question=question,
                context=context,
                temperature=0.0,
                top_p=1.0,
                max_tokens=int(max_tokens),
            )
        except Exception:
            gen = {"answer": None, "evidence_span": None, "violations": ["parse_or_generation_error"]}
        latency_ms = (time.perf_counter() - t0) * 1000.0

        answer = gen.get("answer")
        evidence_span = gen.get("evidence_span")
        violations = list(gen.get("violations") or [])

        answer_ok, _ = _score_match(expected_answer, str(answer or ""), answer_type)
        quote_ok, _ = _score_match(expected_answer, str(evidence_span or ""), answer_type)
        strict_ok = int(bool(answer_ok and quote_ok))

        if strict_ok:
            failure_mode = "strict_correct"
        elif quote_ok and not answer_ok:
            failure_mode = "saw_gold_but_answer_incorrect"
        else:
            failure_mode = "did_not_identify_gold"

        ev_layout = str(gt.get("evidence_layout", r.get("evidence_layout", "")) or "").strip().lower()
        table_query = int(ev_layout == "table")
        financial_table_query = int(any(int(p) in financial_pages.get(doc_id, set()) for p in gold_pages))
        cited_pages = _cited_pages_from_span(evidence_span, top3)

        rows.append(
            {
                "arm": arm,
                "query_id": query_id,
                "doc_id": doc_id,
                "difficulty": difficulty,
                "gold_pages": json.dumps(gold_pages),
                "predicted_pages_top3": json.dumps(predicted_pages),
                "hit3": hit3,
                "strict_correct": strict_ok,
                "quote_support": int(bool(quote_ok)),
                "failure_mode": failure_mode,
                "extracted_answer": answer,
                "cited_pages": json.dumps(cited_pages),
                "cited_spans": json.dumps([evidence_span] if evidence_span else []),
                "context_chars_used": int(len(context)),
                "latency_ms": float(round(latency_ms, 3)),
                "table_query": table_query,
                "financial_table_query": financial_table_query,
                "violations": "|".join(violations),
            }
        )

        if idx % 5 == 0 or idx == total_queries:
            pd.DataFrame(rows).to_csv(arm_root / "per_query_results.csv", index=False)
            print(f"[{arm}] completed {idx}/{total_queries} queries")

    per_query = pd.DataFrame(rows)
    per_query.to_csv(arm_root / "per_query_results.csv", index=False)

    summary = {
        "arm": arm,
        "retrieval_settings_hash": retrieval_hash,
        "metrics_all": _metric_summary(per_query),
        "metrics_table_query": _metric_summary(per_query[per_query["table_query"] == 1].copy()),
        "n_table_query": int((per_query["table_query"] == 1).sum()),
        "n_financial_table_query": int((per_query["financial_table_query"] == 1).sum()),
    }
    (arm_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return per_query, summary, retrieval_hash


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    sample_df = pd.read_csv(args.sample_csv)
    keep_cols = ["doc_id", "query_id", "question", "expected_answer", "answer_type", "difficulty"]
    if "evidence_layout" in sample_df.columns:
        keep_cols.append("evidence_layout")
    sample_df = sample_df[keep_cols].copy()
    docs = sorted(sample_df["doc_id"].astype(str).unique().tolist())

    source_data_root = Path(args.source_data_root)
    eval_map = _load_eval_map(source_data_root=source_data_root, docs=docs)
    financial_pages = _financial_pages_map(source_data_root=source_data_root, docs=docs)

    arm_results: dict[str, pd.DataFrame] = {}
    arm_summaries: dict[str, dict[str, Any]] = {}
    retrieval_hashes: dict[str, str] = {}

    for arm in ARMS:
        per_q, summary, r_hash = _run_arm(
            arm,
            sample_df=sample_df,
            docs=docs,
            out_root=out_root,
            source_data_root=source_data_root,
            k=int(args.k),
            model_path=Path(args.model_path),
            max_tokens=int(args.max_tokens),
            financial_pages=financial_pages,
            eval_map=eval_map,
            reuse_existing=bool(args.reuse_existing),
        )
        arm_results[arm] = per_q
        arm_summaries[arm] = summary
        retrieval_hashes[arm] = r_hash

    # Sanity checks
    n0 = len(sample_df)
    qid_ref = arm_results[ARMS[0]]["query_id"].tolist()
    for arm in ARMS:
        df = arm_results[arm]
        assert len(df) == n0, f"Query count mismatch in {arm}: {len(df)} vs {n0}"
        assert df["query_id"].tolist() == qid_ref, f"Query order mismatch in {arm}"

    hash_ref = retrieval_hashes[ARMS[0]]
    for arm in ARMS[1:]:
        assert retrieval_hashes[arm] == hash_ref, (
            f"Retrieval settings hash mismatch: baseline={hash_ref}, {arm}={retrieval_hashes[arm]}"
        )

    warnings: list[str] = []
    hit_baseline = float(arm_summaries["baseline"]["metrics_all"]["hit_at_3"])
    for arm in [a for a in ARMS if a != "baseline"]:
        hit = float(arm_summaries[arm]["metrics_all"]["hit_at_3"])
        if abs(hit - hit_baseline) > 0.05:
            msg = (
                f"WARNING: Hit@3 changed by more than 0.05 for {arm} vs baseline "
                f"({hit_baseline:.4f} -> {hit:.4f}); isolation may be broken."
            )
            print(msg)
            warnings.append(msg)

    combined_dir = out_root / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    long_df = pd.concat([arm_results[a] for a in ARMS], ignore_index=True)
    long_df.to_csv(combined_dir / "per_query_long.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for scope in ["all", "table_query"]:
            m = arm_summaries[arm]["metrics_all" if scope == "all" else "metrics_table_query"]
            summary_rows.append(
                {
                    "arm": arm,
                    "scope": scope,
                    "n": m["n"],
                    "hit_at_3": m["hit_at_3"],
                    "strict_accuracy": m["strict_accuracy"],
                    "quote_support_rate": m["quote_support_rate"],
                    "p_strict_given_hit3": m["p_strict_given_hit3"],
                    "failure_mode_counts": json.dumps(m["failure_mode_counts"], sort_keys=True),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(combined_dir / "summary_by_arm.csv", index=False)

    paired_tests: dict[str, Any] = {}
    paired_bootstrap: dict[str, Any] = {}
    comparisons = [("baseline", arm) for arm in ARMS if arm != "baseline"]
    for scope_name in ["all", "table_query"]:
        paired_tests[scope_name] = {}
        paired_bootstrap[scope_name] = {}
        for a, b in comparisons:
            da = arm_results[a].copy()
            db = arm_results[b].copy()
            if scope_name == "table_query":
                da = da[da["table_query"] == 1].copy()
                db = db[db["table_query"] == 1].copy()
            da = da.sort_values("query_id").reset_index(drop=True)
            db = db.sort_values("query_id").reset_index(drop=True)
            common = sorted(set(da["query_id"]).intersection(set(db["query_id"])))
            da = da[da["query_id"].isin(common)].sort_values("query_id").reset_index(drop=True)
            db = db[db["query_id"].isin(common)].sort_values("query_id").reset_index(drop=True)
            paired_tests[scope_name][f"{a}_vs_{b}"] = _mcnemar_exact(
                da["strict_correct"].to_numpy(dtype=int),
                db["strict_correct"].to_numpy(dtype=int),
            )
            paired_bootstrap[scope_name][f"{a}_vs_{b}"] = _bootstrap_deltas(
                da,
                db,
                iters=int(args.bootstrap_iters),
                seed=int(args.seed),
            )

    (combined_dir / "paired_tests.json").write_text(json.dumps(paired_tests, indent=2), encoding="utf-8")
    (combined_dir / "paired_bootstrap.json").write_text(json.dumps(paired_bootstrap, indent=2), encoding="utf-8")

    readme = [
        "# Table Chunking Ablation",
        "",
        f"- sample_csv: `{args.sample_csv}`",
        f"- docs: `{', '.join(docs)}`",
        f"- k: `{args.k}`",
        "- fixed retrieval: current hybrid default (SearchService)",
        "- fixed constrained prompt/model settings; only table chunking changes",
        f"- arms: `{', '.join(ARMS)}`",
        f"- warnings: `{len(warnings)}`",
    ]
    if warnings:
        readme.append("")
        readme.extend([f"- {w}" for w in warnings])
    (combined_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print("Saved:", combined_dir / "per_query_long.csv")
    print("Saved:", combined_dir / "summary_by_arm.csv")
    print("Saved:", combined_dir / "paired_tests.json")
    print("Saved:", combined_dir / "paired_bootstrap.json")
    print("Saved:", combined_dir / "README.md")


if __name__ == "__main__":
    main()
