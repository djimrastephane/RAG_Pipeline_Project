from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export canonical consolidated outputs from retrieval_results.json files. "
            "Writes consolidated_answers.csv/jsonl and optional consolidated_table_facts.csv."
        )
    )
    parser.add_argument("--data-root", default="data_processed", help="Root containing per-doc processed folders.")
    parser.add_argument(
        "--results-name",
        default="retrieval_results.json",
        help="Results filename to collect from each document folder.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Which per_k bucket to use for evidence extraction (default: 1).",
    )
    parser.add_argument(
        "--out-dir",
        default="data_processed/consolidated",
        help="Output directory for consolidated exports.",
    )
    parser.add_argument(
        "--snippet-chars",
        type=int,
        default=220,
        help="Max chars for evidence snippet text.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on number of retrieval_results files (0 = all).",
    )
    parser.add_argument(
        "--no-table-facts",
        action="store_true",
        help="Skip consolidated_table_facts export.",
    )
    parser.add_argument(
        "--search-log-jsonl",
        default="",
        help=(
            "Optional JSONL file with API search responses to enrich generation fields. "
            "Accepted line formats include either direct response payload with doc_id, or "
            "{doc_id, response:{...}} wrappers."
        ),
    )
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_json_load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict JSON in {path}")
    return obj


def _to_int_list(v: Any) -> list[int]:
    if v is None:
        return []
    if isinstance(v, list):
        out: list[int] = []
        for x in v:
            try:
                out.append(int(x))
            except Exception:
                continue
        return out
    try:
        return [int(v)]
    except Exception:
        return []


def _safe_topk_bucket(per_k: dict[str, Any], wanted_k: int) -> dict[str, Any]:
    if not isinstance(per_k, dict) or not per_k:
        return {}
    key = str(wanted_k)
    if key in per_k and isinstance(per_k[key], dict):
        return per_k[key]
    # fallback: smallest available k
    numeric_keys = sorted([int(k) for k in per_k.keys() if str(k).isdigit()])
    if numeric_keys:
        return per_k[str(numeric_keys[0])]
    return {}


def _build_chunk_text_lookup(doc_dir: Path) -> dict[str, str]:
    chunks_path = doc_dir / "chunks.parquet"
    if not chunks_path.exists():
        return {}
    try:
        df = pd.read_parquet(chunks_path, columns=["chunk_id", "chunk_id_global", "chunk_text"])
    except Exception:
        return {}
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        txt = str(row.get("chunk_text") or "")
        cid_local = str(row.get("chunk_id") or "").strip()
        cid_global = str(row.get("chunk_id_global") or "").strip()
        if cid_local and cid_local not in out:
            out[cid_local] = txt
        if cid_global and cid_global not in out:
            out[cid_global] = txt
    return out


def _unique_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        s = str(it or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _build_evidence(
    *,
    k_bucket: dict[str, Any],
    fallback_doc_id: str,
    data_root: Path,
    snippet_chars: int,
) -> list[dict[str, Any]]:
    chunk_ids = [str(x) for x in (k_bucket.get("retrieved_chunk_ids") or [])]
    pages = _to_int_list(k_bucket.get("retrieved_pages_ranked") or [])
    doc_ids = [str(x) for x in (k_bucket.get("retrieved_doc_ids_top_k") or [])]
    scores = list(k_bucket.get("retrieved_scores") or [])

    evidence: list[dict[str, Any]] = []
    lookup_cache: dict[str, dict[str, str]] = {}

    n = max(len(chunk_ids), len(pages), len(doc_ids), len(scores))
    for i in range(n):
        chunk_id = chunk_ids[i] if i < len(chunk_ids) else ""
        page = pages[i] if i < len(pages) else None
        doc_id = doc_ids[i] if i < len(doc_ids) else fallback_doc_id
        score = float(scores[i]) if i < len(scores) and scores[i] is not None else None

        if doc_id not in lookup_cache:
            lookup_cache[doc_id] = _build_chunk_text_lookup(data_root / doc_id)
        snippet = ""
        if chunk_id:
            raw = lookup_cache[doc_id].get(chunk_id, "")
            if raw:
                snippet = raw[:snippet_chars].strip()

        evidence.append(
            {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "page": page,
                "snippet": snippet,
                "score": score,
            }
        )
    return evidence


def _as_key(doc_id: str, query_id: Any, question: Any) -> tuple[str, str, str]:
    return (
        str(doc_id or "").strip(),
        str(query_id or "").strip(),
        str(question or "").strip(),
    )


def _load_search_log_generation_map(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """
    Load optional API search JSONL and map generation fields by (doc_id, query_id, question).
    Expected per-line options:
    1) { "doc_id": "...", "query_id": "...", "question": "...", ...response fields... }
    2) { "doc_id": "...", "response": { ...response fields... } }
    """
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            doc_id = str(obj.get("doc_id") or "").strip()
            payload = obj
            if isinstance(obj.get("response"), dict):
                payload = obj.get("response")  # type: ignore[assignment]
                if not doc_id:
                    doc_id = str(obj.get("doc_id") or payload.get("doc_id") or "").strip()
            if not isinstance(payload, dict):
                continue
            qid = payload.get("query_id")
            question = payload.get("question")
            if not doc_id:
                continue
            key = _as_key(doc_id, qid, question)
            out[key] = {
                "generation_status": payload.get("generation_status"),
                "generation_confidence": payload.get("generation_confidence"),
                "low_retrieval_margin": (
                    payload.get("low_retrieval_margin")
                    if payload.get("low_retrieval_margin") is not None
                    else (payload.get("generation_debug", {}) or {}).get("low_retrieval_margin")
                ),
                "retrieval_margin": (
                    payload.get("retrieval_margin")
                    if payload.get("retrieval_margin") is not None
                    else (payload.get("generation_debug", {}) or {}).get("retrieval_margin")
                ),
                "answer_mode": "generated" if payload.get("generated_answer") else "deterministic",
                "final_answer": payload.get("generated_answer") or payload.get("predicted_answer"),
            }
    return out


def export_consolidated_answers(
    *,
    data_root: Path,
    results_name: str,
    top_k: int,
    out_dir: Path,
    snippet_chars: int,
    max_files: int,
    search_log_generation_map: Optional[dict[tuple[str, str, str], dict[str, Any]]] = None,
) -> tuple[Path, Path, int]:
    result_files = sorted(data_root.glob(f"*/{results_name}"))
    if max_files > 0:
        result_files = result_files[:max_files]

    rows: list[dict[str, Any]] = []

    for rp in result_files:
        doc_dir = rp.parent
        doc_id_from_path = doc_dir.name
        payload = safe_json_load(rp)
        run_info = payload.get("run_info", {}) if isinstance(payload.get("run_info"), dict) else {}
        run_utc = str(run_info.get("run_utc") or utc_now_iso())
        run_id = f"{doc_id_from_path}:{rp.stem}:{run_utc}"
        retrieval_scope = "doc"
        results = payload.get("results", [])
        if not isinstance(results, list):
            continue

        for item in results:
            if not isinstance(item, dict):
                continue
            item_doc_id = str(item.get("doc_id") or doc_id_from_path)
            per_k = item.get("per_k", {})
            k_bucket = _safe_topk_bucket(per_k if isinstance(per_k, dict) else {}, top_k)
            evidence = _build_evidence(
                k_bucket=k_bucket,
                fallback_doc_id=item_doc_id,
                data_root=data_root,
                snippet_chars=snippet_chars,
            )
            docs_considered = _unique_keep_order([str(x) for x in (k_bucket.get("retrieved_doc_ids_top_k") or [item_doc_id])])
            docs_evidence = _unique_keep_order([str(e.get("doc_id") or "") for e in evidence])
            retrieved_scores = list(k_bucket.get("retrieved_scores") or [])
            retrieval_margin: Optional[float] = None
            if len(retrieved_scores) >= 2:
                try:
                    retrieval_margin = float(retrieved_scores[0]) - float(retrieved_scores[1])
                except Exception:
                    retrieval_margin = None

            row = {
                "run_id": run_id,
                "query_id": item.get("query_id"),
                "question": item.get("question"),
                "retrieval_scope": retrieval_scope,
                "docs_considered": json.dumps(docs_considered, ensure_ascii=False),
                "docs_evidence": json.dumps(docs_evidence, ensure_ascii=False),
                "final_answer": item.get("extracted_answer"),
                "answer_status": item.get("answer_status"),
                "answer_mode": "deterministic",
                "grounded": bool(len(evidence) > 0),
                "evidence_count": int(len(evidence)),
                "evidence": json.dumps(evidence, ensure_ascii=False),
                "generation_status": "skipped",
                "generation_confidence": None,
                "low_retrieval_margin": None,
                "retrieval_margin": retrieval_margin,
                "topk": int(top_k),
                "timestamp_utc": run_utc,
            }
            # Prefer generation fields from retrieval_results item if present.
            if item.get("generation_status") is not None:
                row["generation_status"] = item.get("generation_status")
            if item.get("generation_confidence") is not None:
                row["generation_confidence"] = item.get("generation_confidence")
            if item.get("low_retrieval_margin") is not None:
                row["low_retrieval_margin"] = item.get("low_retrieval_margin")
            if item.get("retrieval_margin") is not None:
                row["retrieval_margin"] = item.get("retrieval_margin")
            if item.get("generated_answer") is not None:
                row["final_answer"] = item.get("generated_answer")
                row["answer_mode"] = "generated"

            # Optional enrichment from API search logs.
            if search_log_generation_map:
                key = _as_key(item_doc_id, item.get("query_id"), item.get("question"))
                enrich = search_log_generation_map.get(key)
                if enrich:
                    for fld in (
                        "generation_status",
                        "generation_confidence",
                        "low_retrieval_margin",
                        "retrieval_margin",
                        "answer_mode",
                        "final_answer",
                    ):
                        if enrich.get(fld) is not None:
                            row[fld] = enrich.get(fld)

            rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    answers_csv = out_dir / "consolidated_answers.csv"
    answers_jsonl = out_dir / "consolidated_answers.jsonl"

    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(answers_csv, index=False)
        with answers_jsonl.open("w", encoding="utf-8") as f:
            for rec in df.to_dict(orient="records"):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    else:
        pd.DataFrame().to_csv(answers_csv, index=False)
        answers_jsonl.write_text("", encoding="utf-8")

    return answers_csv, answers_jsonl, len(df)


def export_consolidated_table_facts(*, data_root: Path, out_dir: Path) -> tuple[Path, int]:
    rows: list[pd.DataFrame] = []
    for tf_path in sorted(data_root.glob("*/table_facts.parquet")):
        try:
            df = pd.read_parquet(tf_path)
        except Exception:
            continue
        if df.empty:
            continue
        doc_id = tf_path.parent.name
        if "doc_id" not in df.columns:
            df = df.copy()
            df["doc_id"] = doc_id
        rows.append(df)

    out_path = out_dir / "consolidated_table_facts.csv"
    if rows:
        merged = pd.concat(rows, ignore_index=True)
        merged.to_csv(out_path, index=False)
        return out_path, int(len(merged))
    pd.DataFrame().to_csv(out_path, index=False)
    return out_path, 0


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    search_log_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    if str(args.search_log_jsonl).strip():
        search_log_path = Path(str(args.search_log_jsonl)).expanduser().resolve()
        search_log_map = _load_search_log_generation_map(search_log_path)
        print(f"Loaded generation enrichment rows: {len(search_log_map)} from {search_log_path}")

    answers_csv, answers_jsonl, n_rows = export_consolidated_answers(
        data_root=data_root,
        results_name=str(args.results_name),
        top_k=int(args.top_k),
        out_dir=out_dir,
        snippet_chars=int(args.snippet_chars),
        max_files=int(args.max_files),
        search_log_generation_map=search_log_map,
    )
    print(f"Wrote answers CSV:   {answers_csv}")
    print(f"Wrote answers JSONL: {answers_jsonl}")
    print(f"Total consolidated answer rows: {n_rows}")

    if not args.no_table_facts:
        table_path, n_facts = export_consolidated_table_facts(
            data_root=data_root,
            out_dir=out_dir,
        )
        print(f"Wrote table facts CSV: {table_path}")
        print(f"Total consolidated table facts rows: {n_facts}")


if __name__ == "__main__":
    main()
