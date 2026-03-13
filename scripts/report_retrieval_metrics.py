from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FAILURE_STAGE_BY_TYPE = {
    "FP1_MISSING_CONTENT": "retrieval",
    "FP2_MISSED_TOP_RANK": "retrieval",
    "FP3_NOT_IN_CONTEXT": "retrieval",
    "FP4_NOT_EXTRACTED": "generation",
    "FP5_WRONG_FORMAT": "generation",
    "FP6_INCORRECT_SPECIFICITY": "generation",
    "FP7_INCOMPLETE": "generation",
    "HIT": "none",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build retrieval metrics report (CSV + Markdown + LaTeX)."
    )
    parser.add_argument(
        "--data-root",
        default="data_processed",
        help="Root directory containing per-document outputs.",
    )
    parser.add_argument(
        "--docs",
        default="Grampian-2022-2023,Grampian-2023-2024,Grampian-2024-2025",
        help="Comma-separated doc IDs to include.",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Output CSV path. Defaults to <data-root>/retrieval_report.csv.",
    )
    parser.add_argument(
        "--out-md",
        default=None,
        help="Output Markdown path. Defaults to <data-root>/retrieval_report.md.",
    )
    parser.add_argument(
        "--out-tex",
        default=None,
        help="Output LaTeX path. Defaults to <data-root>/retrieval_report.tex.",
    )
    parser.add_argument(
        "--out-queries-csv",
        default=None,
        help="Output per-query CSV path. Defaults to <data-root>/retrieval_queries_report.csv.",
    )
    parser.add_argument(
        "--out-queries-md",
        default=None,
        help="Output per-query Markdown path. Defaults to <data-root>/retrieval_queries_report.md.",
    )
    parser.add_argument(
        "--out-queries-tex",
        default=None,
        help="Output per-query LaTeX path. Defaults to <data-root>/retrieval_queries_report.tex.",
    )
    parser.add_argument(
        "--out-failure-summary",
        default=None,
        help="Output failure-type summary CSV path. Defaults to <data-root>/retrieval_failure_summary.csv.",
    )
    parser.add_argument(
        "--out-table-misses",
        default=None,
        help="Output table-query misses at k=1. Defaults to <data-root>/retrieval_table_misses_k1.csv.",
    )
    return parser.parse_args()


def resolve_output_path(data_root: Path, output_path: str | None, filename: str) -> Path:
    if output_path:
        return Path(output_path)
    return data_root / filename


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    doc_ids = [d.strip() for d in args.docs.split(",") if d.strip()]

    rows: list[dict] = []
    detail_rows: list[dict] = []
    table_misses: list[dict] = []
    for doc_id in doc_ids:
        metrics_path = data_root / doc_id / "retrieval_metrics_hybrid.json"
        if not metrics_path.exists():
            metrics_path = data_root / doc_id / "retrieval_metrics.json"
        if not metrics_path.exists():
            print(f"Missing: {metrics_path}")
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics_by_k = metrics.get("metrics_by_k", {})
        for k, m in metrics_by_k.items():
            rows.append(
                {
                    "doc_id": doc_id,
                    "k": int(k),
                    "hit_rate": m.get("page_hit_rate_at_k", 0.0),
                    "recall": m.get("mean_page_recall_at_k", 0.0),
                    "mrr": m.get("mean_page_mrr_at_k", 0.0),
                    "precision": m.get("mean_page_precision_at_k", 0.0),
                }
            )
        results_path = data_root / doc_id / "retrieval_results_hybrid.json"
        if not results_path.exists():
            results_path = data_root / doc_id / "retrieval_results.json"
        meta_path = data_root / doc_id / "chunk_meta.parquet"
        if not results_path.exists() or not meta_path.exists():
            print(f"Missing per-doc inputs for table/text breakdown: {doc_id}")
            continue
        results = json.loads(results_path.read_text(encoding="utf-8"))
        meta = pd.read_parquet(meta_path)
        if "is_table" not in meta.columns:
            print(f"Missing is_table in chunk_meta.parquet for {doc_id}")
            continue
        table_pages = set(meta.loc[meta["is_table"] == True, "page_start"].dropna().astype(int).tolist())
        for item in results.get("results", []):
            expected_pages = item.get("expected_pages") or []
            is_table_query = bool(set(expected_pages) & table_pages)
            per_k = item.get("per_k", {})
            for k in per_k.keys():
                kdata = per_k.get(k, {})
                detail_rows.append(
                    {
                        "doc_id": doc_id,
                        "k": int(k),
                        "is_table_query": is_table_query,
                        "page_recall_at_k": kdata.get("page_recall_at_k", 0.0),
                        "page_precision_at_k": kdata.get("page_precision_at_k", 0.0),
                        "page_mrr_at_k": kdata.get("page_mrr_at_k", 0.0),
                    }
                )
            k1 = per_k.get("1", {})
            if is_table_query and k1.get("page_recall_at_k", 0.0) <= 0:
                table_misses.append(
                    {
                        "doc_id": doc_id,
                        "query_id": item.get("query_id"),
                        "question": item.get("question"),
                        "expected_pages": expected_pages,
                        "top_pages": k1.get("retrieved_pages_ranked"),
                        "failure_type": item.get("failure_type") or k1.get("failure_stage"),
                    }
                )

    if not rows:
        print("No metrics found.")
        return

    df = pd.DataFrame(rows).sort_values(["doc_id", "k"])
    if detail_rows:
        ddf = pd.DataFrame(detail_rows)
        table_ddf = ddf[ddf["is_table_query"] == True]
        text_ddf = ddf[ddf["is_table_query"] == False]

        table_stats = (
            table_ddf.groupby(["doc_id", "k"])
            .agg(
                table_query_count=("page_recall_at_k", "size"),
                table_hit_rate=("page_recall_at_k", lambda s: float((s > 0).mean())),
                table_recall=("page_recall_at_k", "mean"),
                table_precision=("page_precision_at_k", "mean"),
                table_mrr=("page_mrr_at_k", "mean"),
            )
            .reset_index()
        )
        text_stats = (
            text_ddf.groupby(["doc_id", "k"])
            .agg(
                text_query_count=("page_recall_at_k", "size"),
                text_hit_rate=("page_recall_at_k", lambda s: float((s > 0).mean())),
                text_recall=("page_recall_at_k", "mean"),
                text_precision=("page_precision_at_k", "mean"),
                text_mrr=("page_mrr_at_k", "mean"),
            )
            .reset_index()
        )
        df = df.merge(table_stats, on=["doc_id", "k"], how="left")
        df = df.merge(text_stats, on=["doc_id", "k"], how="left")
        if "table_hit_rate" in df.columns and "text_hit_rate" in df.columns:
            df["delta_hit_rate"] = df["table_hit_rate"] - df["text_hit_rate"]
        if "table_recall" in df.columns and "text_recall" in df.columns:
            df["delta_recall"] = df["table_recall"] - df["text_recall"]
        if "table_precision" in df.columns and "text_precision" in df.columns:
            df["delta_precision"] = df["table_precision"] - df["text_precision"]
        if "table_mrr" in df.columns and "text_mrr" in df.columns:
            df["delta_mrr"] = df["table_mrr"] - df["text_mrr"]

    out_csv = resolve_output_path(data_root, args.out_csv, "retrieval_report.csv")
    out_md = resolve_output_path(data_root, args.out_md, "retrieval_report.md")
    out_tex = resolve_output_path(data_root, args.out_tex, "retrieval_report.tex")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_csv, index=False)
    out_md.write_text(df.to_markdown(index=False), encoding="utf-8")
    out_tex.write_text(
        df.to_latex(index=False, float_format="%.3f"),
        encoding="utf-8",
    )

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_tex}")

    # Per-query report
    query_rows: list[dict] = []
    failure_types: list[str] = []
    for doc_id in doc_ids:
        results_path = data_root / doc_id / "retrieval_results_hybrid.json"
        if not results_path.exists():
            results_path = data_root / doc_id / "retrieval_results.json"
        chunks_path = data_root / doc_id / "chunks.parquet"
        if not results_path.exists() or not chunks_path.exists():
            print(f"Missing per-query inputs for {doc_id}")
            continue
        results = json.loads(results_path.read_text(encoding="utf-8"))
        chunks = pd.read_parquet(chunks_path)
        chunk_text_by_id = {}
        for _, row in chunks.iterrows():
            cid = row.get("chunk_id_global") or row.get("chunk_id")
            if cid:
                chunk_text_by_id[str(cid)] = str(row.get("chunk_text") or "")
        chunk_meta_by_id = {}
        for _, row in chunks.iterrows():
            cid = row.get("chunk_id_global") or row.get("chunk_id")
            if cid:
                chunk_meta_by_id[str(cid)] = {
                    "page_start": row.get("page_start"),
                    "section_title": row.get("section_title"),
                    "subsection_title": row.get("subsection_title"),
                }
        for item in results.get("results", []):
            per_k = item.get("per_k", {})
            k1 = per_k.get("1", {})
            chunk_ids = k1.get("retrieved_chunk_ids") or []
            if isinstance(chunk_ids, str):
                chunk_ids = [chunk_ids]
            top_chunk_id = str(chunk_ids[0]) if chunk_ids else ""
            chunk_meta = chunk_meta_by_id.get(top_chunk_id, {})
            page_hit = item.get("page_hit")
            if page_hit is None:
                page_hit = 1 if k1.get("page_recall_at_k", 0.0) > 0 else 0
            failure_type = item.get("failure_type") or k1.get("failure_stage")
            if failure_type:
                failure_types.append(str(failure_type))
            failure_stage = FAILURE_STAGE_BY_TYPE.get(str(failure_type))
            query_rows.append(
                {
                    "doc_id": doc_id,
                    "query_id": item.get("query_id"),
                    "question": item.get("question"),
                    "expected_pages": item.get("expected_pages"),
                    "expected_section": item.get("expected_section"),
                    "expected_subsection": item.get("expected_subsection"),
                    "evidence_layout": item.get("evidence_layout"),
                    "acceptable_evidence": item.get("acceptable_evidence"),
                    "filter_hints": item.get("filter_hints"),
                    "page_hit": page_hit,
                    "failure_type": failure_type,
                    "failure_stage": failure_stage,
                    "extracted_answer": item.get("extracted_answer"),
                    "extracted_answer_label": item.get("extracted_answer_label"),
                    "top_chunk_id": top_chunk_id,
                    "top_chunk_text": chunk_text_by_id.get(top_chunk_id, ""),
                    "top_pages": k1.get("retrieved_pages_ranked"),
                    "section_title": chunk_meta.get("section_title"),
                    "subsection_title": chunk_meta.get("subsection_title"),
                    "page_start": chunk_meta.get("page_start"),
                }
            )

    if query_rows:
        qdf = pd.DataFrame(query_rows)
        out_q_csv = resolve_output_path(
            data_root, args.out_queries_csv, "retrieval_queries_report.csv"
        )
        out_q_md = resolve_output_path(
            data_root, args.out_queries_md, "retrieval_queries_report.md"
        )
        out_q_tex = resolve_output_path(
            data_root, args.out_queries_tex, "retrieval_queries_report.tex"
        )
        out_q_csv.parent.mkdir(parents=True, exist_ok=True)
        qdf.to_csv(out_q_csv, index=False)
        out_q_md.write_text(qdf.to_markdown(index=False), encoding="utf-8")
        out_q_tex.write_text(qdf.to_latex(index=False), encoding="utf-8")
        print(f"Wrote: {out_q_csv}")
        print(f"Wrote: {out_q_md}")
        print(f"Wrote: {out_q_tex}")

    if failure_types:
        failure_counts = (
            pd.Series(failure_types, name="failure_type")
            .value_counts(dropna=False)
            .sort_index()
        )
        stage_counts = pd.Series(
            [FAILURE_STAGE_BY_TYPE.get(ft, "unknown") for ft in failure_types],
            name="failure_stage",
        ).value_counts(dropna=False)
        summary = {"total_queries": int(failure_counts.sum())}
        for name, count in stage_counts.items():
            summary[f"count_stage_{name}"] = int(count)
        for name, count in failure_counts.items():
            summary[f"count_{name}"] = int(count)
        out_fail = resolve_output_path(
            data_root, args.out_failure_summary, "retrieval_failure_summary.csv"
        )
        out_fail.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([summary]).to_csv(out_fail, index=False)
        print(f"Wrote: {out_fail}")

    if table_misses:
        out_miss = resolve_output_path(
            data_root, args.out_table_misses, "retrieval_table_misses_k1.csv"
        )
        out_miss.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(table_misses).to_csv(out_miss, index=False)
        print(f"Wrote: {out_miss}")


if __name__ == "__main__":
    main()
