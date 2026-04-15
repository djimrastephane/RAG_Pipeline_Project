from __future__ import annotations

from typing import Optional

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick pipeline demo: sections, chunks, tables, retrieval."
    )
    parser.add_argument(
        "--data-dir",
        default="data_processed/Grampian-2022-2023",
        help="Processed document directory.",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=10,
        help="Page to inspect for section/subsection.",
    )
    parser.add_argument(
        "--query-id",
        default="Q_DEF_2023_02",
        help="Query ID to inspect in retrieval results (comma-separated allowed).",
    )
    parser.add_argument(
        "--show-retrieval",
        action="store_true",
        help="Show full retrieval details for the query.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    sections_path = data_dir / "sections.parquet"
    chunks_path = data_dir / "chunks.parquet"
    tables_path = data_dir / "tables_structured.parquet"
    metrics_path = data_dir / "retrieval_metrics.json"
    results_path = data_dir / "retrieval_results.json"

    print(f"Data dir: {data_dir}")

    # Section/subsection for a page
    if sections_path.exists():
        sections = pd.read_parquet(sections_path)
        match = sections[
            (sections["page_start"] <= args.page) & (sections["page_end"] >= args.page)
        ]
        if not match.empty:
            row = match.iloc[-1]
            print("\nSection lookup")
            print(
                f"  page={args.page} section={row.get('section_title')} "
                f"subsection={row.get('subsection_title')}"
            )
    else:
        print(f"\nMissing: {sections_path}")

    # Show a few chunks with section metadata
    if chunks_path.exists():
        chunks = pd.read_parquet(chunks_path)
        print("\nSample chunks")
        cols = ["chunk_id", "page_start", "section_title", "subsection_title"]
        cols = [c for c in cols if c in chunks.columns]
        print(chunks[cols].head(3).to_string(index=False))
    else:
        print(f"\nMissing: {chunks_path}")

    # Show a few tables
    if tables_path.exists():
        tables = pd.read_parquet(tables_path)
        if len(tables) > 0:
            print("\nSample tables")
            cols = ["table_id", "page", "table_type", "rows", "cols"]
            cols = [c for c in cols if c in tables.columns]
            print(tables[cols].head(5).to_string(index=False))
    else:
        print(f"\nMissing: {tables_path}")

    # Retrieval metrics
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        print("\nRetrieval metrics")
        for k in ("1", "3"):
            m = metrics.get("metrics_by_k", {}).get(k, {})
            if m:
                print(f"  k={k} hit_rate={m.get('page_hit_rate_at_k'):.3f} "
                      f"mrr={m.get('mean_page_mrr_at_k'):.3f} "
                      f"precision={m.get('mean_page_precision_at_k'):.3f}")
    else:
        print(f"\nMissing: {metrics_path}")

    # Retrieval result for one or more queries
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))
        query_ids = [q.strip() for q in str(args.query_id).split(",") if q.strip()]
        for query_id in query_ids:
            query = None
            for r in results.get("results", []):
                if r.get("query_id") == query_id:
                    query = r
                    break
            if query:
                print("\nRetrieval example")
                print(f"  query_id={query_id}")
                print(f"  question={query.get('question')}")
                chunks = None
                if chunks_path.exists():
                    chunks = pd.read_parquet(chunks_path)
                if args.show_retrieval:
                    per_k = query.get("per_k", {})
                    for k, info in per_k.items():
                        print(f"  k={k}")
                        print(f"    chunks={info.get('retrieved_chunk_ids')}")
                        print(f"    pages={info.get('retrieved_pages_ranked')}")
                        print(f"    scores={info.get('retrieved_scores')}")
                        print(f"    page_recall={info.get('page_recall_at_k')}")
                        print(f"    chunk_hit={info.get('chunk_hit_at_k')}")
                else:
                    per_k = query.get("per_k", {})
                    k1 = per_k.get("1", {})
                    print(f"  top_chunk={k1.get('retrieved_chunk_ids')}")
                    print(f"  top_pages={k1.get('retrieved_pages_ranked')}")

                    if chunks is not None:
                        top_ids = k1.get("retrieved_chunk_ids") or []
                        if isinstance(top_ids, str):
                            top_ids = [top_ids]
                        if top_ids:
                            cid = top_ids[0]
                            row = chunks[chunks["chunk_id"].astype(str) == cid]
                            if row.empty and "chunk_id_global" in chunks.columns:
                                row = chunks[chunks["chunk_id_global"].astype(str) == cid]
                            if not row.empty:
                                r0 = row.iloc[0]
                                section = r0.get("section_title")
                                subsection = r0.get("subsection_title")
                                text = str(r0.get("chunk_text") or "")
                                question = str(query.get("question") or "").lower()
                                extracted = None
                                extracted_label = None

                                def _extract_quarter_value(
                                    label: str,
                                ) -> tuple[Optional[str], Optional[str]]:
                                    m = re.search(
                                        rf"{label}\s+([\d-]+)\s+([\d-]+)\s+([\d-]+)\s+([\d-]+)",
                                        text,
                                        flags=re.IGNORECASE,
                                    )
                                    if not m:
                                        return None, None
                                    vals = [m.group(i) for i in range(1, 5)]
                                    if "q1" in question:
                                        return (
                                            (None if vals[0] == "-" else vals[0]),
                                            "Q1",
                                        )
                                    if "q2" in question:
                                        return (
                                            (None if vals[1] == "-" else vals[1]),
                                            "Q2",
                                        )
                                    if "q3" in question:
                                        return (
                                            (None if vals[2] == "-" else vals[2]),
                                            "Q3",
                                        )
                                    if "q4" in question:
                                        return (
                                            (None if vals[3] == "-" else vals[3]),
                                            "Q4",
                                        )
                                    return None, None

                                def _lookup_chunk_text(chunk_id: str) -> Optional[str]:
                                    row = chunks[chunks["chunk_id"].astype(str) == chunk_id]
                                    if row.empty and "chunk_id_global" in chunks.columns:
                                        row = chunks[chunks["chunk_id_global"].astype(str) == chunk_id]
                                    if row.empty:
                                        return None
                                    return str(row.iloc[0].get("chunk_text") or "")

                                if "significant" in question and "delay" in question:
                                    extracted, quarter = _extract_quarter_value(
                                        "Significant Delay"
                                    )
                                    if extracted:
                                        extracted_label = (
                                            f"Significant Delay ({quarter})"
                                            if quarter
                                            else "Significant Delay"
                                        )
                                elif "on track" in question:
                                    extracted, quarter = _extract_quarter_value("On Track")
                                    if extracted:
                                        extracted_label = (
                                            f"On Track ({quarter})"
                                            if quarter
                                            else "On Track"
                                        )
                                elif "board committee" in question and "strategic risk register" in question:
                                    m = re.search(
                                        r"the ([A-Za-z &-]+ committee) have delegated responsibility",
                                        text,
                                        flags=re.IGNORECASE,
                                    )
                                    if not m:
                                        m = re.search(
                                            r"the ([A-Za-z &-]+ committee) has delegated responsibility",
                                            text,
                                            flags=re.IGNORECASE,
                                        )
                                    if m:
                                        extracted = m.group(1).strip().title()
                                        extracted_label = "Delegated Committee"
                                elif "endorse" in question and "risk appetite" in question and "strategic risk profile" in question:
                                    ra_date = None
                                    srp_date = None
                                    for sent in re.split(r"(?<=[.!?])\s+", text):
                                        low = sent.lower()
                                        if "endorsed" in low and "risk appetite statement" in low:
                                            m = re.search(
                                                r"endorsed.*?on(?: the)?\s+(\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]+\s+\d{4})",
                                                sent,
                                                flags=re.IGNORECASE,
                                            )
                                            if m:
                                                ra_date = m.group(1)
                                        if "endorsed" in low and "strategic risk profile" in low:
                                            m = re.search(
                                                r"endorsed.*?strategic risk profile.*?in\s+([A-Z][a-z]+\s+\d{4})",
                                                sent,
                                                flags=re.IGNORECASE,
                                            )
                                            if m:
                                                srp_date = m.group(1)
                                    ra = ra_date or ""
                                    srp = srp_date or ""
                                    if not srp:
                                        candidate_ids = (
                                            per_k.get("5", {}).get("retrieved_chunk_ids")
                                            or per_k.get("3", {}).get("retrieved_chunk_ids")
                                            or []
                                        )
                                        for cid2 in candidate_ids:
                                            if cid2 == cid:
                                                continue
                                            other_text = _lookup_chunk_text(str(cid2))
                                            if not other_text:
                                                continue
                                            for sent in re.split(r"(?<=[.!?])\s+", other_text):
                                                low = sent.lower()
                                                if "endorsed" in low and "strategic risk profile" in low:
                                                    m = re.search(
                                                        r"endorsed.*?strategic risk profile.*?in\s+([A-Z][a-z]+\s+\d{4})",
                                                        sent,
                                                        flags=re.IGNORECASE,
                                                    )
                                                    if m:
                                                        srp = m.group(1)
                                                        break
                                            if srp:
                                                break
                                    if ra or srp:
                                        parts = []
                                        if ra:
                                            parts.append(f"Risk Appetite: {ra}")
                                        if srp:
                                            parts.append(f"Strategic Risk Profile: {srp}")
                                        extracted = "; ".join(parts)
                                        extracted_label = "Board Endorsements"
                                elif "endorse" in question and "risk appetite" in question:
                                    for sent in re.split(r"(?<=[.!?])\s+", text):
                                        if "endorsed" in sent and "risk appetite statement" in sent.lower():
                                            m = re.search(
                                                r"endorsed.*?on(?: the)?\s+(\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]+\s+\d{4})",
                                                sent,
                                                flags=re.IGNORECASE,
                                            )
                                            if m:
                                                extracted = m.group(1)
                                                extracted_label = "Risk Appetite Endorsement"
                                                break
                                elif "endorse" in question and "strategic risk profile" in question:
                                    for sent in re.split(r"(?<=[.!?])\s+", text):
                                        if "endorsed" in sent and "strategic risk profile" in sent.lower():
                                            m = re.search(
                                                r"endorsed.*?strategic risk profile.*?in\s+([A-Z][a-z]+\s+\d{4})",
                                                sent,
                                                flags=re.IGNORECASE,
                                            )
                                            if m:
                                                extracted = m.group(1)
                                                extracted_label = "Strategic Risk Profile Endorsement"
                                                break
                                elif (
                                    "significant issue" in question
                                    and "accountable officer" in question
                                ):
                                    for sent in re.split(r"(?<=[.!?])\s+", text):
                                        if "funding arrangement" in sent.lower():
                                            extracted = sent.strip()
                                            extracted_label = "Significant Issue"
                                            break
                                elif "proportion" in question and "complete" in question:
                                    m = re.search(
                                        r"(\d+(?:\.\d+)?)%[^\n]{0,80}complete",
                                        text,
                                        flags=re.IGNORECASE,
                                    )
                                    if not m:
                                        m = re.search(
                                            r"complete[^\n]{0,80}(\d+(?:\.\d+)?)%",
                                            text,
                                            flags=re.IGNORECASE,
                                        )
                                    if m:
                                        extracted = f"{m.group(1)}%"
                                        extracted_label = "Complete (%)"

                                nums = []
                                for m in re.finditer(r"\d+(?:\.\d+)?%?", text):
                                    token = m.group(0)
                                    if (
                                        len(token) == 4
                                        and token.isdigit()
                                        and 1900 <= int(token) <= 2100
                                    ):
                                        continue
                                    nums.append(token)
                                nums = list(dict.fromkeys(nums))
                                print(f"  section={section} subsection={subsection}")
                                if not extracted:
                                    snippet = re.split(r"(?<=[.!?])\s+", text.strip())[0]
                                    snippet = snippet[:200].strip()
                                    extracted = snippet if snippet else "(no extraction rule matched)"
                                    extracted_label = "Snippet"
                                if extracted_label:
                                    print(
                                        f"  extracted_answer={extracted_label}: "
                                        f"{extracted}"
                                    )
                                else:
                                    print(f"  extracted_answer={extracted}")
                                print(f"  numeric_candidates={nums[:25]}")
            else:
                print(f"\nQuery not found: {query_id}")
    else:
        print(f"\nMissing: {results_path}")


if __name__ == "__main__":
    main()
