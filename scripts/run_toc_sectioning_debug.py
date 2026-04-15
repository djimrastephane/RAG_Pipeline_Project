from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pdfplumber
import pymupdf as fitz

repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from rag_pdf.boilerplate import remove_repeated_header_footer_lines, strip_by_coordinates
from rag_pdf.extract_page import extract_page_struct_hybrid
from rag_pdf.headings import (
    is_part_label,
    is_section_anchor_line,
    looks_like_heading_text_only,
    looks_like_lettered_subsection,
    select_heading_candidates,
)
from rag_pdf.sections import build_sections_from_pages, find_section_for_page
from rag_pdf.text_normalize import normalize_line, normalize_page_text, now_utc_iso


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ToC + sectioning debug for one PDF.")
    p.add_argument("input_pdf")
    p.add_argument("output_dir")
    return p.parse_args()


def _extract_top_lines(lines_all: list[dict], k: int = 10) -> list[dict]:
    if not lines_all:
        return []
    sorted_lines = sorted(lines_all, key=lambda l: (float(l.get("y0", 0.0)), float(l.get("x0", 0.0))))
    top: list[dict] = []
    for ln in sorted_lines[:k]:
        txt = str(ln.get("text", "")).strip()
        if not txt:
            continue
        top.append({"text": txt, "y0": float(ln.get("y0", 0.0)), "y1": float(ln.get("y1", 0.0))})
    return top


def _is_common_header_footer_line(text: str, common_header: set[str], common_footer: set[str]) -> bool:
    return (text in common_header or text in common_footer) and not is_section_anchor_line(text)


def main() -> None:
    args = parse_args()
    pdf_path = Path(args.input_pdf).resolve()
    out_root = Path(args.output_dir).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)
    doc_id = pdf_path.stem
    out_dir = out_root / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pages_text_lines: dict[int, list[str]] = {}
    page_heading_candidates: dict[int, list[str]] = {}
    page_top_lines: dict[int, list[dict]] = {}

    with fitz.open(str(pdf_path)) as doc, pdfplumber.open(str(pdf_path)) as pl:
        for i in range(doc.page_count):
            page_no = i + 1
            s, _, _ = extract_page_struct_hybrid(doc, pl, i, pdf_path=str(pdf_path))
            kept, _, _ = strip_by_coordinates(
                s["lines_all"],
                page_height=s["page_height"],
                page_width=s["page_width"],
                rotation=s["rotation"],
            )
            pages_text_lines[page_no] = kept
            page_heading_candidates[page_no] = select_heading_candidates(s["lines_all"], s["p95_font"])
            page_top_lines[page_no] = _extract_top_lines(s["lines_all"], k=10)

    pages_text_lines2, common_header, common_footer = remove_repeated_header_footer_lines(pages_text_lines)
    for page_no, lines in page_top_lines.items():
        filtered: list[dict] = []
        for ln in lines:
            txt = normalize_line(str(ln.get("text", "")))
            if not txt:
                continue
            if _is_common_header_footer_line(txt, common_header, common_footer):
                continue
            filtered.append(ln)
        page_top_lines[page_no] = filtered

    pages_records = []
    run_utc = now_utc_iso()
    for page_no in sorted(pages_text_lines2.keys()):
        clean_text = normalize_page_text("\n".join(pages_text_lines2.get(page_no, [])).strip())
        cleaned_set = {normalize_line(x) for x in pages_text_lines2.get(page_no, []) if normalize_line(x)}
        filtered_headings: list[str] = []
        for cand in page_heading_candidates.get(page_no, []):
            norm = normalize_line(str(cand))
            if not norm:
                continue
            if _is_common_header_footer_line(norm, common_header, common_footer):
                continue
            if cleaned_set and norm not in cleaned_set:
                continue
            filtered_headings.append(cand)
        if not filtered_headings:
            for line in pages_text_lines2.get(page_no, [])[:25]:
                if (
                    looks_like_heading_text_only(line) or looks_like_lettered_subsection(line)
                ) and not is_part_label(line):
                    filtered_headings = [line]
                    break

        pages_records.append(
            {
                "doc_id": doc_id,
                "report_year": None,
                "period_end_date": None,
                "report_year_source": "unknown",
                "run_date_utc": run_utc,
                "page": page_no,
                "clean_text": clean_text,
                "heading_candidates": filtered_headings,
                "top_lines": page_top_lines.get(page_no, []),
            }
        )
    pages_df = pd.DataFrame(pages_records)
    sections_df, diag = build_sections_from_pages(pages_df, return_diagnostics=True)

    # Write outputs
    pages_df.to_parquet(out_dir / "pages.parquet", index=False)
    sections_df.to_parquet(out_dir / "sections.parquet", index=False)
    sections_df.to_csv(out_dir / "sections.csv", index=False)
    toc_df = diag.get("toc_df")
    if isinstance(toc_df, pd.DataFrame):
        toc_df.to_parquet(out_dir / "toc.parquet", index=False)
        toc_df.to_csv(out_dir / "toc.csv", index=False)
    rejected_df = diag.get("rejected_subsections_df")
    if isinstance(rejected_df, pd.DataFrame):
        rejected_df.to_csv(out_dir / "subsection_rejected_candidates.csv", index=False)

    print("ToC detected:", bool(diag.get("toc_detected", False)))
    print("ToC pages:", int(diag.get("toc_pages_count", 0)))
    print("ToC items:", int(diag.get("toc_items_count", 0)))
    print("ToC offset:", int(diag.get("toc_offset", 0)))
    print("ToC offset confidence:", float(diag.get("toc_offset_confidence", 0.0)))
    print("ToC offset support count:", int(diag.get("toc_offset_support_count", 0)))
    print("ToC coverage pct:", float(diag.get("toc_coverage_pct", 0.0)))
    print("ToC override rate:", float(diag.get("toc_override_rate", 0.0)))
    print("Subsection rejects:", int(diag.get("subsection_reject_count", 0)))

    if isinstance(toc_df, pd.DataFrame) and len(toc_df) > 0:
        print("\nFirst 30 ToC items:")
        print(toc_df.head(30).to_string(index=False))

    # boundary samples
    if isinstance(toc_df, pd.DataFrame) and len(toc_df) > 1:
        sample_pages: list[int] = []
        for p in toc_df["start_pdf_page"].dropna().astype(int).tolist()[:5]:
            for q in (p - 1, p, p + 1):
                if q >= 1:
                    sample_pages.append(q)
        sample_pages = sorted(set(sample_pages))[:10]
    else:
        sample_pages = sorted(pages_df["page"].tolist())[:10]

    print("\nSample page labels around boundaries:")
    for p in sample_pages:
        part, sec, sub = find_section_for_page(sections_df, p)
        print(f"page={p:>3} | part={part} | section={sec} | subsection={sub}")

    print("\nSaved:")
    print("-", out_dir / "toc.parquet")
    print("-", out_dir / "sections.parquet")


if __name__ == "__main__":
    main()
