from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

try:
    import pymupdf as fitz  # PyMuPDF
except Exception as e:
    raise RuntimeError(
        "Failed to import PyMuPDF.\n"
        "Fix: pip uninstall -y fitz frontend && pip install -U pymupdf\n"
    ) from e

import pdfplumber

repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from rag_pdf.extract_page import extract_page_struct_hybrid
from rag_pdf.table_detect import classify_page_content, detect_table_type
from rag_pdf.table_extract import extract_table_cells


NUMERIC_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")
MD_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")


def _load_docling_converter() -> tuple[Optional[Any], Optional[str]]:
    try:
        from docling.document_converter import DocumentConverter  # type: ignore

        return DocumentConverter(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _join_lines(lines_all: list[dict]) -> str:
    return "\n".join(str(ln.get("text", "")).strip() for ln in lines_all if str(ln.get("text", "")).strip())


def _split_md_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _extract_tables_from_markdown(md_text: str) -> list[dict]:
    lines = [ln.rstrip() for ln in (md_text or "").splitlines()]
    blocks: list[list[str]] = []
    cur: list[str] = []
    for ln in lines:
        if "|" in ln and ln.strip():
            cur.append(ln)
        else:
            if cur:
                blocks.append(cur)
                cur = []
    if cur:
        blocks.append(cur)

    tables: list[dict] = []
    for block in blocks:
        sep_idx = next((i for i, ln in enumerate(block) if MD_SEP_RE.match(ln)), None)
        if sep_idx is None or sep_idx == 0:
            continue
        header = _split_md_row(block[sep_idx - 1])
        body = [_split_md_row(ln) for ln in block[sep_idx + 1 :] if ln.strip() and not MD_SEP_RE.match(ln)]
        width = max([len(header)] + [len(r) for r in body] + [1])
        header = (header + [""] * width)[:width]
        norm_body = [((r + [""] * width)[:width]) for r in body]
        tables.append({"header": header, "rows": norm_body})
    return tables


def _table_stats_from_df(df: pd.Optional[DataFrame]) -> dict[str, Any]:
    if df is None or len(df) == 0:
        return {
            "rows": 0,
            "cols": 0,
            "non_empty_cells": 0,
            "numeric_cells": 0,
            "density": 0.0,
            "success": False,
        }
    arr = df.fillna("").astype(str)
    rows, cols = arr.shape
    cells = [str(v).strip() for row in arr.values.tolist() for v in row]
    non_empty = sum(1 for v in cells if v)
    numeric = sum(1 for v in cells if NUMERIC_RE.search(v))
    total = max(rows * cols, 1)
    return {
        "rows": int(rows),
        "cols": int(cols),
        "non_empty_cells": int(non_empty),
        "numeric_cells": int(numeric),
        "density": float(non_empty / total),
        "success": rows > 0 and cols > 0 and non_empty > 0,
    }


def _table_stats_from_docling_markdown(md_text: str) -> dict[str, Any]:
    tables = _extract_tables_from_markdown(md_text)
    if not tables:
        return {
            "rows": 0,
            "cols": 0,
            "non_empty_cells": 0,
            "numeric_cells": 0,
            "density": 0.0,
            "success": False,
        }
    best = max(tables, key=lambda t: len(t["rows"]) * len(t["header"]))
    rows = best["rows"]
    cols = len(best["header"])
    cells = [str(v).strip() for row in rows for v in row]
    non_empty = sum(1 for v in cells if v)
    numeric = sum(1 for v in cells if NUMERIC_RE.search(v))
    total = max(len(rows) * cols, 1)
    return {
        "rows": int(len(rows)),
        "cols": int(cols),
        "non_empty_cells": int(non_empty),
        "numeric_cells": int(numeric),
        "density": float(non_empty / total),
        "success": len(rows) > 0 and cols > 0 and non_empty > 0,
    }


def _extract_docling_page_stats(
    converter: Any,
    src_doc: fitz.Document,
    pdf_path: Path,
    page_no: int,
    tmp_dir: Path,
) -> tuple[dict[str, Any], Optional[str]]:
    page_idx = page_no - 1
    page_pdf = tmp_dir / f"{pdf_path.stem}_p{page_no:04d}.pdf"

    single = fitz.open()
    single.insert_pdf(src_doc, from_page=page_idx, to_page=page_idx)
    single.save(str(page_pdf))
    single.close()

    t0 = time.perf_counter()
    try:
        result = converter.convert(str(page_pdf))
    except Exception as e:
        return (
            {
                "rows": 0,
                "cols": 0,
                "non_empty_cells": 0,
                "numeric_cells": 0,
                "density": 0.0,
                "success": False,
                "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
            },
            f"convert_failed: {type(e).__name__}: {e}",
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    doc_obj = getattr(result, "document", result)
    if hasattr(doc_obj, "export_to_markdown"):
        md = doc_obj.export_to_markdown()
    elif hasattr(result, "export_to_markdown"):
        md = result.export_to_markdown()
    else:
        return (
            {
                "rows": 0,
                "cols": 0,
                "non_empty_cells": 0,
                "numeric_cells": 0,
                "density": 0.0,
                "success": False,
                "elapsed_ms": elapsed_ms,
            },
            "no_markdown_export",
        )

    stats = _table_stats_from_docling_markdown(md)
    stats["elapsed_ms"] = elapsed_ms
    return stats, None


def _detect_table_pages(pdf_path: Path, max_scan_pages: int = 0) -> list[dict[str, Any]]:
    table_pages: list[dict[str, Any]] = []
    with fitz.open(pdf_path) as doc, pdfplumber.open(str(pdf_path)) as pdf_plumber:
        total_pages = len(doc)
        scan_n = min(total_pages, max_scan_pages) if max_scan_pages > 0 else total_pages
        for page_idx in range(scan_n):
            page_struct, _, _ = extract_page_struct_hybrid(doc, pdf_plumber, page_idx, str(pdf_path))
            text = _join_lines(page_struct.get("lines_all", []))
            cls = classify_page_content(text)
            if cls["is_table"]:
                table_pages.append(
                    {
                        "page": page_idx + 1,
                        "table_type": cls.get("table_type") or detect_table_type(text),
                        "confidence": cls.get("confidence"),
                        "text_len": len(text),
                    }
                )
    return table_pages


def benchmark_pdf(
    pdf_path: Path,
    converter: Optional[Any],
    max_table_pages: int,
    max_scan_pages: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    detected_pages = _detect_table_pages(pdf_path, max_scan_pages=max_scan_pages)
    selected_pages = detected_pages[:max_table_pages]
    if not selected_pages:
        return pd.DataFrame(), {"pdf": str(pdf_path), "table_pages_detected": 0}

    rows: list[dict[str, Any]] = []
    with fitz.open(pdf_path) as doc, pdfplumber.open(str(pdf_path)) as pdf_plumber, tempfile.TemporaryDirectory(
        prefix="docling_pages_"
    ) as tmp:
        tmp_dir = Path(tmp)
        for item in selected_pages:
            page_no = int(item["page"])
            table_type = item.get("table_type")

            t0 = time.perf_counter()
            df_current = extract_table_cells(pdf_path, pdf_plumber, page_no, table_type)
            elapsed_current = (time.perf_counter() - t0) * 1000.0
            cur_stats = _table_stats_from_df(df_current)
            rows.append(
                {
                    "pdf": str(pdf_path),
                    "page": page_no,
                    "table_type": table_type,
                    "extractor": "current",
                    "elapsed_ms": elapsed_current,
                    **cur_stats,
                }
            )

            if converter is None:
                rows.append(
                    {
                        "pdf": str(pdf_path),
                        "page": page_no,
                        "table_type": table_type,
                        "extractor": "docling",
                        "elapsed_ms": 0.0,
                        "rows": 0,
                        "cols": 0,
                        "non_empty_cells": 0,
                        "numeric_cells": 0,
                        "density": 0.0,
                        "success": False,
                        "error": "docling_not_available",
                    }
                )
                continue

            docling_stats, err = _extract_docling_page_stats(converter, doc, pdf_path, page_no, tmp_dir)
            row = {
                "pdf": str(pdf_path),
                "page": page_no,
                "table_type": table_type,
                "extractor": "docling",
                **docling_stats,
            }
            if err:
                row["error"] = err
            rows.append(row)

    return pd.DataFrame(rows), {
        "pdf": str(pdf_path),
        "table_pages_detected": len(detected_pages),
        "table_pages_benchmarked": len(selected_pages),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark table extraction: current pipeline vs Docling.")
    parser.add_argument("--pdf-path", type=str, default="", help="Single PDF path to benchmark.")
    parser.add_argument("--pdf-dir", type=str, default="", help="Directory of PDFs to benchmark.")
    parser.add_argument("--glob", type=str, default="*.pdf", help="Glob pattern inside --pdf-dir.")
    parser.add_argument("--max-pdfs", type=int, default=3, help="Max PDFs from --pdf-dir.")
    parser.add_argument("--max-table-pages", type=int, default=12, help="Max detected table pages per PDF.")
    parser.add_argument("--max-scan-pages", type=int, default=0, help="Scan only first N pages (0 = all pages).")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data_processed/benchmarks",
        help="Directory for benchmark outputs.",
    )
    return parser.parse_args()


def _resolve_pdf_paths(args: argparse.Namespace) -> list[Path]:
    pdfs: list[Path] = []
    if args.pdf_path:
        p = Path(args.pdf_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"PDF not found: {p}")
        pdfs.append(p)
    if args.pdf_dir:
        pdir = Path(args.pdf_dir).expanduser().resolve()
        if not pdir.exists():
            raise FileNotFoundError(f"PDF directory not found: {pdir}")
        found = sorted(pdir.glob(args.glob))
        pdfs.extend(found[: args.max_pdfs])
    if not pdfs:
        raise ValueError("Provide either --pdf-path or --pdf-dir.")
    return pdfs


def main() -> None:
    args = parse_args()
    pdf_paths = _resolve_pdf_paths(args)

    converter, docling_error = _load_docling_converter()

    all_rows: list[pd.DataFrame] = []
    run_meta: list[dict[str, Any]] = []
    for pdf_path in pdf_paths:
        page_df, meta = benchmark_pdf(
            pdf_path=pdf_path,
            converter=converter,
            max_table_pages=args.max_table_pages,
            max_scan_pages=args.max_scan_pages,
        )
        run_meta.append(meta)
        if not page_df.empty:
            all_rows.append(page_df)

    if not all_rows:
        raise RuntimeError("No benchmark rows were produced (no table-like pages detected).")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    per_page = pd.concat(all_rows, ignore_index=True)
    summary = (
        per_page.groupby("extractor", dropna=False)
        .agg(
            pages=("page", "count"),
            success_rate=("success", "mean"),
            avg_elapsed_ms=("elapsed_ms", "mean"),
            avg_rows=("rows", "mean"),
            avg_cols=("cols", "mean"),
            avg_numeric_cells=("numeric_cells", "mean"),
        )
        .reset_index()
    )

    per_page_path = out_dir / f"table_extract_benchmark_per_page_{stamp}.csv"
    summary_path = out_dir / f"table_extract_benchmark_summary_{stamp}.csv"
    run_path = out_dir / f"table_extract_benchmark_run_{stamp}.json"
    per_page.to_csv(per_page_path, index=False)
    summary.to_csv(summary_path, index=False)

    run_payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "docling_available": converter is not None,
        "docling_import_error": docling_error,
        "inputs": [str(p) for p in pdf_paths],
        "config": {
            "max_table_pages": args.max_table_pages,
            "max_scan_pages": args.max_scan_pages,
            "max_pdfs": args.max_pdfs,
            "glob": args.glob,
        },
        "pdf_meta": run_meta,
        "outputs": {
            "per_page_csv": str(per_page_path),
            "summary_csv": str(summary_path),
        },
    }
    run_path.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")

    print(f"Docling available: {converter is not None}")
    if docling_error:
        print(f"Docling import error: {docling_error}")
    print(f"Per-page results: {per_page_path}")
    print(f"Summary results:  {summary_path}")
    print(f"Run metadata:     {run_path}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
