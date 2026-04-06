from __future__ import annotations

from typing import Optional

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pymupdf as fitz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full pipeline: preprocess -> build index -> eval -> reports."
    )
    parser.add_argument(
        "--pdf-dir",
        required=True,
        help="Directory containing PDF files.",
    )
    parser.add_argument(
        "--pdf-glob",
        default="*.pdf",
        help="Glob pattern for PDFs inside pdf-dir.",
    )
    parser.add_argument(
        "--out-root",
        default="data_processed",
        help="Output root directory for processed data.",
    )
    parser.add_argument(
        "--model",
        default="models/all-MiniLM-L6-v2",
        help="Embedding model name or local path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess PDFs even if outputs already exist.",
    )
    parser.add_argument(
        "--chunk-size-tokens",
        type=int,
        default=224,
        help="Chunk size in tokens passed to preprocessing.",
    )
    parser.add_argument(
        "--chunk-overlap-tokens",
        type=int,
        default=56,
        help="Chunk overlap in tokens passed to preprocessing.",
    )
    return parser.parse_args()


def run_cmd(cmd: list[str], log_path: Optional[Path] = None) -> None:
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(cmd, check=False, stdout=log_file, stderr=subprocess.STDOUT)
    else:
        result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def _extract_pdf_title(pdf_path: Path) -> str:
    """Best-effort title extraction from first page text for terminal visibility."""
    try:
        with fitz.open(str(pdf_path)) as doc:
            if len(doc) == 0:
                return pdf_path.stem
            text = doc[0].get_text("text")
    except Exception:
        return pdf_path.stem

    lines = [re.sub(r"\s+", " ", ln).strip() for ln in str(text).splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return pdf_path.stem

    org = lines[0]
    annual = next((ln for ln in lines if "ANNUAL REPORT" in ln.upper()), "")
    if annual and annual.upper() != org.upper():
        return f"{org} — {annual}"
    return org


def _read_doc_metrics(out_dir: Path) -> dict[str, Optional[float]]:
    """Read key counts/timings from metrics.json for run summary."""
    out: dict[str, Optional[float]] = {
        "pages_total": None,
        "pages_text": None,
        "pages_table": None,
        "tables_extracted": None,
        "ocr_raw_pages_detected": None,
        "ocr_raw_pages_accepted": None,
        "ocr_short_pages_triggered": None,
        "ocr_short_pages_accepted": None,
        "time_text_extract_total": None,
        "time_ocr_raw_total": None,
        "time_total_wall": None,
    }
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        return out
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(metrics, dict):
            return out
        counts = metrics.get("counts", {})
        if isinstance(counts, dict):
            for key in (
                "pages_total",
                "pages_text",
                "pages_table",
                "tables_extracted",
                "ocr_raw_pages_detected",
                "ocr_raw_pages_accepted",
                "ocr_short_pages_triggered",
                "ocr_short_pages_accepted",
            ):
                raw = counts.get(key)
                if raw is not None:
                    out[key] = int(raw)
        timing = metrics.get("timing", {})
        if isinstance(timing, dict):
            for key in ("time_text_extract_total", "time_ocr_raw_total", "time_total_wall"):
                raw = timing.get(key)
                if raw is not None:
                    out[key] = float(raw)
    except Exception:
        return out
    return out


def main() -> None:
    args = parse_args()
    python_bin = sys.executable
    pdf_dir = Path(args.pdf_dir)
    out_root = Path(args.out_root)
    pdfs = sorted(pdf_dir.glob(args.pdf_glob))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir} with glob {args.pdf_glob}")

    processed_docs: list[str] = []
    doc_run_stats: list[dict[str, object]] = []
    print(f"Found {len(pdfs)} PDF(s) in {pdf_dir}")
    for pdf in pdfs:
        doc_id = pdf.stem
        title = _extract_pdf_title(pdf)
        label = f"{doc_id} | {title}"
        out_dir = out_root / doc_id
        if (out_dir / "pages.parquet").exists() and not args.force:
            print(f"Skipping (already processed): {label}")
            processed_docs.append(doc_id)
            doc_run_stats.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "status": "skipped",
                    "metrics": _read_doc_metrics(out_dir),
                    "preprocess_sec": None,
                }
            )
            continue

        log_path = out_dir / "preprocess.log"
        cmd = [
            python_bin,
            "scripts/preprocess_hybrid.py",
            "--pdf-path",
            str(pdf),
            "--out-root",
            str(out_root),
            "--chunk-size-tokens",
            str(int(args.chunk_size_tokens)),
            "--chunk-overlap-tokens",
            str(int(args.chunk_overlap_tokens)),
            "--require-tiktoken",
        ]
        print(f"Processing: {label}")
        print("Running:", " ".join(cmd))
        t0 = time.perf_counter()
        run_cmd(cmd, log_path=log_path)
        preprocess_sec = time.perf_counter() - t0
        metrics = _read_doc_metrics(out_dir)
        pages = metrics.get("pages_total")
        sec_txt = f"{preprocess_sec:.1f}s"
        pages_txt = str(pages) if pages is not None else "n/a"
        print(f"Completed: {doc_id} | pages={pages_txt} | preprocess_time={sec_txt}")
        processed_docs.append(doc_id)
        doc_run_stats.append(
            {
                    "doc_id": doc_id,
                    "title": title,
                    "status": "processed",
                    "metrics": metrics,
                    "preprocess_sec": preprocess_sec,
                }
            )

    print("Building index...")
    run_cmd(
        [
            python_bin,
            "scripts/build_index.py",
            "--data-dir",
            str(out_root),
            "--model",
            str(args.model),
        ]
    )

    print("Running retrieval eval...")
    for doc_id in processed_docs:
        doc_dir = out_root / doc_id
        eval_set = doc_dir / "eval_set.json"
        if not eval_set.exists():
            print(f"Skipping eval (missing eval_set.json): {doc_dir}")
            continue
        run_cmd(
            [
                python_bin,
                "scripts/retrieval_eval_hybrid.py",
                "--data-dir",
                str(doc_dir),
                "--model",
                str(args.model),
            ]
        )

    print("Building reports...")
    run_cmd(
        [
            python_bin,
            "scripts/report_retrieval_metrics.py",
            "--data-root",
            str(out_root),
            "--docs",
            ",".join(processed_docs),
        ]
    )

    print("\nPer-document preprocessing summary (pages vs time):")
    for row in doc_run_stats:
        doc_id = str(row["doc_id"])
        status = str(row["status"])
        metrics = row.get("metrics") or {}
        pages_txt = str(metrics.get("pages_total")) if metrics.get("pages_total") is not None else "n/a"
        text_pages_txt = str(metrics.get("pages_text")) if metrics.get("pages_text") is not None else "n/a"
        table_pages_txt = str(metrics.get("pages_table")) if metrics.get("pages_table") is not None else "n/a"
        tables_txt = str(metrics.get("tables_extracted")) if metrics.get("tables_extracted") is not None else "n/a"
        ocr_raw_txt = (
            f"{metrics.get('ocr_raw_pages_accepted')}/{metrics.get('ocr_raw_pages_detected')}"
            if metrics.get("ocr_raw_pages_detected") is not None
            else "n/a"
        )
        ocr_short_txt = (
            f"{metrics.get('ocr_short_pages_accepted')}/{metrics.get('ocr_short_pages_triggered')}"
            if metrics.get("ocr_short_pages_triggered") is not None
            else "n/a"
        )
        pre_sec = row.get("preprocess_sec")
        sec_txt = f"{float(pre_sec):.1f}s" if pre_sec is not None else "-"
        ocr_time = metrics.get("time_ocr_raw_total")
        ocr_time_txt = f"{float(ocr_time):.1f}s" if ocr_time is not None else "n/a"
        print(
            f"- {doc_id}: status={status}, pages={pages_txt} (text={text_pages_txt}, table={table_pages_txt}), "
            f"tables={tables_txt}, ocr_raw={ocr_raw_txt}, ocr_short={ocr_short_txt}, "
            f"ocr_time={ocr_time_txt}, preprocess_time={sec_txt}"
        )

    print("Full pipeline complete.")


if __name__ == "__main__":
    main()
