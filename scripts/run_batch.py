import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIG_CANDIDATES = (
    Path("configs/batch.json"),
    Path("config/batch.json"),
)


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_pdfs(cfg: dict) -> list[Path]:
    if "pdfs" in cfg:
        return [Path(p) for p in cfg["pdfs"]]
    pdf_dir = Path(cfg["pdf_dir"])
    pdf_glob = cfg.get("pdf_glob", "*.pdf")
    return sorted(pdf_dir.glob(pdf_glob))


def default_config_path() -> str:
    for candidate in DEFAULT_CONFIG_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return str(DEFAULT_CONFIG_CANDIDATES[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch preprocess PDFs.")
    parser.add_argument(
        "--config",
        default=default_config_path(),
        help="Path to batch config JSON.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess PDFs even if outputs already exist.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    out_root = Path(cfg["out_root"])
    pdfs = iter_pdfs(cfg)

    if not pdfs:
        print("No PDFs found.")
        return

    print(f"Found {len(pdfs)} PDFs.")
    summary_rows = []
    summary_path = out_root / "batch_summary.csv"

    for pdf in pdfs:
        doc_id = pdf.stem
        out_dir = out_root / doc_id
        if (out_dir / "pages.parquet").exists() and not args.force:
            print(f"Skipping (already processed): {pdf}")
            summary_rows.append({
                "doc_id": doc_id,
                "pdf_path": str(pdf),
                "status": "skipped",
            })
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "preprocess.log"
        cmd = [
            sys.executable,
            "scripts/preprocess_hybrid.py",
            "--pdf-path",
            str(pdf),
            "--out-root",
            str(out_root),
            "--chunk-size-tokens",
            str(int(cfg.get("chunk_size_tokens", 224))),
            "--chunk-overlap-tokens",
            str(int(cfg.get("chunk_overlap_tokens", 56))),
            "--require-tiktoken",
        ]
        print("\nRunning:", " ".join(cmd))
        with open(log_path, "w", encoding="utf-8") as log_file:
            result = subprocess.run(cmd, check=False, stdout=log_file, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            print(f"Failed: {pdf}")
            summary_rows.append({
                "doc_id": doc_id,
                "pdf_path": str(pdf),
                "status": "failed",
            })
            break

        metrics_path = out_dir / "metrics.json"
        if metrics_path.exists():
            metrics = load_config(metrics_path)
            counts = metrics.get("counts", {})
            timing = metrics.get("timing", {})
            derived = metrics.get("derived", {})
            summary_rows.append({
                "doc_id": doc_id,
                "pdf_path": str(pdf),
                "status": "processed",
                "pages_total": counts.get("pages_total"),
                "sections_detected": counts.get("sections_detected"),
                "chunks_total": counts.get("chunks_total"),
                "tables_extracted": counts.get("tables_extracted"),
                "ocr_raw_pages_detected": counts.get("ocr_raw_pages_detected"),
                "ocr_raw_pages_accepted": counts.get("ocr_raw_pages_accepted"),
                "ocr_short_pages_triggered": counts.get("ocr_short_pages_triggered"),
                "ocr_short_pages_accepted": counts.get("ocr_short_pages_accepted"),
                "ocr_raw_acceptance_rate": derived.get("ocr_raw_acceptance_rate"),
                "ocr_short_acceptance_rate": derived.get("ocr_short_acceptance_rate"),
                "chunks_per_page": derived.get("chunks_per_page"),
                "tables_per_100_pages": derived.get("tables_per_100_pages"),
                "time_text_extract_total": timing.get("time_text_extract_total"),
                "time_coord_strip_total": timing.get("time_coord_strip_total"),
                "time_ocr_raw_total": timing.get("time_ocr_raw_total"),
                "time_total_wall": timing.get("time_total_wall"),
                "time_unit": timing.get("time_unit"),
                "run_utc": metrics.get("run_utc"),
                "git_commit_short": metrics.get("git_commit_short"),
                "embedding_model": metrics.get("embedding_model"),
            })
        else:
            summary_rows.append({
                "doc_id": doc_id,
                "pdf_path": str(pdf),
                "status": "processed",
            })

    if summary_rows:
        fieldnames = sorted({k for row in summary_rows for k in row.keys()})
        with open(summary_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nWrote batch summary: {summary_path}")


if __name__ == "__main__":
    main()
