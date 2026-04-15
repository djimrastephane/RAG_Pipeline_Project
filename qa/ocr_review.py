import pandas as pd
from pathlib import Path

DOC_ID = "Grampian-2022-2023"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(
    PROJECT_ROOT / "data_processed" / DOC_ID
)
MAX_PAGES = 10
SNIPPET_CHARS = 500


def load_ocr_pages():
    ocr_csv = OUT_DIR / "ocr_pages.csv"
    if ocr_csv.exists():
        return pd.read_csv(ocr_csv)
    pages_df = pd.read_parquet(OUT_DIR / "pages.parquet")
    return pages_df.loc[pages_df["extractor"] == "ocr", ["page"]].copy()


def print_ocr_snippets():
    ocr_df = load_ocr_pages()
    if ocr_df.empty:
        print("No OCR pages found.")
        return

    pages_df = pd.read_parquet(OUT_DIR / "pages.parquet")
    pages_df = pages_df.set_index("page")

    print("=" * 70)
    print("OCR PAGE REVIEW")
    print("=" * 70)
    print(f"Total OCR pages: {len(ocr_df)}")
    print(f"Showing up to {MAX_PAGES} pages\n")

    for i, row in ocr_df.head(MAX_PAGES).iterrows():
        page_no = int(row["page"])
        if page_no not in pages_df.index:
            continue
        text = str(pages_df.loc[page_no, "clean_text"])
        text_len = len(text)
        snippet = text[:SNIPPET_CHARS]
        print(f"Page {page_no} | text_len={text_len}")
        print("-" * 70)
        print(snippet)
        print("-" * 70 + "\n")


if __name__ == "__main__":
    print_ocr_snippets()
