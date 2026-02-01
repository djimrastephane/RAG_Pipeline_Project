# make_charts.py
# Generates the PNG figures referenced by main.tex.
# Run this inside your rag_nhs environment:
#   python make_charts.py

import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path("/Users/djimra/Offline Projects/RAG_NHS")
DOC_ID = "nss_annual_report_and_accounts_2023-24"
DATA_DIR = PROJECT_ROOT / "data_processed" / DOC_ID

CHUNKS_PATH = DATA_DIR / "chunks.parquet"
SECTIONS_PATH = DATA_DIR / "sections.parquet"

FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Recommended for reproducible fonts on macOS
os.environ["MPLBACKEND"] = "Agg"


def save_chunk_length_hist():
    chunks = pd.read_parquet(CHUNKS_PATH)
    chunks["chunk_len"] = chunks["chunk_text"].fillna("").str.split().str.len()

    plt.figure()
    plt.hist(chunks["chunk_len"], bins=20)
    plt.xlabel("Chunk length (words)")
    plt.ylabel("Count")
    plt.title("Chunk length distribution")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "chunk_length_hist.png", dpi=200)
    plt.close()


def save_section_wordcount_hist():
    sections = pd.read_parquet(SECTIONS_PATH)
    sections["word_count"] = sections["section_text"].fillna("").str.split().str.len()

    plt.figure()
    plt.hist(sections["word_count"], bins=20)
    plt.xlabel("Section length (words)")
    plt.ylabel("Count")
    plt.title("Section length distribution")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "section_wordcount_hist.png", dpi=200)
    plt.close()


def save_top_section_titles():
    sections = pd.read_parquet(SECTIONS_PATH)
    vc = sections["section_title"].fillna("Unknown").value_counts().head(10)
    vc = vc.sort_values(ascending=True)

    plt.figure(figsize=(8, 4.5))
    plt.barh(vc.index.tolist(), vc.values.tolist())
    plt.xlabel("Frequency")
    plt.title("Top section titles (frequency)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "top_section_titles.png", dpi=200)
    plt.close()


def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Missing: {CHUNKS_PATH}")
    if not SECTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing: {SECTIONS_PATH}")

    save_chunk_length_hist()
    save_section_wordcount_hist()
    save_top_section_titles()

    print("Saved figures to:", FIG_DIR)


if __name__ == "__main__":
    main()