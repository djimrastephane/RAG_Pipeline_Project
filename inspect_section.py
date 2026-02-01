import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path("/Users/djimra/Offline Projects/RAG_NHS")
DOC_FOLDER = PROJECT_ROOT / "data_processed" / "nss_annual_report_and_accounts_2023-24"

sections_path = DOC_FOLDER / "sections.parquet"
chunks_path = DOC_FOLDER / "chunks.parquet"

sections = pd.read_parquet(sections_path)

print("\nFIRST 15 SECTIONS")
print(sections[["part", "section_title", "page_start", "page_end"]].head(15).to_string(index=False))

sections["word_count"] = sections["section_text"].str.split().str.len()
print("\nSECTION WORD COUNT SUMMARY")
print(sections["word_count"].describe().to_string())

chunks = pd.read_parquet(chunks_path)
chunks["chunk_len"] = chunks["chunk_text"].str.split().str.len()

print("\nCHUNK LENGTH SUMMARY")
print(chunks["chunk_len"].describe().to_string())