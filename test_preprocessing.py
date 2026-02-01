import pandas as pd

doc_dir = "/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/data_processed/nhs-england-annual-report-and-accounts-2024-to-2025"
chunks = pd.read_parquet(f"{doc_dir}/chunks.parquet")

ligs = ["\ufb00","\ufb01","\ufb02","\ufb03","\ufb04"]
mask = chunks["chunk_text"].astype(str).apply(lambda t: any(ch in t for ch in ligs))

bad = chunks.loc[mask, ["chunk_id","page_start","chunk_text"]].head(10)
print(bad.to_string(index=False))