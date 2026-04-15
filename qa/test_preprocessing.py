import pandas as pd

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
doc_dir = PROJECT_ROOT / "data_processed" / "nhs-england-annual-report-and-accounts-2024-to-2025"
chunks = pd.read_parquet(doc_dir / "chunks.parquet")

ligatures = ["\ufb00", "\ufb01", "\ufb02", "\ufb03", "\ufb04"]
mask = chunks["chunk_text"].astype(str).apply(lambda t: any(ch in t for ch in ligatures))

bad = chunks.loc[mask, ["chunk_id","page_start","chunk_text"]].head(10)
print(bad.to_string(index=False))
