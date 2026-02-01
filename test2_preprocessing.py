import json
from pathlib import Path

doc_dir = Path("/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/data_processed/nhs-england-annual-report-and-accounts-2024-to-2025")
m = json.loads((doc_dir/"metrics.json").read_text(encoding="utf-8"))
print("schema_version:", m.get("schema_version"))
print("final_text_normalization:", m.get("params", {}).get("final_text_normalization"))