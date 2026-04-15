import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
doc_dir = PROJECT_ROOT / "data_processed" / "nhs-england-annual-report-and-accounts-2024-to-2025"
m = json.loads((doc_dir / "metrics.json").read_text(encoding="utf-8"))
print("schema_version:", m.get("schema_version"))
print("final_text_normalization:", m.get("params", {}).get("final_text_normalization"))
