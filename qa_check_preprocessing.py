from pathlib import Path
import json

base = Path(
    "/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/data_processed"
)

def final_norm_flag(data: dict) -> bool:
    # New schema location (what your metrics file shows)
    v = (
        data.get("embedding", {})
            .get("preprocess_trace", {})
            .get("final_text_normalization", None)
    )
    if v is None:
        # Backward compatible with older schema if it ever existed
        v = data.get("params", {}).get("final_text_normalization", None)

    # Decide what "done" means:
    # - if dict exists (even empty), treat as done
    # - if boolean exists, use it
    # - else False
    if isinstance(v, dict):
        return True
    if isinstance(v, bool):
        return v
    return False

for d in sorted(base.iterdir()):
    if not d.is_dir():
        continue
    m = d / "metrics.json"
    if not m.exists():
        print(d.name, "MISSING metrics.json")
        continue

    data = json.loads(m.read_text())
    print(
        d.name,
        "schema:",
        data.get("schema_version"),
        "final_norm:",
        final_norm_flag(data),
    )