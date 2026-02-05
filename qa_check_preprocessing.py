from pathlib import Path
import json

base = Path(
    "/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/data_processed"
)

def final_norm_flag(data: dict) -> bool:
    v = (
        data.get("embedding", {})
            .get("preprocess_trace", {})
            .get("final_text_normalization", None)
    )
    if v is None:
        v = data.get("params", {}).get("final_text_normalization", None)

    if isinstance(v, dict):
        return True
    if isinstance(v, bool):
        return v
    return False

def embedding_qa_assertions(data: dict, run_name: str) -> None:
    emb = data.get("embedding", {})
    summary = emb.get("embedding_summary", {})

    chunks_embedded = emb.get("chunks_embedded")
    embedding_dim = emb.get("embedding_dim")
    normalised_for_cosine = emb.get("normalised_for_cosine")
    shape = summary.get("shape")

    # Hard failures, these invalidate retrieval if broken
    assert chunks_embedded is not None, f"{run_name}: chunks_embedded missing"
    assert embedding_dim is not None, f"{run_name}: embedding_dim missing"
    assert shape is not None, f"{run_name}: embedding_summary.shape missing"

    assert chunks_embedded == shape[0], (
        f"{run_name}: chunk count mismatch "
        f"(chunks_embedded={chunks_embedded}, shape[0]={shape[0]})"
    )

    assert embedding_dim == shape[1], (
        f"{run_name}: embedding dim mismatch "
        f"(embedding_dim={embedding_dim}, shape[1]={shape[1]})"
    )

    assert normalised_for_cosine is True, (
        f"{run_name}: embeddings not normalised for cosine similarity"
    )

for d in sorted(base.iterdir()):
    if not d.is_dir():
        continue

    m = d / "metrics.json"
    if not m.exists():
        print(d.name, "MISSING metrics.json")
        continue

    data = json.loads(m.read_text())

    try:
        embedding_qa_assertions(data, d.name)
        emb_status = "OK"
    except AssertionError as e:
        emb_status = f"FAIL ({e})"

    print(
        d.name,
        "schema:",
        data.get("schema_version"),
        "final_norm:",
        final_norm_flag(data),
        "embedding_QA:",
        emb_status,
    )