from __future__ import annotations

from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
base = PROJECT_ROOT / "data_processed"


def final_norm_flag(metrics_data: dict) -> bool:
    """
    Determine whether final text normalization was applied.

    Supports both legacy location:
      params.final_text_normalization

    And newer location:
      embedding.preprocess_trace.final_text_normalization
    """
    v = (
        metrics_data.get("embedding", {})
        .get("preprocess_trace", {})
        .get("final_text_normalization", None)
    )
    if v is None:
        v = metrics_data.get("params", {}).get("final_text_normalization", None)

    if isinstance(v, dict):
        return True
    if isinstance(v, bool):
        return v
    return False


def _parse_schema_version(v) -> tuple[int, int]:
    """
    Parse a schema version string like '2.4' into a comparable tuple (2, 4).
    Returns (0, 0) when parsing fails.
    """
    if v is None:
        return 0, 0
    if isinstance(v, (int, float)):
        s = str(v)
    else:
        s = str(v).strip()
    m = s.split(".")
    try:
        major = int(m[0])
        minor = int(m[1]) if len(m) > 1 else 0
        return major, minor
    except Exception:
        return 0, 0


def preprocessing_param_assertions(metrics_data: dict, run_name: str) -> None:
    """
    Validate preprocessing parameters recorded in metrics.json.

    This stays lightweight and focuses on:
    - required keys for the current schema
    - basic sanity ranges for strip fractions and chunk settings

    It does not inspect pages.parquet content, it only checks metadata.
    """
    params = metrics_data.get("params", {})
    sv = _parse_schema_version(metrics_data.get("schema_version"))

    # Always expected in your pipeline
    required = [
        "chunk_size_tokens",
        "chunk_overlap_tokens",
        "top_strip_frac",
        "bottom_strip_frac",
        "header_footer_repeat_frac",
        "min_chunk_words",
        "primary_extractor",
        "fallback_min_chars",
        "fallback_on_bad_text",
        "fallback_on_exception",
    ]
    for k in required:
        assert k in params, f"{run_name}: params.{k} missing"

    # New for rotated/wide-page logic in schema 2.4+
    if sv >= (2, 4):
        assert "left_strip_frac" in params, f"{run_name}: params.left_strip_frac missing"
        assert "right_strip_frac" in params, f"{run_name}: params.right_strip_frac missing"

    # Sanity ranges
    def _frac_ok(x) -> bool:
        try:
            f = float(x)
            return 0.0 <= f <= 0.30
        except Exception:
            return False

    assert _frac_ok(params["top_strip_frac"]), f"{run_name}: top_strip_frac out of range"
    assert _frac_ok(params["bottom_strip_frac"]), f"{run_name}: bottom_strip_frac out of range"
    if sv >= (2, 4):
        assert _frac_ok(params["left_strip_frac"]), f"{run_name}: left_strip_frac out of range"
        assert _frac_ok(params["right_strip_frac"]), f"{run_name}: right_strip_frac out of range"

    # Chunking sanity
    cs = params["chunk_size_tokens"]
    co = params["chunk_overlap_tokens"]
    try:
        cs_i = int(cs)
        co_i = int(co)
    except Exception:
        raise AssertionError(f"{run_name}: chunk_size_tokens/chunk_overlap_tokens not int-like")

    assert cs_i > 0, f"{run_name}: chunk_size_tokens must be > 0"
    assert 0 <= co_i < cs_i, f"{run_name}: chunk_overlap_tokens must be >=0 and < chunk_size_tokens"

    # Repeat fraction sanity
    try:
        r = float(params["header_footer_repeat_frac"])
    except Exception:
        raise AssertionError(f"{run_name}: header_footer_repeat_frac not numeric")
    assert 0.0 <= r <= 1.0, f"{run_name}: header_footer_repeat_frac out of range"


def embedding_qa_assertions(metrics_data: dict, run_name: str) -> None:
    """
    Validate embedding stage metadata when present.

    Some runs may not write 'embedding' yet. In that case, this function
    should be skipped by the caller.
    """
    emb = metrics_data.get("embedding", {})
    summary = emb.get("embedding_summary", {})

    chunks_embedded = emb.get("chunks_embedded")
    embedding_dim = emb.get("embedding_dim")
    normalised_for_cosine = emb.get("normalised_for_cosine")
    shape = summary.get("shape")

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


for run_dir in sorted(base.iterdir()):
    if not run_dir.is_dir():
        continue

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        print(run_dir.name, "MISSING metrics.json")
        continue

    metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))

    # Preprocessing QA (based on metrics.json params)
    try:
        preprocessing_param_assertions(metrics_data, run_dir.name)
        prep_status = "OK"
    except AssertionError as e:
        prep_status = f"FAIL ({e})"

    # Embedding QA (only if embedding block exists)
    if "embedding" in metrics_data and isinstance(metrics_data.get("embedding"), dict) and metrics_data.get("embedding"):
        try:
            embedding_qa_assertions(metrics_data, run_dir.name)
            emb_status = "OK"
        except AssertionError as e:
            emb_status = f"FAIL ({e})"
    else:
        emb_status = "SKIP (no embedding block)"

    run_params = metrics_data.get("params", {})
    left = run_params.get("left_strip_frac", None)
    right = run_params.get("right_strip_frac", None)

    print(
        run_dir.name,
        "schema:",
        metrics_data.get("schema_version"),
        "final_norm:",
        final_norm_flag(metrics_data),
        "preprocessing_QA:",
        prep_status,
        "embedding_QA:",
        emb_status,
        "left_strip:",
        left,
        "right_strip:",
        right,
    )
