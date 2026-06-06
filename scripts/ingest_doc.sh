#!/usr/bin/env bash
# One-shot ingestion: preprocess → build index → run silver eval.
# Usage: bash scripts/ingest_doc.sh Data/Grampian-2024-2025.pdf [--mixed-routing]
set -euo pipefail

PDF_PATH="${1:?Usage: $0 <path/to/doc.pdf> [extra preprocess flags]}"
shift
EXTRA_FLAGS="$*"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DOC_ID="$(basename "$PDF_PATH" .pdf)"
DATA_DIR="$REPO_ROOT/data_processed/$DOC_ID"

echo "=== ingest_doc: $DOC_ID ==="

echo "--- [1/3] Preprocessing ---"
python "$REPO_ROOT/preprocess_hybrid.py" \
    --pdf-path "$PDF_PATH" \
    --mixed-routing \
    $EXTRA_FLAGS

echo "--- [2/3] Building FAISS index ---"
python "$REPO_ROOT/scripts/build_index.py" \
    --data-dir "$DATA_DIR"

echo "--- [3/3] Silver eval ---"
python "$REPO_ROOT/scripts/generate_silver_eval_sets.py" \
    --doc-id "$DOC_ID" \
    --force

if [ -f "$DATA_DIR/eval_set.json" ]; then
    echo "--- [3b/3] Retrieval regression check ---"
    python "$REPO_ROOT/scripts/retrieval_eval_hybrid.py" \
        --data-dir "$DATA_DIR" \
        --k-list 1,3,5,10
fi

echo "=== Done: $DOC_ID ==="
