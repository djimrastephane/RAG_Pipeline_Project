#!/usr/bin/env bash
# Batch preprocess all PDFs and rebuild FAISS indexes.
# Usage: bash scripts/preprocess_batch.sh [--skip-index]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PDF_DIR="$REPO_ROOT/Data/Annual Accounts NHS Grampian/Preliminary_Test"
OUT_ROOT="$REPO_ROOT/data_processed"
LOG_FILE="$OUT_ROOT/preprocess_batch.log"
PYTHON="/opt/anaconda3/envs/rag-pipeline/bin/python"
SKIP_INDEX=0

# Prevent HuggingFace tokenizer from forking (causes segfault on macOS during batch encoding)
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

for arg in "$@"; do
  [ "$arg" = "--skip-index" ] && SKIP_INDEX=1
done

mkdir -p "$OUT_ROOT"

log() { echo "[$(date '+%Y-%m-%dT%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "=== Batch preprocessing start ==="
log "PDF dir : $PDF_DIR"
log "Out root: $OUT_ROOT"

PASS=0
FAIL=0
SKIP=0
FAILED_DOCS=()

while IFS= read -r -d '' pdf; do
  doc_id="$(basename "$pdf" .pdf)"
  out_dir="$OUT_ROOT/$doc_id"

  # Skip if core artifacts already exist
  if [ -f "$out_dir/chunks.parquet" ] && [ -f "$out_dir/pages.parquet" ]; then
    log "SKIP $doc_id (already preprocessed)"
    SKIP=$((SKIP + 1))
    continue
  fi

  log "--- START $doc_id ---"
  if "$PYTHON" "$REPO_ROOT/preprocess_hybrid.py" \
      --pdf-path "$pdf" \
      --out-root "$OUT_ROOT" \
      --mixed-routing \
      >> "$LOG_FILE" 2>&1; then
    log "OK   $doc_id"
    PASS=$((PASS + 1))
  else
    log "FAIL $doc_id (exit $?)"
    FAIL=$((FAIL + 1))
    FAILED_DOCS+=("$doc_id")
  fi
done < <(find "$PDF_DIR" -maxdepth 1 -name "*.pdf" -print0 | sort -z)

log "=== Preprocessing done: $PASS ok, $SKIP skipped, $FAIL failed ==="
if [ ${#FAILED_DOCS[@]} -gt 0 ]; then
  log "Failed: ${FAILED_DOCS[*]}"
fi

if [ "$SKIP_INDEX" -eq 1 ]; then
  log "Skipping index build (--skip-index)."
  exit 0
fi

log "=== Building FAISS indexes ==="
if "$PYTHON" "$REPO_ROOT/scripts/build_index.py" \
    --data-dir "$OUT_ROOT" \
    --device cpu \
    >> "$LOG_FILE" 2>&1; then
  log "=== Index build complete ==="
else
  log "=== Index build FAILED (check $LOG_FILE) ==="
  exit 1
fi
