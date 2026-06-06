"""
eval_finetuned_hybrid.py

Non-destructive experiment: test the fine-tuned bi-encoder in the full
hybrid pipeline without touching any existing indexes or configs.

Steps
-----
1. Create a temp staging dir with symlinks to chunks.parquet / eval_set.json
2. Build new FAISS indexes using models/miniLM-finetuned
3. Run retrieval_eval_hybrid.py for each doc
4. Compare aggregate metrics against the existing baseline results
5. Print a side-by-side table and write results/eval_finetuned_hybrid_<date>.json

Usage
-----
    python scripts/eval_finetuned_hybrid.py

Nothing in data_processed/ is modified.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_BASE = ROOT / "data_processed"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FINETUNED_MODEL = str(ROOT / "models" / "miniLM-finetuned")
STAGING_DIR = ROOT / "data_processed_finetuned_test"

_CONDA_PYTHON = Path("/opt/anaconda3/envs/rag-pipeline/bin/python")
PYTHON = str(_CONDA_PYTHON) if _CONDA_PYTHON.exists() else sys.executable

DOC_IDS = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]

BUILD_INDEX = ROOT / "scripts" / "build_index.py"
EVAL_HYBRID = ROOT / "scripts" / "retrieval_eval_hybrid.py"


# ---------------------------------------------------------------------------
# Step 1 — staging area
# ---------------------------------------------------------------------------

def setup_staging() -> None:
    """Create per-doc staging dirs with symlinks to source data files."""
    STAGING_DIR.mkdir(exist_ok=True)
    for doc_id in DOC_IDS:
        src = DATA_BASE / doc_id
        dst = STAGING_DIR / doc_id
        dst.mkdir(exist_ok=True)
        for fname in ("chunks.parquet", "eval_set.json"):
            link = dst / fname
            target = src / fname
            if not target.exists():
                raise FileNotFoundError(f"Missing source file: {target}")
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(target)
    print(f"Staging dir ready: {STAGING_DIR}")


# ---------------------------------------------------------------------------
# Step 2 — build indexes
# ---------------------------------------------------------------------------

def build_indexes() -> None:
    print("\n=== Building FAISS indexes with fine-tuned model ===")
    cmd = [
        PYTHON, str(BUILD_INDEX),
        "--data-dir", str(STAGING_DIR),
        "--model", FINETUNED_MODEL,
        "--device", "mps",
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=False)
    if result.returncode != 0:
        raise RuntimeError("build_index.py failed — see output above")


# ---------------------------------------------------------------------------
# Step 3 — run hybrid eval per doc
# ---------------------------------------------------------------------------

def run_eval(doc_id: str) -> Path:
    doc_dir = STAGING_DIR / doc_id
    out_path = doc_dir / "retrieval_results_hybrid.json"
    print(f"\n  Evaluating {doc_id} …")
    cmd = [
        PYTHON, str(EVAL_HYBRID),
        "--data-dir", str(doc_dir),
        "--model", FINETUNED_MODEL,
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-3000:])
        raise RuntimeError(f"retrieval_eval_hybrid.py failed for {doc_id}")
    return out_path


# ---------------------------------------------------------------------------
# Step 4 — aggregate metrics from result files
# ---------------------------------------------------------------------------

def extract_metrics(results_path: Path) -> dict:
    """Compute page-level Recall@k and MRR@k from a retrieval_results_hybrid.json."""
    with open(results_path) as f:
        d = json.load(f)
    results = d.get("results", [])
    ks = [1, 3, 5, 10]
    recall = {k: [] for k in ks}
    mrr = {k: [] for k in ks}
    for r in results:
        per_k = r.get("per_k", {})
        for k in ks:
            entry = per_k.get(str(k), {})
            recall[k].append(entry.get("page_recall_at_k", 0.0))
            mrr[k].append(entry.get("page_mrr_at_k", 0.0))
    out = {}
    for k in ks:
        out[f"recall@{k}"] = float(np.mean(recall[k])) if recall[k] else 0.0
        out[f"mrr@{k}"] = float(np.mean(mrr[k])) if mrr[k] else 0.0
    return out


def aggregate(doc_metrics: list[dict]) -> dict:
    keys = doc_metrics[0].keys()
    return {k: float(np.mean([d[k] for d in doc_metrics])) for k in keys}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Fine-tuned bi-encoder — full hybrid pipeline test")
    print("=" * 60)

    # 1. Staging
    setup_staging()

    # 2. Build indexes
    build_indexes()

    # 3. Run eval per doc
    print("\n=== Running hybrid retrieval eval ===")
    finetuned_per_doc = {}
    for doc_id in DOC_IDS:
        out_path = run_eval(doc_id)
        finetuned_per_doc[doc_id] = extract_metrics(out_path)

    finetuned_agg = aggregate(list(finetuned_per_doc.values()))

    # 4. Load baseline metrics from existing results
    baseline_per_doc = {}
    for doc_id in DOC_IDS:
        baseline_path = DATA_BASE / doc_id / "retrieval_results_hybrid.json"
        if baseline_path.exists():
            baseline_per_doc[doc_id] = extract_metrics(baseline_path)
        else:
            print(f"  WARNING: no baseline results for {doc_id}")

    if baseline_per_doc:
        baseline_agg = aggregate(list(baseline_per_doc.values()))
    else:
        baseline_agg = {k: 0.0 for k in finetuned_agg}

    # 5. Print comparison
    ks = [1, 3, 5, 10]
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS  (page-level, all 5 Grampian docs, hybrid RRF)")
    print("=" * 70)
    print(f"{'Metric':<15} {'Baseline':>10} {'Fine-tuned':>12} {'Delta':>8}")
    print("-" * 50)
    for k in ks:
        for metric in ("recall", "mrr"):
            key = f"{metric}@{k}"
            b = baseline_agg.get(key, 0.0)
            ft = finetuned_agg.get(key, 0.0)
            print(f"{key:<15} {b:>10.3f} {ft:>12.3f} {ft-b:>+8.3f}")
        if k < 10:
            print()

    # Per-doc breakdown for Recall@1 and MRR@10
    print("\n--- Per-doc Recall@1 ---")
    print(f"{'Doc':<25} {'Baseline':>10} {'Fine-tuned':>12} {'Delta':>8}")
    print("-" * 58)
    for doc_id in DOC_IDS:
        b = baseline_per_doc.get(doc_id, {}).get("recall@1", 0.0)
        ft = finetuned_per_doc[doc_id].get("recall@1", 0.0)
        print(f"{doc_id:<25} {b:>10.3f} {ft:>12.3f} {ft-b:>+8.3f}")

    # 6. Save JSON
    result = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "finetuned_model": FINETUNED_MODEL,
        "baseline_model": "sentence-transformers/all-MiniLM-L6-v2",
        "doc_ids": DOC_IDS,
        "aggregate": {
            "baseline": baseline_agg,
            "finetuned": finetuned_agg,
            "delta": {k: round(finetuned_agg[k] - baseline_agg.get(k, 0), 4)
                      for k in finetuned_agg},
        },
        "per_doc": {
            doc_id: {
                "baseline": baseline_per_doc.get(doc_id, {}),
                "finetuned": finetuned_per_doc[doc_id],
            }
            for doc_id in DOC_IDS
        },
    }
    stamp = datetime.now().strftime("%Y-%m-%d")
    out_path = RESULTS_DIR / f"eval_finetuned_hybrid_{stamp}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
