"""
build_global_indexes.py

Build multi-document retrieval artifacts:
1) Global dense index (FAISS) across all document chunks.
2) Lexical scope manifest (doc/trust/global) for BM25 routing/debug.

Inputs (per document folder under --data-root):
- embeddings.npy
- chunk_meta.parquet
- chunks.parquet

Outputs (under --out-dir):
- global_dense.faiss
- global_embeddings.npy
- global_meta.parquet
- lexical_manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional
import re

import faiss
import numpy as np
import pandas as pd


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)


TRUST_CANONICAL_MAP: dict[str, str] = {
    "ayrshire and arran": "NHS Ayrshire & Arran",
    "borders": "NHS Borders",
    "dumfries & galloway": "NHS Dumfries & Galloway",
    "dumfries and galloway": "NHS Dumfries & Galloway",
    "fife": "NHS Fife",
    "forth valley": "NHS Forth Valley",
    "grampian": "NHS Grampian",
    "greater glasgow & clyde": "NHS Greater Glasgow & Clyde",
    "greater glasgow and clyde": "NHS Greater Glasgow & Clyde",
    "highland": "NHS Highland",
    "lanarkshire": "NHS Lanarkshire",
    "lothian": "NHS Lothian",
    "orkney": "NHS Orkney",
    "shetland": "NHS Shetland",
    "tayside": "NHS Tayside",
    "western isles": "NHS Western Isles",
}


def trust_from_doc_id(doc_id: str) -> str:
    d = str(doc_id or "").strip()
    if not d:
        return ""
    base = d.split("-", 1)[0].strip()
    norm = base.lower().replace("nhs ", "").replace("_", " ")
    norm = re.sub(r"\s+", " ", norm).strip()
    return TRUST_CANONICAL_MAP.get(norm, f"NHS {base.replace('_', ' ').strip()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build global dense index + lexical manifest from per-document artifacts."
    )
    parser.add_argument(
        "--data-root",
        default="data_processed",
        help="Root directory containing processed document folders.",
    )
    parser.add_argument(
        "--out-dir",
        default="data_processed/_global",
        help="Output directory for global artifacts.",
    )
    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Also save concatenated global embeddings.npy (useful for debugging).",
    )
    return parser.parse_args()


def load_doc_artifacts(doc_dir: Path) -> Optional[tuple[pd.DataFrame, pd.DataFrame, np.ndarray]]:
    emb_path = doc_dir / "embeddings.npy"
    meta_path = doc_dir / "chunk_meta.parquet"
    chunks_path = doc_dir / "chunks.parquet"
    if not (emb_path.exists() and meta_path.exists() and chunks_path.exists()):
        return None
    try:
        emb = np.load(emb_path).astype("float32")
        meta = pd.read_parquet(meta_path)
        chunks = pd.read_parquet(chunks_path)
    except Exception:
        return None
    if len(meta) == 0 or emb.shape[0] != len(meta):
        return None
    return meta, chunks, emb


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metas: list[pd.DataFrame] = []
    embs: list[np.ndarray] = []
    lexical_rows: list[dict[str, Any]] = []
    doc_counts: dict[str, int] = {}
    trust_counts: dict[str, int] = {}

    for doc_dir in sorted(data_root.iterdir()):
        if not doc_dir.is_dir() or doc_dir.name.startswith("_"):
            continue
        loaded = load_doc_artifacts(doc_dir)
        if loaded is None:
            continue
        meta, chunks, emb = loaded

        if "doc_id" not in meta.columns:
            meta["doc_id"] = doc_dir.name
        meta["doc_id"] = meta["doc_id"].fillna(doc_dir.name).astype(str)
        if "trust_id" not in meta.columns:
            meta["trust_id"] = meta["doc_id"].map(trust_from_doc_id)
        if "year" not in meta.columns:
            if "report_year" in meta.columns:
                meta["year"] = pd.to_numeric(meta["report_year"], errors="coerce").astype("Int64")
            else:
                meta["year"] = pd.Series([pd.NA] * len(meta), dtype="Int64")

        metas.append(meta)
        embs.append(emb)

        # Lexical manifest stats.
        did = str(meta["doc_id"].iloc[0]) if len(meta) else doc_dir.name
        tid = trust_from_doc_id(did)
        doc_counts[did] = doc_counts.get(did, 0) + int(len(meta))
        trust_counts[tid] = trust_counts.get(tid, 0) + int(len(meta))

        # Optional lexical row dump for debugging/reporting.
        chunk_text_map: dict[str, str] = {}
        for _, r in chunks.iterrows():
            cid = str(r.get("chunk_id_global") or r.get("chunk_id") or "")
            if cid:
                chunk_text_map[cid] = str(r.get("chunk_text") or "")
        for _, r in meta.iterrows():
            cid = str(r.get("chunk_id_global") or r.get("chunk_id") or "")
            lexical_rows.append(
                {
                    "chunk_id": cid,
                    "doc_id": str(r.get("doc_id") or did),
                    "trust_id": str(r.get("trust_id") or tid),
                    "year": None if pd.isna(r.get("year")) else int(r.get("year")),
                    "chunk_text": chunk_text_map.get(cid, ""),
                }
            )

    if not metas or not embs:
        raise FileNotFoundError(f"No eligible document artifacts found under {data_root}")

    global_meta = pd.concat(metas, ignore_index=True)
    global_emb = np.vstack(embs).astype("float32")
    global_emb = l2_normalize(global_emb).astype("float32")

    d = int(global_emb.shape[1])
    index = faiss.IndexFlatIP(d)
    index.add(global_emb)

    faiss.write_index(index, str(out_dir / "global_dense.faiss"))
    global_meta.to_parquet(out_dir / "global_meta.parquet", index=False)
    if args.save_embeddings:
        np.save(out_dir / "global_embeddings.npy", global_emb)

    lexical_manifest = {
        "scope": {
            "global": {"num_chunks": int(len(global_meta))},
            "doc": {k: {"num_chunks": int(v)} for k, v in sorted(doc_counts.items())},
            "trust": {k: {"num_chunks": int(v)} for k, v in sorted(trust_counts.items())},
        }
    }
    (out_dir / "lexical_manifest.json").write_text(
        json.dumps(lexical_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame(lexical_rows).to_parquet(out_dir / "lexical_corpus.parquet", index=False)

    print(f"Saved: {out_dir / 'global_dense.faiss'}")
    print(f"Saved: {out_dir / 'global_meta.parquet'}")
    if args.save_embeddings:
        print(f"Saved: {out_dir / 'global_embeddings.npy'}")
    print(f"Saved: {out_dir / 'lexical_manifest.json'}")
    print(f"Saved: {out_dir / 'lexical_corpus.parquet'}")
    print(f"Global chunks: {len(global_meta)}")
    print(f"Docs indexed: {len(doc_counts)}")
    print(f"Trusts indexed: {len(trust_counts)}")


if __name__ == "__main__":
    main()
