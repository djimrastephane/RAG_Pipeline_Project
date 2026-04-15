#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import _matplotlib_env
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create raw embedding tile heatmaps for selected chunk vectors."
    )
    p.add_argument("--doc-id", required=True, help="Document id under data_processed/")
    p.add_argument("--data-root", default="data_processed", help="Processed data root.")
    p.add_argument("--out-dir", default="results", help="Output directory.")
    p.add_argument("--samples", type=int, default=3, help="Number of embedding tiles.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--chunk-ids",
        nargs="*",
        default=None,
        help="Optional explicit chunk_ids to visualize (otherwise random sample).",
    )
    return p.parse_args()


def factor_shape(d: int) -> tuple[int, int]:
    """Pick near-square factors for reshaping a 1D vector into a 2D tile."""
    root = int(np.sqrt(d))
    for r in range(root, 0, -1):
        if d % r == 0:
            c = d // r
            return r, c
    return 1, d


def short_label(doc_id: str, chunk_id: str) -> str:
    cid = str(chunk_id)
    if ":" in cid:
        cid = cid.split(":")[-1]
    return f"{doc_id}\\n{cid}"


def main() -> None:
    args = parse_args()
    doc_dir = Path(args.data_root) / args.doc_id
    emb_path = doc_dir / "embeddings.npy"
    meta_path = doc_dir / "chunk_meta.parquet"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not emb_path.exists():
        raise FileNotFoundError(f"Missing embeddings: {emb_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")

    emb = np.load(emb_path).astype(np.float32, copy=False)
    meta = pd.read_parquet(meta_path)
    if "chunk_id_global" in meta.columns:
        id_col = "chunk_id_global"
    elif "chunk_id" in meta.columns:
        id_col = "chunk_id"
    else:
        raise ValueError("chunk_meta.parquet missing chunk_id columns.")

    n = min(len(meta), emb.shape[0])
    emb = emb[:n]
    meta = meta.iloc[:n].reset_index(drop=True)
    ids = meta[id_col].astype(str).tolist()
    id_to_idx = {cid: i for i, cid in enumerate(ids)}

    chosen_idx: list[int] = []
    if args.chunk_ids:
        for cid in args.chunk_ids:
            if cid in id_to_idx:
                chosen_idx.append(id_to_idx[cid])
    if not chosen_idx:
        rng = np.random.default_rng(int(args.seed))
        take = min(int(args.samples), n)
        chosen_idx = rng.choice(n, size=take, replace=False).tolist()

    tile_rows, tile_cols = factor_shape(int(emb.shape[1]))
    k = len(chosen_idx)
    fig, axes = plt.subplots(1, k, figsize=(3.2 * k, 3.1))
    if k == 1:
        axes = [axes]

    for ax, idx in zip(axes, chosen_idx):
        vec = emb[int(idx)]
        tile = vec.reshape(tile_rows, tile_cols)
        ax.imshow(tile, aspect="auto", cmap="YlOrRd")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(short_label(args.doc_id, ids[int(idx)]), fontsize=10)

    fig.suptitle(f"Raw MiniLM embedding tiles ({args.doc_id})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = out_dir / f"embedding_raw_tiles_{args.doc_id}.png"
    fig.savefig(out_path, dpi=250)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
