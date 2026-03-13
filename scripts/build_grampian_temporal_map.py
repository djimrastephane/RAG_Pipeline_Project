#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DOC_RE = re.compile(r"^Grampian-(\d{4})-(\d{4})$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a single shared 2D embedding map across all Grampian docs and export temporal artifacts."
    )
    p.add_argument("--data-root", default="data_processed")
    p.add_argument("--out-dir", default="results/wizmap")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-neighbors", type=int, default=20)
    p.add_argument("--min-dist", type=float, default=0.08)
    p.add_argument("--metric", default="cosine")
    p.add_argument("--max-preview-chars", type=int, default=220)
    p.add_argument("--gif-fps", type=int, default=1)
    return p.parse_args()


def _parse_years(doc_id: str) -> tuple[int | None, int | None]:
    m = DOC_RE.match(doc_id)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _choose_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _trunc(text: Any, n: int) -> str:
    s = str(text or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _build_gif(df: pd.DataFrame, out_gif: Path, fps: int) -> None:
    import io
    from PIL import Image

    years = sorted([int(y) for y in df["report_year"].dropna().unique().tolist()])
    if not years:
        return

    xlo, xhi = float(df["x"].min()), float(df["x"].max())
    ylo, yhi = float(df["y"].min()), float(df["y"].max())
    frames: list[Image.Image] = []

    for yr in years:
        sub = df[df["report_year"] == yr]
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(df["x"], df["y"], s=7, alpha=0.08, c="#9aa0a6", edgecolors="none")
        ax.scatter(sub["x"], sub["y"], s=14, alpha=0.85, c="#005f73", edgecolors="none")
        ax.set_xlim(xlo - 0.5, xhi + 0.5)
        ax.set_ylim(ylo - 0.5, yhi + 0.5)
        ax.set_title(f"NHS Grampian shared embedding map - report year {yr}")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.text(
            0.02,
            0.98,
            f"Highlighted: {len(sub)} chunks",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        fig.tight_layout()
        bio = io.BytesIO()
        fig.savefig(bio, format="png", dpi=170)
        plt.close(fig)
        bio.seek(0)
        frames.append(Image.open(bio).convert("P"))

    if frames:
        duration_ms = max(250, int(1000 / max(1, fps)))
        frames[0].save(
            out_gif,
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=duration_ms,
            loop=0,
        )


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import umap
    except ImportError as exc:
        raise ImportError("Missing dependency 'umap-learn'. Install with: pip install umap-learn") from exc

    rows: list[pd.DataFrame] = []
    embs: list[np.ndarray] = []
    docs_used: list[str] = []
    docs_missing: list[str] = []

    for d in sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("Grampian-")], key=lambda p: p.name):
        emb_path = d / "embeddings.npy"
        chunks_path = d / "chunks.parquet"
        if not emb_path.exists() or not chunks_path.exists():
            docs_missing.append(d.name)
            continue
        emb = np.load(emb_path).astype(np.float32, copy=False)
        meta = pd.read_parquet(chunks_path)
        id_col = _choose_col(meta, ["chunk_id_global", "chunk_id"])
        text_col = _choose_col(meta, ["chunk_text", "text"])
        page_start_col = _choose_col(meta, ["page_start", "page"])
        page_end_col = _choose_col(meta, ["page_end", "page"])
        section_col = _choose_col(meta, ["section_title", "section"])
        if not id_col or not text_col:
            docs_missing.append(d.name)
            continue

        n = min(len(meta), emb.shape[0])
        meta = meta.iloc[:n].copy().reset_index(drop=True)
        emb = emb[:n]
        sy, ry = _parse_years(d.name)
        meta["doc_id"] = d.name
        meta["start_year"] = sy
        meta["report_year"] = ry
        meta["id"] = meta[id_col].astype(str)
        meta["text_preview"] = meta[text_col].map(lambda t: _trunc(t, int(args.max_preview_chars)))
        meta["section"] = meta[section_col].astype(str) if section_col else "Unknown"
        meta["page_start_num"] = pd.to_numeric(meta[page_start_col], errors="coerce") if page_start_col else np.nan
        meta["page_end_num"] = pd.to_numeric(meta[page_end_col], errors="coerce") if page_end_col else np.nan
        rows.append(meta[["id", "doc_id", "start_year", "report_year", "section", "text_preview", "page_start_num", "page_end_num"]])
        embs.append(emb)
        docs_used.append(d.name)

    if not rows or not embs:
        raise RuntimeError("No valid Grampian docs found in data root.")

    meta_all = pd.concat(rows, ignore_index=True)
    emb_all = np.vstack(embs).astype(np.float32, copy=False)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(args.n_neighbors),
        min_dist=float(args.min_dist),
        metric=str(args.metric),
        random_state=int(args.seed),
    )
    coords = reducer.fit_transform(emb_all)
    meta_all["x"] = coords[:, 0]
    meta_all["y"] = coords[:, 1]

    out_csv = out_dir / "grampian_temporal_joint_umap.csv"
    out_json = out_dir / "grampian_temporal_joint_umap.json"
    out_png = out_dir / "grampian_temporal_joint_preview.png"
    out_gif = out_dir / "grampian_temporal_joint_animation.gif"
    out_summary = out_dir / "grampian_temporal_joint_summary.json"

    meta_all.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(meta_all.to_dict(orient="records"), ensure_ascii=False), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(10, 8))
    years = sorted([int(y) for y in meta_all["report_year"].dropna().unique().tolist()])
    cmap = plt.get_cmap("viridis", len(years) if years else 1)
    for i, yr in enumerate(years):
        s = meta_all[meta_all["report_year"] == yr]
        ax.scatter(s["x"], s["y"], s=10, alpha=0.70, edgecolors="none", color=cmap(i), label=str(yr))
    ax.set_title("NHS Grampian shared embedding map (all available years)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(title="Report year", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    _build_gif(meta_all, out_gif, fps=int(args.gif_fps))

    summary = {
        "docs_used": docs_used,
        "docs_missing_or_skipped": docs_missing,
        "n_docs_used": len(docs_used),
        "n_points_total": int(len(meta_all)),
        "years_present": years,
        "out_csv": str(out_csv),
        "out_json": str(out_json),
        "out_png": str(out_png),
        "out_gif": str(out_gif),
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
