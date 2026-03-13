#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


DEFAULT_DOC_ID = "Grampian-2023-2024"

COLUMN_CANDIDATES: dict[str, list[str]] = {
    "id": ["chunk_id_global", "chunk_id", "id"],
    "doc_id": ["doc_id", "corpus_id", "document_id", "report_id"],
    "text": ["chunk_text", "text", "content", "body"],
    "section": ["section_title", "section", "section_name", "heading"],
    "subsection": ["subsection_title", "subsection", "subsection_name", "subheading"],
    "page_start": ["page_start", "start_page", "page_from", "page"],
    "page_end": ["page_end", "end_page", "page_to", "page"],
    "table_like": ["is_table_like", "is_table", "table_flag", "table_like"],
    "chunk_type": ["chunk_type", "table_type", "part", "segment_type"],
    "many_numbers": ["many_numbers", "numeric_heavy", "has_many_numbers"],
}

FINANCE_RE = re.compile(
    r"\b(finance|financial|budget|expenditure|income|spend|spending|surplus|deficit|"
    r"accounts?|statement of|cash flow|taxpayer|equity|savings?)\b",
    re.IGNORECASE,
)
GOV_RE = re.compile(
    r"\b(governance|accountability|remuneration|audit|assurance|board|committee|"
    r"directors?|risk management)\b",
    re.IGNORECASE,
)
CLINICAL_RE = re.compile(
    r"\b(performance|waiting times?|clinical|treatment|patient|outcome|safety|care quality)\b",
    re.IGNORECASE,
)
FINANCIAL_TABLE_RE = re.compile(
    r"\b(financial statements?|statement of|cash flow|financial position|"
    r"comprehensive net expenditure|notes to the accounts?)\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export chunk embedding UMAP projection for WIZMAP with metadata and categories."
    )
    parser.add_argument("--doc-id", default=DEFAULT_DOC_ID, help="Document id to visualize.")
    parser.add_argument("--doc-dir", default=None, help="Document directory containing chunks/embeddings files.")
    parser.add_argument("--chunks-path", default=None, help="Explicit chunks file path (.parquet or .csv).")
    parser.add_argument("--embeddings-path", default=None, help="Explicit embeddings file path (.npy or .parquet/.csv).")
    parser.add_argument("--out-dir", default="results/wizmap", help="Output directory.")
    parser.add_argument(
        "--searchable-dir",
        default=None,
        help="Optional WizMap searchable output directory. Writes data.ndjson and grid.json.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible UMAP.")
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors.")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist.")
    parser.add_argument("--metric", default="cosine", help="UMAP metric.")
    parser.add_argument("--max-preview-chars", type=int, default=220, help="Max chars for hover text preview.")
    parser.add_argument(
        "--highlight-chunk-ids",
        default="",
        help="Comma-separated chunk ids to highlight (chunk_id or chunk_id_global).",
    )
    parser.add_argument(
        "--highlight-pages",
        default="",
        help="Comma-separated page numbers or ranges (e.g., '94,95,100-104') to highlight.",
    )
    parser.add_argument(
        "--financial-pages",
        default="",
        help="Optional page list/ranges treated as financial statement pages for category rules.",
    )
    return parser.parse_args()


def choose_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    lowered = {c.lower(): c for c in df.columns}
    for col in candidates:
        match = lowered.get(col.lower())
        if match:
            return match
    return None


def parse_int_set(spec: str) -> set[int]:
    pages: set[int] = set()
    text = (spec or "").strip()
    if not text:
        return pages
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            try:
                lo = int(lo_s.strip())
                hi = int(hi_s.strip())
            except ValueError:
                continue
            if lo > hi:
                lo, hi = hi, lo
            pages.update(range(lo, hi + 1))
        else:
            try:
                pages.add(int(token))
            except ValueError:
                continue
    return pages


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def find_default_doc_dir(doc_id: str) -> Path | None:
    roots = [
        Path("data_processed_tiktoken_all_docs_224_56") / doc_id,
        Path("data_processed") / doc_id,
        Path("data_processed_toc_upgrade_5docs") / doc_id,
        Path("data_processed_toc_upgrade_test") / doc_id,
        Path("archive/2026-02-28_ablation_cleanup/data_processed_ablation_intrinsic/chunk_280_90_seg_off") / doc_id,
    ]
    for candidate in roots:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    chunks_path = Path(args.chunks_path).expanduser() if args.chunks_path else None
    emb_path = Path(args.embeddings_path).expanduser() if args.embeddings_path else None
    doc_dir = Path(args.doc_dir).expanduser() if args.doc_dir else find_default_doc_dir(args.doc_id)

    if chunks_path is None:
        if doc_dir is None:
            raise FileNotFoundError("Could not infer --doc-dir. Please pass --doc-dir or --chunks-path.")
        for name in ("chunks.parquet", "chunks.csv", "chunk_meta.parquet", "chunk_meta.csv"):
            candidate = doc_dir / name
            if candidate.exists():
                chunks_path = candidate
                break
    if chunks_path is None:
        raise FileNotFoundError("Could not find chunks file. Pass --chunks-path explicitly.")

    if emb_path is None:
        if doc_dir is None:
            doc_dir = chunks_path.parent
        for name in ("embeddings.npy", "embeddings.parquet", "embeddings.csv"):
            candidate = doc_dir / name
            if candidate.exists():
                emb_path = candidate
                break
    if emb_path is None:
        raise FileNotFoundError(
            "Could not find embeddings file in doc directory. Pass --embeddings-path explicitly."
        )
    return chunks_path, emb_path


def parse_embedding_cell(cell: Any) -> np.ndarray:
    if isinstance(cell, np.ndarray):
        return cell.astype(np.float32, copy=False)
    if isinstance(cell, list):
        return np.asarray(cell, dtype=np.float32)
    if isinstance(cell, str):
        parsed = ast.literal_eval(cell)
        return np.asarray(parsed, dtype=np.float32)
    raise ValueError("Unsupported embedding cell format.")


def load_embeddings(path: Path, meta_df: pd.DataFrame, id_col: str) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        emb = np.load(path)
        if emb.ndim != 2:
            raise ValueError(f"Expected 2D embedding array, got shape={emb.shape}")
        return emb.astype(np.float32, copy=False)

    emb_df = load_table(path)
    vector_col = choose_column(emb_df, ["embedding", "vector", "emb", "values"])
    emb_id_col = choose_column(emb_df, COLUMN_CANDIDATES["id"])

    if vector_col is not None:
        vectors = np.vstack([parse_embedding_cell(v) for v in emb_df[vector_col].tolist()]).astype(np.float32, copy=False)
    else:
        numeric_cols = [c for c in emb_df.columns if pd.api.types.is_numeric_dtype(emb_df[c])]
        if emb_id_col and emb_id_col in numeric_cols:
            numeric_cols.remove(emb_id_col)
        if not numeric_cols:
            raise ValueError("No embedding vector column found in embeddings table.")
        vectors = emb_df[numeric_cols].to_numpy(dtype=np.float32, copy=True)

    if emb_id_col and id_col in meta_df.columns:
        emb_df = emb_df.copy()
        emb_df["_emb_row"] = np.arange(len(emb_df))
        indexer = emb_df.set_index(emb_id_col)["_emb_row"]
        mapped_rows = meta_df[id_col].map(indexer)
        if mapped_rows.notna().all():
            row_ids = mapped_rows.astype(int).to_numpy()
            return vectors[row_ids]
    return vectors


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def make_page_label(start: Any, end: Any) -> str:
    if pd.isna(start) and pd.isna(end):
        return "Unknown"
    if pd.isna(end) or start == end:
        return str(int(start)) if not pd.isna(start) else str(int(end))
    return f"{int(start)}-{int(end)}"


def truncate_text(text: Any, max_chars: int) -> str:
    raw = "" if pd.isna(text) else str(text)
    clean = re.sub(r"\s+", " ", raw).strip()
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 1].rstrip() + "…"


def derive_category(row: pd.Series, financial_pages: set[int]) -> str:
    section = str(row.get("section", "") or "")
    subsection = str(row.get("subsection", "") or "")
    chunk_type = str(row.get("chunk_type", "") or "")
    text = str(row.get("text", "") or "")
    context = " ".join([section, subsection, chunk_type, text[:500]])

    page_start = row.get("page_start")
    page_end = row.get("page_end")
    start = int(page_start) if not pd.isna(page_start) else None
    end = int(page_end) if not pd.isna(page_end) else start
    is_financial_page = False
    if start is not None and end is not None and financial_pages:
        is_financial_page = any(p in financial_pages for p in range(start, end + 1))

    is_table_like = to_bool(row.get("table_like")) or to_bool(row.get("many_numbers"))
    if is_table_like and (FINANCE_RE.search(context) or FINANCIAL_TABLE_RE.search(context) or is_financial_page):
        return "Financial table"
    if FINANCE_RE.search(context):
        return "Financial narrative"
    if GOV_RE.search(context):
        return "Governance"
    if CLINICAL_RE.search(context):
        return "Clinical performance"
    return "Strategy / narrative"


def build_plot(df: pd.DataFrame, out_path: Path, doc_id: str) -> None:
    import matplotlib.pyplot as plt

    doc_label = str(doc_id).replace("-", " ", 1)
    legend_label = {
        "Financial table": "Financial tables",
        "Financial narrative": "Financial narrative",
        "Governance": "Governance",
        "Clinical performance": "Clinical performance",
        "Strategy / narrative": "Strategy / narrative",
    }
    legend_order = [
        "Financial narrative",
        "Clinical performance",
        "Governance",
        "Financial table",
        "Strategy / narrative",
    ]
    palette = {
        "Financial table": "#d73027",
        "Financial narrative": "#fc8d59",
        "Governance": "#4575b4",
        "Clinical performance": "#1a9850",
        "Strategy / narrative": "#756bb1",
    }
    fig, ax = plt.subplots(figsize=(10, 8))
    grouped = {k: v for k, v in df.groupby("category", sort=False)}
    ordered_categories = [c for c in legend_order if c in grouped] + [
        c for c in grouped if c not in legend_order
    ]
    for category in ordered_categories:
        group = grouped[category]
        ax.scatter(
            group["x"],
            group["y"],
            s=18,
            alpha=0.8,
            c=palette.get(category, "#666666"),
            label=legend_label.get(category, category),
            edgecolors="none",
        )

    highlighted = df[df["highlight"]]
    if not highlighted.empty:
        ax.scatter(
            highlighted["x"],
            highlighted["y"],
            s=58,
            marker="x",
            c="#111111",
            linewidths=1.2,
            label="Query-related chunks",
        )

    ax.set_title(f"UMAP projection of chunk embeddings (NHS {doc_label})")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def slugify_label(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    parts = [p for p in text.split("-") if p]
    return "-".join(parts[:4]) or "chunk"


def build_search_text(row: pd.Series) -> str:
    chunk_id = str(row["id"])
    page = str(row["page"])
    category = str(row["category"])
    section = str(row["section"])
    text = str(row["text"])
    return f"chunk_id {chunk_id} page {page} category {category} section {section} text {text}"


def build_topic_grid(df: pd.DataFrame, x_range: list[float], y_range: list[float]) -> dict[str, Any]:
    x0, x1 = x_range
    y0, y1 = y_range
    topic_min_x = float(np.floor(x0))
    topic_min_y = float(np.floor(y0))
    topic_max_x = float(np.ceil(x1))
    topic_max_y = float(np.ceil(y1))
    cell_size = 0.25
    extent = [
        [int(np.floor(topic_min_x)), int(np.floor(topic_min_y))],
        [int(np.ceil(topic_max_x)), int(np.ceil(topic_max_y))],
    ]
    topic_levels: dict[str, list[list[Any]]] = {}
    for level in range(6, 12):
        cell_labels: dict[tuple[float, float], dict[str, int]] = {}
        for row in df.itertuples(index=False):
            label = slugify_label(f"{row.section} {row.text}")
            cx = topic_min_x + (np.floor((float(row.x) - topic_min_x) / cell_size) + 0.5) * cell_size
            cy = topic_min_y + (np.floor((float(row.y) - topic_min_y) / cell_size) + 0.5) * cell_size
            bucket = (round(float(cx), 3), round(float(cy), 3))
            counts = cell_labels.setdefault(bucket, {})
            counts[label] = counts.get(label, 0) + 1
        entries: list[list[Any]] = []
        for (cx, cy), counts in cell_labels.items():
            best = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
            entries.append([cx, cy, best])
        topic_levels[str(level)] = sorted(entries, key=lambda item: (item[0], item[1], item[2]))
    return {
        "extent": extent,
        "data": topic_levels,
        "range": [topic_min_x, topic_min_y, topic_max_x, topic_max_y],
    }


def write_searchable_wizmap(df: pd.DataFrame, out_dir: Path, doc_id: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pad_ratio = 0.05
    x_pad = max((float(df["x"].max()) - float(df["x"].min())) * pad_ratio, 1e-6)
    y_pad = max((float(df["y"].max()) - float(df["y"].min())) * pad_ratio, 1e-6)
    x_range = [float(df["x"].min()) - x_pad, float(df["x"].max()) + x_pad]
    y_range = [float(df["y"].min()) - y_pad, float(df["y"].max()) + y_pad]

    grid_size = 200
    hist, _, _ = np.histogram2d(
        df["y"].to_numpy(dtype=np.float32),
        df["x"].to_numpy(dtype=np.float32),
        bins=grid_size,
        range=[[y_range[0], y_range[1]], [x_range[0], x_range[1]]],
    )
    density = gaussian_filter(hist.astype(np.float32), sigma=2.0)
    if float(density.max()) > 0:
        density = density / float(density.max()) * 0.0677
    grid = np.round(density, 4).tolist()

    ndjson_path = out_dir / "data.ndjson"
    with ndjson_path.open("w", encoding="utf-8") as handle:
        for row in df.itertuples(index=False):
            payload = [float(row.x), float(row.y), build_search_text(pd.Series(row._asdict()))]
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")

    grid_payload = {
        "grid": grid,
        "xRange": x_range,
        "yRange": y_range,
        "padded": True,
        "sampleSize": int(len(df)),
        "totalPointSize": int(len(df)),
        "topic": build_topic_grid(df, x_range=x_range, y_range=y_range),
        "embeddingName": f"NHS {doc_id} (WIZMAP)",
    }
    (out_dir / "grid.json").write_text(
        json.dumps(grid_payload, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    np.random.seed(int(args.seed))
    try:
        import umap
    except ImportError as exc:  # pragma: no cover - dependency availability is environment-specific
        raise ImportError(
            "Missing dependency 'umap-learn'. Install it with: pip install umap-learn"
        ) from exc

    chunks_path, emb_path = resolve_paths(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_df = load_table(chunks_path)

    id_col = choose_column(meta_df, COLUMN_CANDIDATES["id"])
    if not id_col:
        raise ValueError("Could not find chunk id column in chunks metadata.")
    doc_col = choose_column(meta_df, COLUMN_CANDIDATES["doc_id"])
    text_col = choose_column(meta_df, COLUMN_CANDIDATES["text"])
    section_col = choose_column(meta_df, COLUMN_CANDIDATES["section"])
    subsection_col = choose_column(meta_df, COLUMN_CANDIDATES["subsection"])
    page_start_col = choose_column(meta_df, COLUMN_CANDIDATES["page_start"])
    page_end_col = choose_column(meta_df, COLUMN_CANDIDATES["page_end"])
    table_like_col = choose_column(meta_df, COLUMN_CANDIDATES["table_like"])
    chunk_type_col = choose_column(meta_df, COLUMN_CANDIDATES["chunk_type"])
    many_numbers_col = choose_column(meta_df, COLUMN_CANDIDATES["many_numbers"])

    if doc_col and args.doc_id:
        mask = meta_df[doc_col].astype(str).str.lower() == args.doc_id.lower()
        if mask.any():
            meta_df = meta_df.loc[mask].reset_index(drop=True)

    emb = load_embeddings(emb_path, meta_df, id_col=id_col)
    n = min(len(meta_df), emb.shape[0])
    meta_df = meta_df.iloc[:n].copy().reset_index(drop=True)
    emb = emb[:n]
    if n == 0:
        raise ValueError("No chunks available after filtering.")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(args.n_neighbors),
        min_dist=float(args.min_dist),
        metric=args.metric,
        random_state=int(args.seed),
    )
    coords = reducer.fit_transform(emb)

    out_df = pd.DataFrame(
        {
            "id": meta_df[id_col].astype(str),
            "x": coords[:, 0],
            "y": coords[:, 1],
            "label": (
                meta_df[section_col].astype(str)
                if section_col
                else pd.Series(["Unknown"] * len(meta_df))
            ),
            "text": (
                meta_df[text_col].map(lambda t: truncate_text(t, int(args.max_preview_chars)))
                if text_col
                else pd.Series([""] * len(meta_df))
            ),
            "page": pd.Series(
                [
                    make_page_label(
                        meta_df.iloc[i][page_start_col] if page_start_col else np.nan,
                        meta_df.iloc[i][page_end_col] if page_end_col else np.nan,
                    )
                    for i in range(len(meta_df))
                ]
            ),
            "section": (
                meta_df[section_col].astype(str)
                if section_col
                else pd.Series(["Unknown"] * len(meta_df))
            ),
            "subsection": (
                meta_df[subsection_col].astype(str)
                if subsection_col
                else pd.Series(["Unknown"] * len(meta_df))
            ),
            "page_start": (
                pd.to_numeric(meta_df[page_start_col], errors="coerce")
                if page_start_col
                else pd.Series([np.nan] * len(meta_df))
            ),
            "page_end": (
                pd.to_numeric(meta_df[page_end_col], errors="coerce")
                if page_end_col
                else pd.Series([np.nan] * len(meta_df))
            ),
            "table_like": (
                meta_df[table_like_col].map(to_bool)
                if table_like_col
                else pd.Series([False] * len(meta_df))
            ),
            "many_numbers": (
                meta_df[many_numbers_col].map(to_bool)
                if many_numbers_col
                else pd.Series([False] * len(meta_df))
            ),
            "chunk_type": (
                meta_df[chunk_type_col].astype(str)
                if chunk_type_col
                else pd.Series([""] * len(meta_df))
            ),
        }
    )

    financial_pages = parse_int_set(args.financial_pages)
    out_df["category"] = out_df.apply(lambda r: derive_category(r, financial_pages=financial_pages), axis=1)

    highlight_ids = {c.strip() for c in args.highlight_chunk_ids.split(",") if c.strip()}
    highlight_pages = parse_int_set(args.highlight_pages)
    out_df["highlight"] = out_df.apply(
        lambda r: (
            (str(r["id"]) in highlight_ids)
            or any(
                p in highlight_pages
                for p in range(
                    int(r["page_start"]) if not pd.isna(r["page_start"]) else -1,
                    (int(r["page_end"]) if not pd.isna(r["page_end"]) else -2) + 1,
                )
            )
        ),
        axis=1,
    )

    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", args.doc_id)
    out_csv = out_dir / f"{slug}_wizmap_umap.csv"
    out_json = out_dir / f"{slug}_wizmap_umap.json"
    out_plot = out_dir / f"{slug}_wizmap_umap_preview.png"

    export_cols = ["id", "x", "y", "label", "text", "page", "section", "category", "highlight"]
    out_df[export_cols].to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(out_df[export_cols].to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8")
    build_plot(out_df, out_plot, args.doc_id)

    if args.searchable_dir:
        write_searchable_wizmap(out_df, Path(args.searchable_dir), args.doc_id)

    counts = out_df["category"].value_counts(dropna=False).sort_values(ascending=False)
    print(f"Document: {args.doc_id}")
    print(f"Chunks loaded: {len(out_df)}")
    print(f"Categories: {counts.shape[0]}")
    print("Category counts:")
    for category, count in counts.items():
        print(f"  - {category}: {int(count)}")
    print(f"Highlighted points: {int(out_df['highlight'].sum())}")
    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote JSON: {out_json}")
    print(f"Wrote preview plot: {out_plot}")
    if args.searchable_dir:
        print(f"Wrote searchable WizMap: {Path(args.searchable_dir)}")


if __name__ == "__main__":
    main()
