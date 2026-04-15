#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build WizMap searchable files from an existing projection CSV."
    )
    p.add_argument("--input-csv", required=True, help="Projection CSV containing x/y and metadata.")
    p.add_argument("--doc-id", required=True, help="Document id to filter.")
    p.add_argument("--out-dir", required=True, help="Output directory for data.ndjson and grid.json.")
    return p.parse_args()


def slugify_label(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    parts = [part for part in text.split("-") if part]
    return "-".join(parts[:4]) or "chunk"


def make_page_label(start: Any, end: Any) -> str:
    if pd.isna(start) and pd.isna(end):
        return "Unknown"
    if pd.isna(end) or start == end:
        return str(int(start)) if not pd.isna(start) else str(int(end))
    return f"{int(start)}-{int(end)}"


def infer_category(section: str, text: str) -> str:
    context = f"{section} {text}".lower()
    if any(token in context for token in ["financial", "cash flow", "statement", "expenditure", "income", "budget"]):
        return "Financial narrative"
    if any(token in context for token in ["governance", "audit", "committee", "remuneration", "accountable"]):
        return "Governance"
    if any(token in context for token in ["performance", "patient", "clinical", "treatment", "waiting"]):
        return "Clinical performance"
    return "Strategy / narrative"


def build_search_text(row: pd.Series) -> str:
    return (
        f"chunk_id {row['id']} page {row['page']} category {row['category']} "
        f"section {row['section']} text {row['text_preview']}"
    )


def build_topic_grid(df: pd.DataFrame, x_range: list[float], y_range: list[float]) -> dict[str, Any]:
    x0, x1 = x_range
    y0, y1 = y_range
    topic_min_x = float(np.floor(x0))
    topic_min_y = float(np.floor(y0))
    topic_max_x = float(np.ceil(x1))
    topic_max_y = float(np.ceil(y1))
    cell_size = 0.25
    levels: dict[str, list[list[Any]]] = {}
    for level in range(6, 12):
        cells: dict[tuple[float, float], dict[str, int]] = {}
        for row in df.itertuples(index=False):
            label = slugify_label(f"{row.section} {row.text_preview}")
            cx = topic_min_x + (np.floor((float(row.x) - topic_min_x) / cell_size) + 0.5) * cell_size
            cy = topic_min_y + (np.floor((float(row.y) - topic_min_y) / cell_size) + 0.5) * cell_size
            key = (round(float(cx), 3), round(float(cy), 3))
            labels = cells.setdefault(key, {})
            labels[label] = labels.get(label, 0) + 1
        entries = []
        for (cx, cy), counts in cells.items():
            best = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
            entries.append([cx, cy, best])
        levels[str(level)] = sorted(entries, key=lambda item: (item[0], item[1], item[2]))
    return {
        "extent": [
            [int(np.floor(topic_min_x)), int(np.floor(topic_min_y))],
            [int(np.ceil(topic_max_x)), int(np.ceil(topic_max_y))],
        ],
        "data": levels,
        "range": [topic_min_x, topic_min_y, topic_max_x, topic_max_y],
    }


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    df = df[df["doc_id"].astype(str) == args.doc_id].copy().reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No rows found for doc_id={args.doc_id}")

    df["page"] = [
        make_page_label(start, end)
        for start, end in zip(df["page_start_num"], df["page_end_num"])
    ]
    df["category"] = [
        infer_category(str(section), str(text))
        for section, text in zip(df["section"], df["text_preview"])
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pad_ratio = 0.05
    x_pad = max((float(df["x"].max()) - float(df["x"].min())) * pad_ratio, 1e-6)
    y_pad = max((float(df["y"].max()) - float(df["y"].min())) * pad_ratio, 1e-6)
    x_range = [float(df["x"].min()) - x_pad, float(df["x"].max()) + x_pad]
    y_range = [float(df["y"].min()) - y_pad, float(df["y"].max()) + y_pad]

    hist, _, _ = np.histogram2d(
        df["y"].to_numpy(dtype=np.float32),
        df["x"].to_numpy(dtype=np.float32),
        bins=200,
        range=[[y_range[0], y_range[1]], [x_range[0], x_range[1]]],
    )
    density = gaussian_filter(hist.astype(np.float32), sigma=2.0)
    if float(density.max()) > 0:
        density = density / float(density.max()) * 0.0677

    with (out_dir / "data.ndjson").open("w", encoding="utf-8") as handle:
        for row in df.to_dict(orient="records"):
            handle.write(
                json.dumps(
                    [float(row["x"]), float(row["y"]), build_search_text(pd.Series(row))],
                    ensure_ascii=False,
                )
            )
            handle.write("\n")

    payload = {
        "grid": np.round(density, 4).tolist(),
        "xRange": x_range,
        "yRange": y_range,
        "padded": True,
        "sampleSize": int(len(df)),
        "totalPointSize": int(len(df)),
        "topic": build_topic_grid(df, x_range=x_range, y_range=y_range),
        "embeddingName": f"NHS {args.doc_id} (WIZMAP)",
    }
    (out_dir / "grid.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_dir / 'data.ndjson'}")
    print(f"Wrote {out_dir / 'grid.json'}")


if __name__ == "__main__":
    main()
