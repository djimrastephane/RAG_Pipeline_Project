#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a standalone Plotly HTML viewer for the Grampian shared UMAP."
    )
    p.add_argument("--input-csv", default="results/wizmap/grampian_temporal_joint_umap.csv")
    p.add_argument("--out-html", default="results/wizmap/grampian_temporal_joint_wizstyle.html")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_csv = Path(args.input_csv)
    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    required = {
        "id",
        "doc_id",
        "report_year",
        "x",
        "y",
        "section",
        "text_preview",
        "page_start_num",
        "page_end_num",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {in_csv}: {sorted(missing)}")

    df = df.copy()
    df["report_year"] = pd.to_numeric(df["report_year"], errors="coerce").astype("Int64")
    df = df[df["report_year"].notna()].reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows with valid report_year.")
    df["report_year"] = df["report_year"].astype(int).astype(str)
    df["page_label"] = df.apply(
        lambda r: ""
        if pd.isna(r["page_start_num"])
        else (
            str(int(r["page_start_num"]))
            if pd.isna(r["page_end_num"]) or int(r["page_start_num"]) == int(r["page_end_num"])
            else f"{int(r['page_start_num'])}-{int(r['page_end_num'])}"
        ),
        axis=1,
    )
    df["hover_text"] = (
        "<b>" + df["doc_id"].astype(str) + "</b><br>"
        + "Year: " + df["report_year"].astype(str) + "<br>"
        + "Section: " + df["section"].astype(str) + "<br>"
        + "Page: " + df["page_label"].replace("", "Unknown").astype(str) + "<br>"
        + "Chunk: " + df["id"].astype(str) + "<br>"
        + df["text_preview"].astype(str)
    )

    fig = px.scatter(
        df,
        x="x",
        y="y",
        animation_frame="report_year",
        hover_name="doc_id",
        hover_data={
            "section": True,
            "page_label": True,
            "id": True,
            "text_preview": True,
            "x": False,
            "y": False,
        },
        custom_data=["hover_text"],
        template="plotly_dark",
        title="NHS Grampian Shared Map",
    )
    fig.update_traces(
        marker={"size": 6, "color": "#f97316", "opacity": 0.8},
        hovertemplate="%{customdata[0]}<extra></extra>",
    )
    fig.update_layout(
        paper_bgcolor="#0b0f14",
        plot_bgcolor="#090d13",
        font={"family": "IBM Plex Sans, Avenir Next, Segoe UI, sans-serif", "color": "#d7e1ea"},
        title={"x": 0.02},
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
    )

    # Make the animation usable as a year scrubber rather than autoplaying.
    if fig.layout.updatemenus:
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 900
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 300

    fig.write_html(
        out_html,
        include_plotlyjs=True,
        full_html=True,
        config={"displaylogo": False, "responsive": True},
    )
    print(f"Wrote {out_html}")


if __name__ == "__main__":
    main()
