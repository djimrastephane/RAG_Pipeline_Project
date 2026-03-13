#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a standalone interactive HTML viewer for the Grampian temporal joint UMAP."
    )
    p.add_argument("--input-csv", default="results/wizmap/grampian_temporal_joint_umap.csv")
    p.add_argument("--out-html", default="results/wizmap/grampian_temporal_joint_interactive.html")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_csv = Path(args.input_csv)
    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    needed = {"id", "doc_id", "report_year", "x", "y", "section", "text_preview", "page_start_num", "page_end_num"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {in_csv}: {sorted(missing)}")

    df = df.copy()
    df["report_year"] = pd.to_numeric(df["report_year"], errors="coerce").astype("Int64")
    df = df[df["report_year"].notna()].reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows with valid report_year.")
    df["report_year"] = df["report_year"].astype(int)

    years = sorted(df["report_year"].unique().tolist())
    year_counts = df.groupby("report_year").size().to_dict()

    records = []
    for r in df.itertuples(index=False):
        records.append(
            {
                "id": str(r.id),
                "doc_id": str(r.doc_id),
                "report_year": int(r.report_year),
                "x": float(r.x),
                "y": float(r.y),
                "section": str(r.section),
                "text_preview": str(r.text_preview),
                "page_start": None if pd.isna(r.page_start_num) else int(r.page_start_num),
                "page_end": None if pd.isna(r.page_end_num) else int(r.page_end_num),
            }
        )

    payload = {
        "points": records,
        "years": years,
        "year_counts": {str(k): int(v) for k, v in year_counts.items()},
        "x_min": float(df["x"].min()),
        "x_max": float(df["x"].max()),
        "y_min": float(df["y"].min()),
        "y_max": float(df["y"].max()),
    }

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Grampian Temporal Embedding Map</title>
  <style>
    :root {{
      --bg: #f4f1ea;
      --fg: #1f2933;
      --muted: #5f6c7b;
      --accent: #0b6e4f;
      --accent2: #d1495b;
      --panel: #ffffff;
      --border: #d9d3c7;
    }}
    html, body {{ margin: 0; padding: 0; background: var(--bg); color: var(--fg); font-family: "Avenir Next", "Segoe UI", sans-serif; }}
    .wrap {{ max-width: 1200px; margin: 16px auto; padding: 0 16px 20px; }}
    h1 {{ margin: 6px 0 6px; font-size: 1.35rem; }}
    .sub {{ color: var(--muted); margin: 0 0 12px; }}
    .controls {{
      display: grid;
      grid-template-columns: 120px 1fr 120px 120px 220px;
      gap: 10px;
      align-items: center;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
      margin-bottom: 10px;
    }}
    .controls button {{
      border: 1px solid var(--border);
      background: #fff;
      border-radius: 8px;
      padding: 8px 10px;
      cursor: pointer;
    }}
    .controls button:hover {{ border-color: var(--accent); }}
    .controls input[type=range] {{ width: 100%; }}
    .badge {{
      text-align: center;
      font-weight: 600;
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 10px;
    }}
    .canvas-wrap {{
      position: relative;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
    }}
    canvas {{ display: block; width: 100%; height: 72vh; min-height: 520px; }}
    .tooltip {{
      position: absolute;
      pointer-events: none;
      background: rgba(31, 41, 51, 0.95);
      color: #fff;
      font-size: 12px;
      padding: 8px 9px;
      border-radius: 8px;
      max-width: 360px;
      line-height: 1.35;
      display: none;
      z-index: 3;
    }}
    .legend {{ margin-top: 8px; color: var(--muted); font-size: 13px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>NHS Grampian Embedding Shift (Shared UMAP)</h1>
    <p class="sub">Single common map for all available years. Move the slider or play to inspect year-by-year distribution changes.</p>
    <div class="controls">
      <button id="playBtn">Play</button>
      <input id="yearSlider" type="range" min="0" max="0" step="1" value="0" />
      <div id="yearBadge" class="badge"></div>
      <div id="countBadge" class="badge"></div>
      <label><input id="ghostToggle" type="checkbox" checked /> Show all years as faint background</label>
    </div>
    <div class="canvas-wrap">
      <canvas id="plot"></canvas>
      <div id="tip" class="tooltip"></div>
    </div>
    <div class="legend">Color: active year points. Gray: all-year background. Hover points for chunk details.</div>
  </div>
  <script>
    const DATA = {json.dumps(payload, ensure_ascii=False)};
    const canvas = document.getElementById('plot');
    const tip = document.getElementById('tip');
    const slider = document.getElementById('yearSlider');
    const yearBadge = document.getElementById('yearBadge');
    const countBadge = document.getElementById('countBadge');
    const playBtn = document.getElementById('playBtn');
    const ghostToggle = document.getElementById('ghostToggle');

    const years = DATA.years;
    slider.max = String(years.length - 1);
    slider.value = "0";

    const DPR = Math.max(1, window.devicePixelRatio || 1);
    const ctx = canvas.getContext('2d');
    const PAD = 40;
    let playing = false;
    let timer = null;
    let hover = null;
    let activeYear = years[0];

    const points = DATA.points;
    const byYear = new Map();
    for (const y of years) byYear.set(y, []);
    for (const p of points) byYear.get(p.report_year).push(p);

    function resize() {{
      const r = canvas.getBoundingClientRect();
      canvas.width = Math.floor(r.width * DPR);
      canvas.height = Math.floor(r.height * DPR);
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
      draw();
    }}

    function sx(x) {{
      const w = canvas.clientWidth;
      return PAD + ((x - DATA.x_min) / (DATA.x_max - DATA.x_min)) * (w - 2 * PAD);
    }}
    function sy(y) {{
      const h = canvas.clientHeight;
      return h - PAD - ((y - DATA.y_min) / (DATA.y_max - DATA.y_min)) * (h - 2 * PAD);
    }}

    function drawAxes() {{
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      ctx.strokeStyle = '#d2ccc2';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(PAD, h - PAD);
      ctx.lineTo(w - PAD, h - PAD);
      ctx.moveTo(PAD, PAD);
      ctx.lineTo(PAD, h - PAD);
      ctx.stroke();
    }}

    function draw() {{
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      ctx.clearRect(0, 0, w, h);
      drawAxes();

      if (ghostToggle.checked) {{
        ctx.fillStyle = 'rgba(120, 130, 140, 0.14)';
        for (const p of points) {{
          ctx.beginPath();
          ctx.arc(sx(p.x), sy(p.y), 2, 0, Math.PI * 2);
          ctx.fill();
        }}
      }}

      const curr = byYear.get(activeYear) || [];
      ctx.fillStyle = 'rgba(11, 110, 79, 0.92)';
      for (const p of curr) {{
        const x = sx(p.x), y = sy(p.y);
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
      }}

      if (hover) {{
        ctx.strokeStyle = '#d1495b';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(sx(hover.x), sy(hover.y), 6, 0, Math.PI * 2);
        ctx.stroke();
      }}

      yearBadge.textContent = `Year: ${{activeYear}}`;
      const c = DATA.year_counts[String(activeYear)] || 0;
      countBadge.textContent = `Chunks: ${{c}}`;
    }}

    function nearestPoint(mx, my) {{
      const curr = byYear.get(activeYear) || [];
      let best = null;
      let bestD2 = 999999;
      for (const p of curr) {{
        const dx = sx(p.x) - mx;
        const dy = sy(p.y) - my;
        const d2 = dx * dx + dy * dy;
        if (d2 < bestD2) {{
          bestD2 = d2;
          best = p;
        }}
      }}
      return bestD2 <= 90 ? best : null;
    }}

    function showTip(p, x, y) {{
      if (!p) {{
        tip.style.display = 'none';
        return;
      }}
      const page = (p.page_start == null) ? 'Unknown' :
        (p.page_end == null || p.page_end === p.page_start ? `${{p.page_start}}` : `${{p.page_start}}-${{p.page_end}}`);
      tip.innerHTML = `<b>${{p.doc_id}}</b><br/>Chunk: ${{p.id}}<br/>Section: ${{p.section}}<br/>Page: ${{page}}<br/><div style="margin-top:4px">${{p.text_preview}}</div>`;
      tip.style.left = `${{x + 12}}px`;
      tip.style.top = `${{y + 12}}px`;
      tip.style.display = 'block';
    }}

    slider.addEventListener('input', () => {{
      activeYear = years[Number(slider.value)];
      hover = null;
      showTip(null, 0, 0);
      draw();
    }});

    ghostToggle.addEventListener('change', draw);

    playBtn.addEventListener('click', () => {{
      playing = !playing;
      playBtn.textContent = playing ? 'Pause' : 'Play';
      if (playing) {{
        timer = setInterval(() => {{
          let i = Number(slider.value);
          i = (i + 1) % years.length;
          slider.value = String(i);
          activeYear = years[i];
          hover = null;
          showTip(null, 0, 0);
          draw();
        }}, 1200);
      }} else if (timer) {{
        clearInterval(timer);
        timer = null;
      }}
    }});

    canvas.addEventListener('mousemove', (ev) => {{
      const r = canvas.getBoundingClientRect();
      const mx = ev.clientX - r.left;
      const my = ev.clientY - r.top;
      hover = nearestPoint(mx, my);
      showTip(hover, mx, my);
      draw();
    }});
    canvas.addEventListener('mouseleave', () => {{
      hover = null;
      showTip(null, 0, 0);
      draw();
    }});

    window.addEventListener('resize', resize);
    resize();
  </script>
</body>
</html>
"""

    out_html.write_text(html, encoding="utf-8")
    print(f"Wrote {out_html}")


if __name__ == "__main__":
    main()
