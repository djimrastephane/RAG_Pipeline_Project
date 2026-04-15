from __future__ import annotations

import base64
import ast
import hashlib
import json
import re
import sys
import time
from html import escape
from pathlib import Path
from urllib.parse import quote
import requests
import streamlit as st
import streamlit.components.v1 as components
import os
import pandas as pd
import numpy as np

try:
    from app.ui import _matplotlib_env  # noqa: F401
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from app.ui import _matplotlib_env  # noqa: F401

try:
    from app.ui.components import results_to_dataframe
    from app.ui.ui_artifacts import (
        artifact_state,
        load_csv,
        metrics_by_k_from_metrics,
        project_root,
        retrieval_metrics_path,
        run_info_from_metrics,
    )
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from app.ui.components import results_to_dataframe
    from app.ui.ui_artifacts import (
        artifact_state,
        load_csv,
        metrics_by_k_from_metrics,
        project_root,
        retrieval_metrics_path,
        run_info_from_metrics,
    )


API_BASE = st.sidebar.text_input("API Base URL", value="http://localhost:8000")
DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Manrope:wght@400;500;600;700&display=swap');
      :root {
        --bg-0: #07121c;
        --bg-1: #0b1827;
        --bg-2: #112136;
        --panel: rgba(13, 24, 40, 0.78);
        --panel-strong: rgba(16, 29, 46, 0.95);
        --line: rgba(110, 151, 205, 0.24);
        --line-strong: rgba(116, 172, 232, 0.42);
        --text: #e7f0ff;
        --muted: #9eb4d2;
        --accent: #2bb3c8;
        --accent-soft: rgba(43, 179, 200, 0.28);
        --ok: #6ce3a5;
        --bad: #ff8c8c;
      }
      html, body, [class*="css"] {
        font-family: "Manrope", "Space Grotesk", "Avenir Next", "Segoe UI", sans-serif !important;
      }
      .stApp {
        background:
          radial-gradient(1200px 420px at 12% -8%, rgba(43,179,200,0.2), transparent 55%),
          radial-gradient(900px 360px at 88% -10%, rgba(39,121,204,0.15), transparent 55%),
          linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 48%, var(--bg-2) 100%);
        color: var(--text);
      }
      [data-testid="stHeader"] {
        background: transparent;
      }
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(11, 21, 34, 0.95), rgba(10, 18, 30, 0.95));
        border-right: 1px solid var(--line);
      }
      h1, h2, h3 {
        font-family: "Space Grotesk", "Manrope", sans-serif !important;
        letter-spacing: 0.01em;
      }
      [data-testid="stAppViewContainer"] .main {
        animation: app-fade-in 320ms ease-out;
      }
      @keyframes app-fade-in {
        from { opacity: 0; transform: translateY(4px); }
        to { opacity: 1; transform: translateY(0); }
      }
      .hero {
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.95rem 1rem 1.05rem 1rem;
        margin: 0.15rem 0 0.9rem 0;
        background:
          linear-gradient(120deg, rgba(25, 52, 78, 0.85), rgba(17, 35, 58, 0.9)),
          linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0));
        box-shadow: 0 12px 30px rgba(3, 8, 15, 0.35), inset 0 1px 0 rgba(255,255,255,0.05);
      }
      .hero-kicker {
        color: var(--muted);
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        margin-bottom: 0.2rem;
      }
      .hero-title {
        color: var(--text);
        font-size: 1.44rem;
        font-family: "Space Grotesk", "Manrope", sans-serif;
        font-weight: 700;
        line-height: 1.18;
      }
      .hero-subtitle {
        color: #b8c9e2;
        font-size: 0.9rem;
        margin-top: 0.22rem;
      }
      .doc-context {
        border: 1px solid var(--line);
        border-radius: 12px;
        background: linear-gradient(180deg, rgba(15, 28, 45, 0.88) 0%, rgba(12, 23, 38, 0.9) 100%);
        padding: 0.62rem 0.8rem;
        margin: 0.35rem 0 0.85rem 0;
        color: #c9d9f2;
        font-size: 0.84rem;
      }
      .doc-section-title {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.11em;
        color: var(--muted);
        margin: 0.42rem 0 0.43rem 0;
      }
      .doc-stat-card {
        background: linear-gradient(180deg, var(--panel-strong) 0%, rgba(14, 25, 40, 0.98) 100%);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 0.72rem 0.78rem 0.75rem 0.78rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.03), 0 6px 18px rgba(4,10,18,0.24);
        min-height: 86px;
      }
      .doc-stat-label {
        font-size: 0.72rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 0.2rem;
      }
      .doc-stat-value {
        font-size: 1.48rem;
        line-height: 1.1;
        font-weight: 700;
        color: var(--text);
      }
      .doc-pill {
        display: inline-block;
        margin-top: 0.18rem;
        padding: 0.2rem 0.56rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
      }
      .doc-pill-ok {
        color: #d5ffe7;
        background: rgba(39, 130, 82, 0.35);
        border: 1px solid rgba(104, 227, 163, 0.45);
      }
      .doc-pill-miss {
        color: #ffe0e0;
        background: rgba(156, 52, 52, 0.3);
        border: 1px solid rgba(239, 120, 120, 0.43);
      }
      .status-banner {
        border-radius: 11px;
        padding: 0.62rem 0.82rem;
        font-weight: 600;
        margin: 0.45rem 0 0.7rem 0;
        border: 1px solid var(--line);
      }
      .status-ok {
        color: #ceffe3;
        background: linear-gradient(180deg, rgba(33, 116, 75, 0.38) 0%, rgba(21, 80, 52, 0.35) 100%);
        border-color: rgba(97, 214, 156, 0.58);
      }
      .status-miss {
        color: #ffe2e2;
        background: linear-gradient(180deg, rgba(143, 58, 58, 0.35) 0%, rgba(110, 43, 43, 0.35) 100%);
        border-color: rgba(229, 123, 123, 0.48);
      }
      .answer-card {
        border: 1px solid var(--line);
        border-radius: 12px;
        background: linear-gradient(180deg, rgba(17, 30, 47, 0.82), rgba(15, 25, 40, 0.86));
        padding: 0.68rem 0.82rem;
        margin-bottom: 0.55rem;
      }
      .answer-k {
        color: var(--muted);
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.075em;
        margin-bottom: 0.18rem;
      }
      .answer-v {
        color: #f2f8ff;
        font-size: 0.94rem;
      }
      .stButton > button {
        border-radius: 10px !important;
        border: 1px solid var(--line-strong) !important;
        background: linear-gradient(180deg, rgba(35, 97, 145, 0.45), rgba(24, 66, 103, 0.55)) !important;
        color: #ebf5ff !important;
        font-weight: 600 !important;
        transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease !important;
      }
      .stButton > button:hover {
        border-color: rgba(85, 196, 222, 0.68) !important;
        box-shadow: 0 0 0 2px rgba(43, 179, 200, 0.18), 0 8px 18px rgba(3, 12, 23, 0.35) !important;
        transform: translateY(-1px);
      }
      .stTextArea textarea, .stSelectbox [data-baseweb="select"] > div {
        border-radius: 10px !important;
        border: 1px solid var(--line) !important;
        background: rgba(16, 28, 44, 0.78) !important;
      }
      .stDataFrame {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
      }
      .chunk-card {
        border: 1px solid var(--line);
        border-radius: 12px;
        background: linear-gradient(180deg, rgba(17, 30, 47, 0.82), rgba(15, 25, 40, 0.86));
        padding: 0.75rem 0.85rem;
        margin-bottom: 0.75rem;
      }
      .paper-panel {
        border: 1px solid rgba(208, 221, 240, 0.55);
        border-radius: 10px;
        background: #f8fbff;
        color: #162232;
        padding: 0.95rem 1rem;
        margin: 0.4rem 0 1rem 0;
        box-shadow: 0 10px 22px rgba(4, 10, 18, 0.18);
      }
      .paper-panel-title {
        font-family: "Space Grotesk", "Manrope", sans-serif;
        font-size: 0.92rem;
        font-weight: 700;
        color: #132235;
        margin-bottom: 0.45rem;
      }
      .paper-context-box {
        border: 1px solid rgba(27, 41, 58, 0.45);
        background: #ffffff;
        border-radius: 8px;
        padding: 0.7rem 0.8rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 0.82rem;
        line-height: 1.48;
        color: #1d2b3a;
        white-space: pre-wrap;
      }
      .paper-answer-box {
        display: inline-block;
        min-width: 280px;
        border: 1px solid rgba(27, 41, 58, 0.45);
        background: #ffffff;
        border-radius: 8px;
        padding: 0.5rem 0.65rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 0.82rem;
        color: #1d2b3a;
      }
      .paper-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #22364d;
        margin: 0 0 0.22rem 0;
      }
      .paper-caption {
        margin-top: 0.62rem;
        color: #33465e;
        font-size: 0.78rem;
        line-height: 1.45;
      }
      .paper-doc-hit {
        font-weight: 700;
      }
      .paper-status {
        margin-top: 0.62rem;
        padding: 0.48rem 0.6rem;
        border-radius: 8px;
        font-size: 0.78rem;
        line-height: 1.4;
      }
      .paper-status-warn {
        background: rgba(140, 71, 71, 0.12);
        border: 1px solid rgba(145, 67, 67, 0.28);
        color: #6d2f2f;
      }
      .paper-status-ok {
        background: rgba(66, 125, 91, 0.13);
        border: 1px solid rgba(66, 125, 91, 0.28);
        color: #25543a;
      }
      .chunk-meta {
        color: var(--muted);
        font-size: 0.78rem;
        margin-bottom: 0.45rem;
      }
      .token-stream {
        border: 1px solid rgba(116, 172, 232, 0.22);
        border-radius: 10px;
        background: rgba(8, 17, 29, 0.72);
        padding: 0.55rem 0.6rem;
        line-height: 2;
        word-break: break-word;
      }
      .tok {
        display: inline;
        border-radius: 5px;
        padding: 0.08rem 0.02rem;
      }
      .tok-overlap-prev {
        background: rgba(108, 227, 165, 0.24);
        box-shadow: inset 0 0 0 1px rgba(108, 227, 165, 0.22);
      }
      .tok-overlap-next {
        background: rgba(43, 179, 200, 0.22);
        box-shadow: inset 0 0 0 1px rgba(43, 179, 200, 0.2);
      }
      .tok-legend {
        display: inline-block;
        margin-right: 0.55rem;
        padding: 0.16rem 0.45rem;
        border-radius: 999px;
        font-size: 0.76rem;
        border: 1px solid var(--line);
      }
      .tok-legend-prev {
        background: rgba(108, 227, 165, 0.2);
      }
      .tok-legend-next {
        background: rgba(43, 179, 200, 0.2);
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _stat_card(label: str, value: str) -> str:
    """Render one styled stat card for document metadata."""
    return (
        f"<div class='doc-stat-card'>"
        f"<div class='doc-stat-label'>{label}</div>"
        f"<div class='doc-stat-value'>{value}</div>"
        f"</div>"
    )


def _status_card(label: str, ok: bool) -> str:
    """Render one styled status card with availability pill."""
    cls = "doc-pill-ok" if ok else "doc-pill-miss"
    txt = "Available" if ok else "Missing"
    return (
        f"<div class='doc-stat-card'>"
        f"<div class='doc-stat-label'>{label}</div>"
        f"<span class='doc-pill {cls}'>{txt}</span>"
        f"</div>"
    )


def _short_chunk_name(chunk_id: str) -> str:
    """Return compact chunk identifier for labels while keeping full id elsewhere."""
    text = str(chunk_id or "")
    if ":" in text:
        return text.split(":", 1)[1]
    return text


def _format_pages(pages: object) -> str:
    """Render page list in compact display format."""
    if isinstance(pages, list):
        return ", ".join(str(p) for p in pages)
    return str(pages or "")


def _is_markdown_alignment_row(cells: list[str]) -> bool:
    """Return True when a markdown row is only alignment markers like '---' or ':---:'."""
    non_empty = [c.strip() for c in cells if str(c).strip()]
    if not non_empty:
        return False
    return all(bool(re.fullmatch(r":?-{3,}:?", cell)) for cell in non_empty)


def _parse_markdown_table_to_df(markdown_text: str) -> pd.DataFrame:
    """Parse markdown table text into a DataFrame for readable preview and CSV export."""
    rows: list[list[str]] = []
    for raw_line in str(markdown_text or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or line.count("|") < 2:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)

    if not rows:
        return pd.DataFrame()

    max_cols = max(len(r) for r in rows)
    rows = [r + [""] * (max_cols - len(r)) for r in rows]
    rows = [r for r in rows if not _is_markdown_alignment_row(r)]
    if not rows:
        return pd.DataFrame()

    header = rows[0]
    non_empty_header = [h for h in header if str(h).strip()]
    numeric_like_header = all(bool(re.fullmatch(r"\d+", str(h).strip())) for h in non_empty_header) if non_empty_header else False
    use_generic_header = (not non_empty_header) or numeric_like_header or len(non_empty_header) <= 1

    if use_generic_header:
        columns = [f"col_{i + 1}" for i in range(max_cols)]
        data_rows = rows
    else:
        columns = []
        seen: dict[str, int] = {}
        for idx, cell in enumerate(header):
            base = str(cell).strip() or f"col_{idx + 1}"
            if base in seen:
                seen[base] += 1
                columns.append(f"{base}_{seen[base]}")
            else:
                seen[base] = 1
                columns.append(base)
        data_rows = rows[1:]

    df = pd.DataFrame(data_rows, columns=columns)
    if df.empty:
        return df
    non_empty_mask = ~df.apply(lambda r: all(str(v).strip() == "" for v in r), axis=1)
    return df.loc[non_empty_mask].reset_index(drop=True)


def _render_token_sequence_html(
    tokens: list[object],
    *,
    prefix_overlap: int = 0,
    suffix_overlap: int = 0,
) -> str:
    """Render a token stream with overlap bands highlighted."""
    spans: list[str] = []
    total = len(tokens)
    prefix_overlap = max(0, min(int(prefix_overlap), total))
    suffix_overlap = max(0, min(int(suffix_overlap), max(0, total - prefix_overlap)))
    suffix_start = total - suffix_overlap
    for idx, token in enumerate(tokens):
        cls = "tok"
        if idx < prefix_overlap:
            cls += " tok-overlap-prev"
        elif idx >= suffix_start and idx < total:
            cls += " tok-overlap-next"
        spans.append(f"<span class='{cls}'>{escape(str(token)).replace(' ', '&nbsp;')}</span>")
    return "<div class='token-stream'>" + "".join(spans) + "</div>"


def _truncate_paper_text(text: object, limit: int = 220) -> str:
    """Trim long snippet text for the paper-style preview panel."""
    raw = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(raw) <= limit:
        return raw
    return raw[: max(0, limit - 3)].rstrip() + "..."


def _build_paper_style_prompt_panel(
    *,
    question: str,
    expected_answer: str,
    results: list[dict],
    retrieved_answer: str = "",
) -> str:
    """Render a paper-style prompt/answer example panel from current query context."""
    blocks = [
        "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).",
        "",
    ]
    expected_norm = str(expected_answer or "").strip().lower()
    for idx, result in enumerate(results[:3], start=1):
        snippet = _truncate_paper_text(result.get("chunk_text") or result.get("snippet") or "")
        title_bits = []
        section = str(result.get("section_title") or "").strip()
        subsection = str(result.get("subsection_title") or "").strip()
        if section:
            title_bits.append(section)
        if subsection and subsection.lower() != section.lower():
            title_bits.append(subsection)
        title_text = " / ".join(title_bits) if title_bits else _short_chunk_name(str(result.get("chunk_id") or f"chunk_{idx}"))
        line = f"Document [{idx}] [Title: {title_text}] {snippet}"
        if expected_norm and expected_norm in str(snippet).lower():
            line = f"<span class='paper-doc-hit'>{escape(line)}</span>"
        else:
            line = escape(line)
        blocks.append(line)
    blocks.extend(
        [
            "",
            f"Question: {escape(str(question or '').strip())}",
            "Answer:",
        ]
    )
    context_html = "<br>".join(blocks)
    answer_text = escape(str(expected_answer or "").strip()) or "Not available for the selected query."
    retrieved_text = str(retrieved_answer or "").strip()
    status_html = ""
    if not results:
        status_html = (
            "<div class='paper-status paper-status-warn'>"
            "Run Search to populate the input context from live retrieval results."
            "</div>"
        )
    elif expected_answer and retrieved_text:
        expected_norm = re.sub(r"\s+", " ", str(expected_answer).strip().lower())
        retrieved_norm = re.sub(r"\s+", " ", retrieved_text.lower())
        if expected_norm and expected_norm in retrieved_norm:
            status_html = (
                "<div class='paper-status paper-status-ok'>"
                "Live retrieval answer matches the selected gold answer."
                "</div>"
            )
        else:
            status_html = (
                "<div class='paper-status paper-status-warn'>"
                f"Live retrieval currently disagrees with the selected gold answer. "
                f"Retrieved answer: <strong>{escape(retrieved_text)}</strong>"
                "</div>"
            )
    return f"""
    <div class="paper-panel">
      <div class="paper-panel-title">Paper-Style Prompt Example</div>
      <div class="paper-label">Input Context</div>
      <div class="paper-context-box">{context_html}</div>
      <div style="margin-top:0.7rem;">
        <div class="paper-label">Desired Answer</div>
        <div class="paper-answer-box">{answer_text}</div>
      </div>
      {status_html}
      <div class="paper-caption">
        Illustration only. This panel mirrors the paper-style multi-document QA prompt format using the current query and retrieved context.
      </div>
    </div>
    """


def _load_page_chunk_inspector(api_base: str, doc_id: str, page_no: int) -> dict:
    """Fetch page chunk-inspection payload from the API."""
    r = requests.get(f"{str(api_base).rstrip('/')}/api/v1/docs/{doc_id}/pages/{int(page_no)}/chunks", timeout=120)
    r.raise_for_status()
    return r.json()


def api_get(path: str) -> dict:
    """Execute GET request against API and return JSON body."""
    r = requests.get(f"{API_BASE}{path}", timeout=120)
    r.raise_for_status()
    return r.json()


def api_post_json(path: str, payload: dict) -> dict:
    """Execute JSON POST request against API and return JSON body."""
    r = requests.post(f"{API_BASE}{path}", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def api_post_upload(path: str, pdf_file, eval_file=None) -> dict:
    """Upload PDF (+ optional eval set) to API using JSON/base64 payload."""
    payload = {
        "pdf_filename": pdf_file.name,
        "pdf_base64": base64.b64encode(pdf_file.getvalue()).decode("utf-8"),
    }
    if eval_file is not None:
        payload["eval_filename"] = eval_file.name
        payload["eval_base64"] = base64.b64encode(eval_file.getvalue()).decode("utf-8")
    r = requests.post(f"{API_BASE}{path}", json=payload, timeout=1800)
    r.raise_for_status()
    return r.json()

def _pipeline_demo_html() -> str:
    """Load static pipeline architecture demo HTML for embedding in Streamlit."""
    html_path = Path(__file__).resolve().parent / "pipeline_demo.html"
    if not html_path.exists():
        return (
            "<div style='padding:1rem;border:1px solid #444;border-radius:8px;'>"
            "Pipeline demo file not found: app/ui/pipeline_demo.html"
            "</div>"
        )
    return html_path.read_text(encoding="utf-8")


st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Retrieval Reliability Workbench</div>
      <div class="hero-title">RAG Retrieval Debug UI</div>
      <div class="hero-subtitle">
        Upload and inspect processed reports, run controlled retrieval checks, and diagnose ranking behavior.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

def _parse_pages_value(v: object) -> list[int]:
    if isinstance(v, list):
        out: list[int] = []
        for x in v:
            try:
                out.append(int(x))
            except Exception:
                continue
        return out
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        try:
            j = json.loads(s)
            if isinstance(j, list):
                return [int(x) for x in j if str(x).strip().isdigit()]
        except Exception:
            pass
        try:
            j = ast.literal_eval(s)
            if isinstance(j, list):
                return [int(x) for x in j if str(x).strip().isdigit()]
        except Exception:
            pass
        return [int(x) for x in re.findall(r"\d+", s)]
    return []


def _load_artifacts(data_root: Path, results_dir: Path, doc_id: str) -> dict:
    metrics_path = retrieval_metrics_path(data_root, doc_id) if doc_id else Path("")
    failure_audit_path = _resolve_failure_audit_path(results_dir, doc_id)
    return {
        "run_info": run_info_from_metrics(data_root, doc_id) if doc_id else {},
        "metrics_by_k": metrics_by_k_from_metrics(data_root, doc_id) if doc_id else {},
        "metrics_state": artifact_state(metrics_path) if doc_id else {},
        "failure_audit_path": str(failure_audit_path) if failure_audit_path else "",
        "failure_audit": load_csv(str(failure_audit_path)) if failure_audit_path else pd.DataFrame(),
        "fp2": load_csv(str(data_root / "fp2_audit_last4.csv")),
        "fp2_classified": load_csv(str(data_root / "fp2_audit_last4_classified.csv")),
    }


@st.cache_data(show_spinner=False)
def _load_embedding_artifacts(data_root_str: str, doc_id: str) -> dict:
    data_root = Path(data_root_str)
    doc_dir = data_root / str(doc_id)
    emb_path = doc_dir / "embeddings.npy"
    meta_path = doc_dir / "chunk_meta.parquet"
    if not emb_path.exists():
        return {"ok": False, "error": f"Missing embeddings file: {emb_path}"}
    if not meta_path.exists():
        return {"ok": False, "error": f"Missing metadata file: {meta_path}"}

    emb = np.load(emb_path).astype(np.float32, copy=False)
    cols = ["chunk_id", "chunk_id_global", "section_title", "subsection_title", "is_table", "page_start"]
    meta = pd.read_parquet(meta_path)
    present = [c for c in cols if c in meta.columns]
    meta = meta[present].copy()
    n = min(len(meta), int(emb.shape[0]))
    if n <= 1:
        return {"ok": False, "error": f"Not enough vectors to visualize for {doc_id}."}
    emb = emb[:n]
    meta = meta.iloc[:n].reset_index(drop=True)
    return {"ok": True, "emb": emb, "meta": meta}


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


@st.cache_resource(show_spinner=False)
def _load_query_encoder(model_path_str: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_path_str)


@st.cache_resource(show_spinner=False)
def _fit_wizmap_projection_model(data_root_str: str, doc_id: str, method: str, seed: int):
    live = _load_embedding_artifacts(str(data_root_str), str(doc_id))
    if not bool(live.get("ok")):
        raise ValueError(str(live.get("error") or "Could not load embedding artifacts."))
    emb = np.asarray(live["emb"], dtype=np.float32)

    method_norm = str(method).strip().lower()
    if method_norm == "pca":
        from sklearn.decomposition import PCA

        model = PCA(n_components=2, random_state=int(seed))
        model.fit(emb)
        return model
    if method_norm == "umap":
        import umap  # type: ignore

        model = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=int(seed),
        )
        model.fit(emb)
        return model
    raise ValueError(f"Query projection is not supported for method: {method}")


def _compute_query_projection(
    data_root_str: str,
    doc_id: str,
    question: str,
    source_csv: Path | None = None,
    method: str = "umap",
    seed: int = 42,
) -> dict | None:
    text = str(question or "").strip()
    if not text:
        return None

    model_path = project_root() / "models" / "all-MiniLM-L6-v2"
    if not model_path.exists():
        raise ValueError(f"Embedding model not found: {model_path}")

    encoder = _load_query_encoder(str(model_path))
    query_emb = encoder.encode([text], convert_to_numpy=True, normalize_embeddings=False).astype("float32")
    query_emb = _l2_normalize(query_emb).astype("float32")

    try:
        reducer = _fit_wizmap_projection_model(str(data_root_str), str(doc_id), method=method, seed=seed)
        xy = np.asarray(reducer.transform(query_emb), dtype=np.float32)
        if xy.ndim != 2 or xy.shape[0] != 1 or xy.shape[1] != 2:
            raise ValueError("Query projection did not return a 2D coordinate.")
        coord_x = float(xy[0, 0])
        coord_y = float(xy[0, 1])
        projection_method = "umap_transform"
    except Exception:
        if source_csv is None or not source_csv.exists():
            raise
        live = _load_embedding_artifacts(str(data_root_str), str(doc_id))
        if not bool(live.get("ok")):
            raise ValueError(str(live.get("error") or "Could not load embedding artifacts."))
        meta = pd.DataFrame(live["meta"]).copy()
        emb = np.asarray(live["emb"], dtype=np.float32)
        src_df = pd.read_csv(source_csv)
        if "id" not in src_df.columns or "x" not in src_df.columns or "y" not in src_df.columns:
            raise ValueError("Source CSV missing id/x/y required for query fallback projection.")

        id_series = None
        for candidate in ("chunk_id_global", "chunk_id"):
            if candidate in meta.columns:
                id_series = meta[candidate].astype(str)
                break
        if id_series is None:
            raise ValueError("Chunk metadata missing chunk id columns for query fallback projection.")

        coord_map = {
            str(row.id): (float(row.x), float(row.y))
            for row in src_df.itertuples(index=False)
            if pd.notna(getattr(row, "x", np.nan)) and pd.notna(getattr(row, "y", np.nan))
        }
        matched_idx = [i for i, cid in enumerate(id_series.tolist()) if cid in coord_map]
        if not matched_idx:
            raise ValueError("Could not align source CSV ids with document embeddings for query fallback projection.")

        matched_emb = emb[matched_idx]
        matched_ids = [id_series.iloc[i] for i in matched_idx]
        sims = np.asarray(matched_emb @ query_emb[0], dtype=np.float32)
        top_k = min(12, int(sims.shape[0]))
        order = np.argsort(-sims)[:top_k]
        top_sims = sims[order]
        top_ids = [matched_ids[i] for i in order]
        top_xy = np.asarray([coord_map[cid] for cid in top_ids], dtype=np.float32)
        weights = np.exp(top_sims - float(np.max(top_sims)))
        weights_sum = float(np.sum(weights))
        if weights_sum <= 0:
            raise ValueError("Invalid neighbor weights for query fallback projection.")
        weights = weights / weights_sum
        coord_x = float(np.sum(top_xy[:, 0] * weights))
        coord_y = float(np.sum(top_xy[:, 1] * weights))
        projection_method = "neighbor_weighted_fallback"

    return {
        "id": f"query::{doc_id}",
        "x": coord_x,
        "y": coord_y,
        "text": f"QUERY_POINT live query ({projection_method}): {text}",
        "label": "QUERY_POINT",
        "page": "",
        "section": "Live query",
        "category": "Query",
        "highlight": True,
    }


def _wizmap_output_contains_query(output_dir: Path, doc_id: str) -> bool:
    data_path = output_dir / "data.ndjson"
    if not data_path.exists():
        return False
    target = f"query::{doc_id}"
    try:
        with data_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if target in line or "QUERY_POINT live query:" in line:
                    return True
    except Exception:
        return False
    return False


def _coalesce_wizmap_text(*values: object) -> str:
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return ""


@st.cache_data(show_spinner=False)
def _project_embeddings(emb: np.ndarray, method: str, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n = int(emb.shape[0])
    rng = np.random.default_rng(int(seed))
    take_n = min(n, int(max_points))
    take_idx = np.sort(rng.choice(n, size=take_n, replace=False))
    x = emb[take_idx]

    method_norm = str(method).strip().lower()
    if method_norm == "pca":
        from sklearn.decomposition import PCA

        xy = PCA(n_components=2, random_state=int(seed)).fit_transform(x)
    elif method_norm == "t-sne":
        from sklearn.manifold import TSNE

        perp = max(5, min(30, take_n - 1))
        xy = TSNE(
            n_components=2,
            random_state=int(seed),
            init="pca",
            learning_rate="auto",
            perplexity=float(perp),
        ).fit_transform(x)
    elif method_norm == "umap":
        import umap  # type: ignore

        xy = umap.UMAP(n_components=2, random_state=int(seed)).fit_transform(x)
    else:
        raise ValueError(f"Unknown projection method: {method}")
    return take_idx, xy


@st.cache_data(show_spinner=False)
def _sample_cosine_pairs(emb: np.ndarray, sample_size: int, seed: int) -> np.ndarray:
    n = int(emb.shape[0])
    if n < 2:
        return np.array([], dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    m = int(min(sample_size, max(2, n * 20)))
    i = rng.integers(0, n, size=m)
    j = rng.integers(0, n, size=m)
    same = i == j
    while np.any(same):
        j[same] = rng.integers(0, n, size=int(np.sum(same)))
        same = i == j
    return np.sum(emb[i] * emb[j], axis=1)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower()).strip("_")


def _doc_label_for_plot(doc_id: str) -> str:
    m = re.match(r"^([A-Za-z]+)-(\d{4})-(\d{4})$", str(doc_id or "").strip())
    if not m:
        return str(doc_id or "")
    trust = m.group(1)
    y1 = m.group(2)
    y2 = m.group(3)
    return f"NHS {trust} Annual Report ({y1}\u2013{y2})"


def _resolve_failure_audit_path(results_dir: Path, doc_id: str) -> Path | None:
    if not results_dir.exists() or not doc_id:
        return None
    doc_key = _slug(doc_id)
    direct_candidates = [
        results_dir / f"{doc_key}_failure_audit.csv",
        results_dir / f"{doc_id}_failure_audit.csv",
    ]
    for p in direct_candidates:
        if p.exists():
            return p

    pattern_candidates = sorted(results_dir.glob(f"*{doc_key}*failure_audit*.csv"))
    if pattern_candidates:
        return pattern_candidates[0]
    fallback = results_dir / "grampian_2024_2025_failure_audit.csv"
    if fallback.exists():
        return fallback
    return None


def _build_wizmap_url(data_url: str, grid_url: str, state_key: str | None = None) -> str:
    data = str(data_url or "").strip()
    grid = str(grid_url or "").strip()
    if not data or not grid:
        return ""
    url = (
        "https://poloclub.github.io/wizmap/"
        f"?dataURL={quote(data, safe='')}&gridURL={quote(grid, safe='')}"
    )
    if state_key:
        url += f"&v={quote(str(state_key), safe='')}"
    return url


def _doc_default_port(doc_id: str) -> int:
    # Deterministic doc-specific localhost port to avoid stale cross-doc servers.
    key = str(doc_id or "")
    return 8700 + (sum(ord(ch) for ch in key) % 200)


@st.cache_data(show_spinner=False)
def _current_wizmap_id_set(data_root_str: str, doc_id: str) -> set[str]:
    live = _load_embedding_artifacts(str(data_root_str), str(doc_id))
    if not bool(live.get("ok")):
        return set()
    meta = pd.DataFrame(live.get("meta") or [])
    for col in ("chunk_id_global", "chunk_id"):
        if col in meta.columns:
            vals = {
                str(v).strip()
                for v in meta[col].tolist()
                if str(v).strip() and str(v).strip().lower() != "nan"
            }
            if vals:
                return vals
    return set()


@st.cache_data(show_spinner=False)
def _read_wizmap_source_ids(path_str: str) -> list[str]:
    path = Path(path_str)
    try:
        df = pd.read_csv(path, usecols=["id"])
    except Exception:
        try:
            df = pd.read_csv(path)
        except Exception:
            return []
    if "id" not in df.columns:
        return []
    out: list[str] = []
    for raw in df["id"].tolist():
        text = str(raw).strip()
        if not text or text.lower() == "nan" or text.startswith("query::"):
            continue
        out.append(text)
    return out


def _score_wizmap_source_csv(path: Path, current_ids: set[str]) -> tuple[float, int, int, float]:
    source_ids = _read_wizmap_source_ids(str(path))
    if not source_ids or not current_ids:
        return (0.0, 0, 0, float(path.stat().st_mtime if path.exists() else 0.0))
    overlap = sum(1 for cid in source_ids if cid in current_ids)
    ratio = float(overlap) / float(max(1, len(source_ids)))
    return (ratio, overlap, len(source_ids), float(path.stat().st_mtime))


def _resolve_wizmap_source_csv(doc_id: str, data_root_str: str) -> Path | None:
    wiz_root = project_root() / "results" / "wizmap"
    if not wiz_root.exists():
        return None
    current_ids = _current_wizmap_id_set(str(data_root_str), str(doc_id))
    candidates = [
        wiz_root / f"{doc_id}_wizmap_umap.csv",
        wiz_root / f"{_slug(doc_id)}_wizmap_umap.csv",
    ]
    globbed = sorted(wiz_root.glob(f"*{doc_id}*wizmap_umap.csv"))
    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for p in candidates + globbed:
        if p.exists() and str(p) not in seen:
            seen.add(str(p))
            unique_candidates.append(p)
    if not unique_candidates:
        return None
    if not current_ids:
        return unique_candidates[0]
    scored = []
    for idx, p in enumerate(unique_candidates):
        ratio, overlap, total, mtime = _score_wizmap_source_csv(p, current_ids)
        exact_name = 1 if p.name == f"{doc_id}_wizmap_umap.csv" else 0
        scored.append((ratio, overlap, exact_name, mtime, -idx, total, p))
    scored.sort(reverse=True)
    return scored[0][-1]


def _resolve_chunks_for_doc(doc_id: str, data_root_str: str) -> Path | None:
    active_root = Path(str(data_root_str)).expanduser()
    roots = [
        active_root / doc_id,
        project_root() / "data_processed" / doc_id,
        project_root() / "data_variants" / "toc_upgrade_5docs" / doc_id,
        project_root() / "data_variants" / "toc_upgrade_test" / doc_id,
    ]
    for root in roots:
        candidate = root / "chunks.parquet"
        if candidate.exists():
            return candidate
    return None


def _discover_wizmap_dir(doc_id: str) -> Path | None:
    root = project_root() / "results" / "wizmap"
    preferred = [
        root / doc_id / "searchable",
        root / doc_id / "cleanlabels",
        root / doc_id,
    ]
    for p in preferred:
        if (p / "data.ndjson").exists() and (p / "grid.json").exists():
            return p
    return None


def _validate_wizmap_source_csv(doc_id: str, source_csv: Path, data_root_str: str) -> tuple[bool, str]:
    if not source_csv.exists():
        return False, f"Source CSV not found: {source_csv}"
    try:
        df = pd.read_csv(source_csv)
    except Exception as e:
        return False, f"Could not read source CSV: {type(e).__name__}: {e}"
    required = {"x", "y", "id"}
    if not required.issubset(df.columns):
        return False, f"Source CSV missing required columns: {sorted(required)}"

    current_ids = _current_wizmap_id_set(str(data_root_str), str(doc_id))
    if not current_ids:
        return False, f"Current corpus artifacts for {doc_id} are unavailable under {data_root_str}."

    source_ids = [
        str(v).strip()
        for v in df["id"].tolist()
        if str(v).strip() and str(v).strip().lower() != "nan" and not str(v).startswith("query::")
    ]
    if not source_ids:
        return False, "Source CSV does not contain any chunk ids."
    overlap = sum(1 for cid in source_ids if cid in current_ids)
    ratio = float(overlap) / float(max(1, len(source_ids)))
    if overlap == 0:
        return False, (
            f"Source CSV does not match the active corpus for {doc_id}. "
            "No chunk ids overlap with the current embeddings/chunk metadata."
        )
    if ratio < 0.80:
        return False, (
            f"Source CSV appears stale or from a different corpus variant for {doc_id}. "
            f"Matched {overlap}/{len(source_ids)} ids ({ratio:.1%}) against the active corpus."
        )
    return True, f"Validated source CSV against active corpus ids: {overlap}/{len(source_ids)} matched."


def _format_result_card_meta(result: dict) -> str:
    parts: list[str] = []
    dense = result.get("dense_raw_score")
    bm25 = result.get("bm25_raw_score")
    fused = result.get("score")
    parts.append(f"dense={dense}")
    parts.append(f"bm25={bm25}")
    parts.append(f"fused={fused}")
    table_chunk_kind = result.get("table_chunk_kind")
    if table_chunk_kind:
        parts.append(f"table_chunk_kind={table_chunk_kind}")
    row_start_idx = result.get("row_start_idx")
    row_end_idx = result.get("row_end_idx")
    if row_start_idx is not None or row_end_idx is not None:
        parts.append(f"row_range={row_start_idx}..{row_end_idx}")
    return " | ".join(parts)


def _generate_wizmap_files(
    doc_id: str,
    source_csv: Path,
    output_dir: Path,
    include_chunk_text: bool = True,
    use_category_groups: bool = False,
    query_text: str | None = None,
    data_root_str: str | None = None,
) -> tuple[bool, str]:
    try:
        import wizmap  # type: ignore
    except Exception as e:
        return False, f"Missing `wizmap` package: {type(e).__name__}: {e}"

    ok_source, source_msg = _validate_wizmap_source_csv(
        doc_id=str(doc_id),
        source_csv=source_csv,
        data_root_str=str(data_root_str or (project_root() / "data_processed")),
    )
    if not ok_source:
        return False, source_msg

    try:
        df = pd.read_csv(source_csv)
    except Exception as e:
        return False, f"Could not read source CSV: {type(e).__name__}: {e}"

    if "text" not in df.columns:
        df["text"] = ""
    if "id" not in df.columns:
        df["id"] = [f"row_{i}" for i in range(len(df))]
    if "category" not in df.columns:
        df["category"] = "Unknown"
    if "section" not in df.columns:
        df["section"] = ""
    if "page" not in df.columns:
        df["page"] = ""

    query_note = ""
    query_text_clean = str(query_text or "").strip()
    if query_text_clean:
        try:
            query_row = _compute_query_projection(
                data_root_str=str(data_root_str or (project_root() / "data_processed")),
                doc_id=str(doc_id),
                question=query_text_clean,
                source_csv=source_csv,
                method="umap",
                seed=42,
            )
            if query_row is not None:
                df = pd.concat([df, pd.DataFrame([query_row])], ignore_index=True)
                query_note = " Included live query point."
        except Exception as e:
            query_note = f" Query point skipped: {type(e).__name__}: {e}"

    if include_chunk_text:
        chunks_path = _resolve_chunks_for_doc(doc_id, str(data_root_str or (project_root() / "data_processed")))
        if chunks_path is None:
            return False, f"Could not locate chunks.parquet for {doc_id} under the active corpus root."
        try:
            chunks = pd.read_parquet(chunks_path)
        except Exception as e:
            return False, f"Could not read chunk text source: {type(e).__name__}: {e}"

        if "chunk_text" not in chunks.columns:
            return False, f"Chunk source is missing `chunk_text`: {chunks_path}"

        join_col = None
        best_overlap = -1
        source_ids = {str(v).strip() for v in df["id"].tolist() if str(v).strip() and not str(v).startswith("query::")}
        for candidate in ("chunk_id_global", "chunk_id"):
            if candidate not in chunks.columns:
                continue
            chunk_ids = {
                str(v).strip()
                for v in chunks[candidate].tolist()
                if str(v).strip() and str(v).strip().lower() != "nan"
            }
            overlap = len(source_ids.intersection(chunk_ids))
            if overlap > best_overlap:
                best_overlap = overlap
                join_col = candidate
        if not join_col or best_overlap <= 0:
            return False, (
                f"Could not align WIZMAP source ids with chunk text in {chunks_path}. "
                "Regenerate the projection from the active corpus before building searchable WIZMAP files."
            )

        tmp = chunks[[join_col, "chunk_text"]].copy()
        tmp = tmp.rename(columns={join_col: "id", "chunk_text": "chunk_text_full"})
        df = df.merge(tmp, on="id", how="left")
        source_rows = df[~df["id"].astype(str).str.startswith("query::")].copy()
        matched = int(source_rows["chunk_text_full"].notna().sum()) if "chunk_text_full" in source_rows.columns else 0
        expected = int(len(source_rows))
        if matched < expected:
            return False, (
                f"WIZMAP text enrichment only matched {matched}/{expected} source rows against {chunks_path}. "
                "Regenerate the source CSV from the active corpus to avoid stale searchable content."
            )

    xs = pd.to_numeric(df["x"], errors="coerce").fillna(0.0).astype(float).tolist()
    ys = pd.to_numeric(df["y"], errors="coerce").fillna(0.0).astype(float).tolist()

    def _mk_text(r: pd.Series) -> str:
        body = _coalesce_wizmap_text(r.get("chunk_text_full"), r.get("text"), r.get("label"))
        body = re.sub(r"\s+", " ", body).strip()[:1200]
        return (
            f"chunk_id {r.get('id','')} page {r.get('page','')} "
            f"category {r.get('category','')} section {r.get('section','')} text {body}"
        )

    texts = df.apply(_mk_text, axis=1).tolist()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if use_category_groups:
            labels, group_names = pd.factorize(df["category"].fillna("Unknown"))
            data_list = wizmap.generate_data_list(xs=xs, ys=ys, texts=texts, labels=labels.tolist())
            grid_dict = wizmap.generate_grid_dict(
                xs=xs,
                ys=ys,
                texts=texts,
                labels=labels.tolist(),
                group_names=group_names.tolist(),
                embedding_name=f"NHS {doc_id} (WIZMAP)",
                grid_size=200,
                max_zoom_scale=30,
                random_seed=42,
            )
        else:
            data_list = wizmap.generate_data_list(xs=xs, ys=ys, texts=texts)
            grid_dict = wizmap.generate_grid_dict(
                xs=xs,
                ys=ys,
                texts=texts,
                embedding_name=f"NHS {doc_id} (WIZMAP)",
                grid_size=200,
                max_zoom_scale=30,
                random_seed=42,
            )
        wizmap.save_json_files(
            data_list,
            grid_dict,
            output_dir=str(output_dir),
            data_json_name="data.ndjson",
            grid_json_name="grid.json",
        )
        return True, f"{source_msg} Wrote {output_dir / 'data.ndjson'} and {output_dir / 'grid.json'}.{query_note}"
    except Exception as e:
        return False, f"WIZMAP generation failed: {type(e).__name__}: {e}"


DEFAULT_DOCS = ["Grampian-2022-2023", "Grampian-2023-2024", "Grampian-2024-2025"]
SIDEBAR_RESULTS_DIR_DEFAULT = (
    "/Users/djimra/MSc Data Science Jan 2025/Thesis documents/"
    "RAG_Pipeline_Project/results"
)
DEFAULT_RESULTS_DIR = os.getenv("RAG_RESULTS_DIR", SIDEBAR_RESULTS_DIR_DEFAULT)
DATA_ROOT = project_root() / "data_processed"

st.sidebar.markdown("### Demo Controls")
examiner_mode = st.sidebar.toggle("Examiner Mode", value=True)
show_diagnostics = st.sidebar.toggle("Show Diagnostics", value=examiner_mode)
prefer_table_chunks = st.sidebar.toggle("Prefer Table Chunks", value=False)
k = st.sidebar.selectbox("Top-K", options=[1, 3, 5, 10], index=3)
results_dir_input = st.sidebar.text_input("RESULTS_DIR", value=DEFAULT_RESULTS_DIR)
results_dir = Path(results_dir_input)
if not results_dir.exists():
    st.sidebar.error(f"RESULTS_DIR not found: {results_dir}")

doc_options = []
doc_title_by_id: dict[str, str] = {}
try:
    docs_payload = api_get("/api/v1/docs")
    available = [str(x) for x in docs_payload.get("docs", [])]
    docs_detail = docs_payload.get("docs_detail", [])
    if isinstance(docs_detail, list):
        for item in docs_detail:
            if isinstance(item, dict):
                doc_key = str(item.get("doc_id") or "").strip()
                if doc_key:
                    doc_title_by_id[doc_key] = str(item.get("title") or doc_key)
    default_present = [d for d in DEFAULT_DOCS if d in available]
    remaining = [d for d in available if d not in set(default_present)]
    doc_options = default_present + remaining
except Exception as e:
    st.error(f"Could not load docs list: {e}")

doc_id = st.sidebar.selectbox(
    "Document",
    options=doc_options,
    index=0 if doc_options else None,
    format_func=lambda d: f"{d} | {doc_title_by_id.get(d, d)}",
)

# Keep a visible main-panel selector for examiner workflows.
if doc_options:
    st.subheader("2) Select Document")
    current_idx = doc_options.index(doc_id) if doc_id in doc_options else 0
    doc_id = st.selectbox(
        "Document",
        options=doc_options,
        index=current_idx,
        format_func=lambda d: f"{d} | {doc_title_by_id.get(d, d)}",
        key="main_doc_selector",
    )

if "artifacts_cache" not in st.session_state:
    st.session_state["artifacts_cache"] = {}
if st.sidebar.button("Load artifacts", disabled=(not doc_id)):
    st.session_state["artifacts_cache"][doc_id] = _load_artifacts(DATA_ROOT, results_dir, doc_id)
    st.sidebar.success("Artifacts loaded.")

artifacts = st.session_state["artifacts_cache"].get(doc_id, _load_artifacts(DATA_ROOT, results_dir, doc_id) if doc_id else {})

if "last_search" not in st.session_state:
    st.session_state["last_search"] = None
if st.session_state.get("last_search_doc_id") != doc_id:
    st.session_state["last_search"] = None
    st.session_state["last_search_doc_id"] = doc_id

stats = None
eval_items = []
if doc_id:
    try:
        stats = api_get(f"/api/v1/docs/{doc_id}/stats")
        eval_items = api_get(f"/api/v1/docs/{doc_id}/eval-items").get("items", [])
    except Exception as e:
        st.error(f"Failed to load doc stats/eval items: {e}")

if stats:
    st.markdown("<div class='doc-section-title'>Document</div>", unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    d1.markdown(_stat_card("Pages", str(stats["page_count"])), unsafe_allow_html=True)
    d2.markdown(_stat_card("Chunks", str(stats["chunk_count"])), unsafe_allow_html=True)
    d3.markdown(_stat_card("Table Chunks", str(stats["table_chunk_count"])), unsafe_allow_html=True)
    d4.markdown(_stat_card("Extracted Tables", str(stats.get("table_count") or 0)), unsafe_allow_html=True)
    pipe_chunk_size = stats.get("chunk_size_tokens")
    pipe_overlap = stats.get("chunk_overlap_tokens")
    pipe_tokenizer = stats.get("tokenizer_backend") or "unknown"
    pipe_require_tiktoken = stats.get("require_tiktoken")
    pipe_table_chunking = stats.get("table_chunking") or "unknown"
    provenance_bits = [
        f"Chunking: {pipe_chunk_size}/{pipe_overlap}" if pipe_chunk_size and pipe_overlap is not None else "Chunking: n/a",
        f"Table chunking: {pipe_table_chunking}",
        f"Tokenizer: {pipe_tokenizer}",
    ]
    if pipe_require_tiktoken is not None:
        provenance_bits.append(f"Require tiktoken: {'yes' if pipe_require_tiktoken else 'no'}")
    st.caption(" | ".join(provenance_bits))
    s1, s2, s3 = st.columns(3)
    s1.markdown(_status_card("Eval Set", bool(stats.get("has_eval_set", False))), unsafe_allow_html=True)
    s2.markdown(_status_card("Pipeline Log", bool(stats.get("has_pipeline_log", False))), unsafe_allow_html=True)
    s3.markdown(_status_card("Structured Tables", bool(stats.get("has_tables_structured", False))), unsafe_allow_html=True)

tabs = st.tabs(
    [
        "Retrieval",
        "Tables",
        "Chunk Inspector",
        "Failure Audit",
        "Embedding Diagnostics",
        "Run Info",
        "Pipeline Architecture",
        "System Metrics",
    ]
)

with tabs[0]:
    st.subheader("Retrieval")
    demo_query_map = {
        "Q_EFF_2024_01": "How many milestones were significantly delayed in Q4?",
        "Q_EFF_2024_02": "What proportion of deliverables were completed?",
        "Q_EFF_2024_03": "How many milestones were on track in Q1 (June 2024)?",
    }
    selected_demo_qid = st.selectbox(
        "Demo queries",
        options=[""] + list(demo_query_map.keys()),
        format_func=lambda q: f"{q}: {demo_query_map[q]}" if q else "(none)",
    )
    default_q = demo_query_map.get(selected_demo_qid, eval_items[0]["question"] if eval_items else "")
    question = st.text_area("Question", value=default_q, height=100)
    query_id_options = [""] + [str(i.get("query_id", "")) for i in eval_items if i.get("query_id")]
    default_qid_idx = 0
    if selected_demo_qid and selected_demo_qid in query_id_options:
        default_qid_idx = query_id_options.index(selected_demo_qid)
    query_id = st.selectbox("Optional query_id", options=query_id_options, index=default_qid_idx)
    selected_eval_item = next((item for item in eval_items if str(item.get("query_id") or "") == str(query_id or "")), None)
    if selected_eval_item is None and question.strip():
        selected_eval_item = next((item for item in eval_items if str(item.get("question") or "").strip() == question.strip()), None)
    last = st.session_state.get("last_search")
    panel_results = []
    panel_retrieved_answer = ""
    if isinstance(last, dict) and str(last.get("question") or "").strip() == question.strip():
        panel_results = list(last.get("results") or [])
        panel_retrieved_answer = str(
            last.get("generated_answer")
            or last.get("predicted_answer")
            or ""
        ).strip()
    panel_expected_answer = ""
    if isinstance(selected_eval_item, dict):
        panel_expected_answer = str(selected_eval_item.get("expected_answer") or "").strip()
    st.markdown(
        _build_paper_style_prompt_panel(
            question=question,
            expected_answer=panel_expected_answer,
            results=panel_results,
            retrieved_answer=panel_retrieved_answer,
        ),
        unsafe_allow_html=True,
    )
    include_generated_answer = st.toggle("Generate answer (Local LLM)", value=False)
    with st.expander("Generation context controls (live)", expanded=False):
        gen_max_context_chunks = st.slider("Max context chunks", 1, 20, 5, 1)
        gen_max_context_chars = st.slider("Max context chars", 1000, 20000, 9000, 500)
        gen_max_chunk_chars = st.slider("Max chars per chunk", 200, 4000, 2200, 100)
        gen_timeout_seconds = st.slider("Generation timeout (sec)", 5, 180, 20, 5)
    with st.expander("Advanced retrieval controls (live search scope + filters)", expanded=False):
        retrieval_scope = st.selectbox(
            "Dense retrieval scope",
            options=["doc", "trust", "global"],
            index=0,
            help=(
                "`doc` searches only the selected document. "
                "`trust` searches all documents from the same NHS trust/board. "
                "`global` searches every indexed document. "
                "Use `trust` or `global` when the answer may live outside the currently selected report."
            ),
        )
        lexical_scope = st.selectbox(
            "BM25 / lexical scope",
            options=["doc", "trust", "global"],
            index=0,
            help=(
                "Scope for the lexical BM25 side of hybrid fusion. "
                "Keeping this aligned with dense scope is the safest default. "
                "Widen it when exact wording may appear in related reports outside the selected document."
            ),
        )
        filter_is_table_label = st.selectbox(
            "Evidence type filter",
            options=["Any", "Only tables", "Only narrative"],
            index=0,
            help="Restrict search to table chunks only, narrative chunks only, or both.",
        )
        filter_year_text = st.text_input(
            "Filter report year",
            value="",
            help="Optional year filter, mainly useful with `trust` or `global` scope.",
        ).strip()
        filter_doc_id = st.text_input(
            "Filter exact doc_id",
            value="",
            help="Optional exact document filter. Most useful when searching in `trust` or `global` scope.",
        ).strip()
        filter_trust_id = st.text_input(
            "Filter exact trust_id",
            value="",
            help="Optional trust/board filter. Most useful with `global` scope.",
        ).strip()
        filter_section_contains = st.text_input(
            "Section title contains",
            value="",
            help="Optional case-insensitive section filter, useful when you know the answer should live in a broad section.",
        ).strip()
        filter_subsection_contains = st.text_input(
            "Subsection title contains",
            value="",
            help="Optional case-insensitive subsection filter, useful for targeted queries against known headings.",
        ).strip()

    if st.button("Run Search", disabled=(not doc_id or not question.strip())):
        try:
            filter_is_table = None
            if filter_is_table_label == "Only tables":
                filter_is_table = True
            elif filter_is_table_label == "Only narrative":
                filter_is_table = False
            filter_year = None
            if filter_year_text:
                if not filter_year_text.isdigit():
                    raise ValueError("Filter year must be a whole number.")
                filter_year = int(filter_year_text)
            payload = {
                "question": question.strip(),
                "k": int(k),
                "query_id": (query_id or None),
                "include_generated_answer": bool(include_generated_answer),
                "retrieval_scope": str(retrieval_scope),
                "lexical_scope": str(lexical_scope),
                "filter_doc_id": (filter_doc_id or None),
                "filter_trust_id": (filter_trust_id or None),
                "filter_year": filter_year,
                "filter_is_table": filter_is_table,
                "filter_section_contains": (filter_section_contains or None),
                "filter_subsection_contains": (filter_subsection_contains or None),
                "gen_max_context_chunks": int(gen_max_context_chunks),
                "gen_max_context_chars": int(gen_max_context_chars),
                "gen_max_chunk_chars": int(gen_max_chunk_chars),
                "gen_timeout_seconds": float(gen_timeout_seconds),
            }
            out = api_post_json(f"/api/v1/docs/{doc_id}/search", payload)
            st.session_state["last_search"] = out
            st.session_state["last_search_doc_id"] = doc_id
        except Exception as e:
            st.error(f"Search failed: {e}")

    last = st.session_state.get("last_search")
    if isinstance(last, dict) and last.get("question"):
        retrieval_cfg = last.get("retrieval_config") if isinstance(last.get("retrieval_config"), dict) else {}
        if retrieval_cfg:
            st.caption(
                "Retrieval config: "
                f"{last.get('retrieval_mode', 'hybrid_rrf_dense_bm25')} | "
                f"rrf_k={retrieval_cfg.get('rrf_k')} | "
                f"dense_w={retrieval_cfg.get('dense_weight')} | "
                f"bm25_w={retrieval_cfg.get('bm25_weight')} | "
                f"subsection_boost={retrieval_cfg.get('enable_subsection_boost')}"
            )
        gen_dbg = last.get("generation_debug") if isinstance(last.get("generation_debug"), dict) else {}
        gen_status = str(last.get("generation_status") or gen_dbg.get("status") or "").strip()
        gen_citations = last.get("generated_citations") if isinstance(last.get("generated_citations"), list) else []
        gen_confidence = last.get("generation_confidence")
        include_generated_answer_last = bool(last.get("include_generated_answer", False))
        generated_answer = str(last.get("generated_answer") or "").strip()
        predicted_answer = str(last.get("predicted_answer") or "").strip()
        answer_source_chunk_id = str(last.get("answer_source_chunk_id") or "").strip()
        if generated_answer:
            inline_citations = ""
            if gen_citations:
                inline_citations = " " + " ".join(
                    f"[{c.get('chunk_id')}, p{c.get('page')}]"
                    for c in gen_citations
                    if c.get("chunk_id") is not None and c.get("page") is not None
                )
            st.markdown("**Generated Answer (Local LLM)**")
            st.write(f"{generated_answer}{inline_citations}")
        elif gen_status == "insufficient_evidence":
            st.info("Generated answer gated: insufficient grounded evidence.")
        if predicted_answer and (not include_generated_answer_last or not generated_answer):
            st.markdown("**Fallback Answer (retrieval-only, no LLM)**")
            st.write(predicted_answer)
            if answer_source_chunk_id:
                st.caption(f"Source chunk: {answer_source_chunk_id}")
        if gen_dbg:
            st.caption(
                "Generation: "
                f"{gen_dbg.get('provider', 'local_ollama')} | "
                f"status={gen_status or gen_dbg.get('status')} | "
                f"model={gen_dbg.get('model')}"
            )
            if gen_dbg.get("error"):
                st.caption(f"Generation error: {gen_dbg.get('error')}")
            if bool(gen_dbg.get("low_retrieval_margin", False)):
                margin = gen_dbg.get("retrieval_margin")
                threshold = gen_dbg.get("retrieval_margin_threshold")
                st.warning(
                    "Low retrieval margin between top-1 and top-2 chunks "
                    f"(margin={margin}, threshold={threshold})."
                )
        if gen_status in {"insufficient_evidence", "error"} and gen_dbg:
            with st.expander("Why Gated / Generation Debug", expanded=False):
                st.write(
                    {
                        "status": gen_status or gen_dbg.get("status"),
                        "citations_parsed": int(gen_dbg.get("citations_parsed", 0) or 0),
                        "citations_valid": int(gen_dbg.get("citations_valid", 0) or 0),
                        "citations_rejected": int(gen_dbg.get("citations_rejected", 0) or 0),
                        "context_chunks_used": int(gen_dbg.get("context_chunks_used", 0) or 0),
                        "context_chars_used": int(gen_dbg.get("context_chars_used", 0) or 0),
                        "context_truncated": bool(gen_dbg.get("context_truncated", False)),
                        "latency_ms": gen_dbg.get("latency_ms"),
                        "retrieval_margin": gen_dbg.get("retrieval_margin"),
                        "retrieval_margin_threshold": gen_dbg.get("retrieval_margin_threshold"),
                        "low_retrieval_margin": bool(gen_dbg.get("low_retrieval_margin", False)),
                    }
                )
        if gen_citations:
            st.caption("Citations: " + ", ".join(f"[{c.get('chunk_id')}, p{c.get('page')}]" for c in gen_citations))
        if gen_confidence is not None:
            st.caption(f"Generation confidence: {float(gen_confidence):.2f}")

        df = results_to_dataframe(last.get("results", []))
        if not df.empty:
            if "fused_score" not in df.columns and "score" in df.columns:
                df["fused_score"] = pd.to_numeric(df["score"], errors="coerce")
            if show_diagnostics and examiner_mode:
                top2 = df.sort_values("rank").head(2)
                if len(top2) == 2:
                    s1 = float(top2.iloc[0].get("fused_score") or 0.0)
                    s2 = float(top2.iloc[1].get("fused_score") or 0.0)
                    c1 = top2.iloc[0].get("dense_raw_score")
                    c2 = top2.iloc[1].get("dense_raw_score")
                    st.markdown("**Ranking margin**")
                    st.write(
                        {
                            "top1_fused": round(s1, 6),
                            "top2_fused": round(s2, 6),
                            "margin": round(s1 - s2, 6),
                            "top1_cosine_est": c1,
                            "top2_cosine_est": c2,
                        }
                    )

            if "is_table" in df.columns:
                df["evidence_layout"] = df["is_table"].apply(lambda x: "table" if bool(x) else "narrative")
            else:
                df["evidence_layout"] = "narrative"

            if prefer_table_chunks and "is_table" in df.columns:
                table_boost = st.slider("Table boost (simulated rerank)", 0.00, 0.10, 0.02, 0.01)
                sim = df.copy()
                sim["fused_score_before"] = pd.to_numeric(sim["fused_score"], errors="coerce").fillna(0.0)
                sim["fused_score_after"] = sim["fused_score_before"] + sim["is_table"].fillna(False).astype(float) * table_boost
                sim["rank_before"] = sim["fused_score_before"].rank(ascending=False, method="min").astype(int)
                sim["rank_after"] = sim["fused_score_after"].rank(ascending=False, method="min").astype(int)
                st.markdown("**Simulated rerank (display only)**")
                st.dataframe(
                    sim[["chunk_id", "is_table", "rank_before", "rank_after", "fused_score_before", "fused_score_after"]]
                    .sort_values("rank_after")
                    .head(int(k)),
                    use_container_width=True,
                    hide_index=True,
                )

            display_cols = [
                "rank",
                "pages",
                "chunk_id",
                "is_table",
                "evidence_layout",
                "table_chunk_kind",
                "row_start_idx",
                "row_end_idx",
                "section_title",
                "subsection_title",
                "dense_raw_score",
                "bm25_raw_score",
                "fused_score",
                "snippet",
            ]
            present_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(df[present_cols], use_container_width=True, hide_index=True)

            st.markdown("**Result cards**")
            for r in last.get("results", []):
                chunk_kind = r.get("table_chunk_kind")
                row_start_idx = r.get("row_start_idx")
                row_end_idx = r.get("row_end_idx")
                kind_bits = []
                if chunk_kind:
                    kind_bits.append(str(chunk_kind))
                if row_start_idx is not None or row_end_idx is not None:
                    kind_bits.append(f"rows {row_start_idx}..{row_end_idx}")
                kind_suffix = f" | {' | '.join(kind_bits)}" if kind_bits else ""
                title = (
                    f"Rank {r.get('rank')} | {_short_chunk_name(str(r.get('chunk_id')))} | "
                    f"p{_format_pages(r.get('pages', []))} | "
                    f"{'table' if bool(r.get('is_table')) else 'narrative'}"
                    f"{kind_suffix}"
                )
                with st.expander(title, expanded=False):
                    st.caption(_format_result_card_meta(r))
                    chunk_text = (r.get("chunk_text", "") or "")
                    preview_chars = 1400 if bool(r.get("is_table")) else 700
                    st.code(chunk_text[:preview_chars], language="text")
                    if len(chunk_text) > preview_chars:
                        show_full = st.checkbox(
                            "Show full chunk text",
                            value=False,
                            key=f"show_full_chunk::{r.get('chunk_id')}::{r.get('rank')}",
                        )
                        if show_full:
                            st.code(chunk_text, language="text")
        else:
            st.info("No results.")

with tabs[1]:
    st.subheader("Tables")
    if doc_id:
        if "tables_cache" not in st.session_state:
            st.session_state["tables_cache"] = {}
        table_cache_key = f"tables::{doc_id}"
        if table_cache_key not in st.session_state["tables_cache"]:
            try:
                st.session_state["tables_cache"][table_cache_key] = api_get(f"/api/v1/docs/{doc_id}/tables?limit=500")
            except Exception as e:
                st.warning(f"Failed to load table list: {e}")
                st.session_state["tables_cache"][table_cache_key] = {"has_tables": False, "items": [], "total": 0}

        tables_payload = st.session_state["tables_cache"].get(table_cache_key, {"has_tables": False, "items": [], "total": 0})
        table_items = tables_payload.get("items", []) if isinstance(tables_payload, dict) else []
        if table_items:
            st.caption(f"{tables_payload.get('total', len(table_items))} table(s) available")
            table_rows = []
            for t in table_items:
                table_rows.append(
                    {
                        "page": t.get("page"),
                        "table_id": t.get("table_id"),
                        "table_type": t.get("table_type"),
                        "rows": t.get("rows"),
                        "cols": t.get("cols"),
                        "method": t.get("extraction_method"),
                    }
                )
            st.dataframe(table_rows, use_container_width=True, hide_index=True)

            selected_table_id = st.selectbox(
                "Inspect table",
                options=[str(t.get("table_id")) for t in table_items if t.get("table_id")],
                index=0,
                key=f"tables_tab_table_id_{doc_id}",
            )
            selected_table = next((t for t in table_items if str(t.get("table_id")) == str(selected_table_id)), None)
            if selected_table:
                st.write(selected_table.get("table_summary") or "N/A")
                md = selected_table.get("table_markdown") or ""
                parsed_df = _parse_markdown_table_to_df(md)
                if not parsed_df.empty:
                    st.dataframe(parsed_df, use_container_width=True, hide_index=True, height=260)
                with st.expander("Raw markdown", expanded=False):
                    st.code(md if md.strip() else "(empty)", language="markdown")
        else:
            st.info("No extracted tables available for this document.")

with tabs[2]:
    st.subheader("Chunk Inspector")
    if not doc_id:
        st.info("Select a document.")
    elif not stats:
        st.info("Document stats unavailable.")
    else:
        max_page = int(stats.get("page_count") or 1)
        page_default = min(max_page, 1 if st.session_state.get(f"chunk_page_{doc_id}") is None else int(st.session_state.get(f"chunk_page_{doc_id}")))
        page_no = st.number_input(
            "Page",
            min_value=1,
            max_value=max_page,
            value=max(1, page_default),
            step=1,
            key=f"chunk_page_{doc_id}",
        )
        try:
            inspector = _load_page_chunk_inspector(API_BASE, str(doc_id), int(page_no))
        except Exception as e:
            st.error(f"Could not load page chunk view: {e}")
            inspector = {}

        if inspector:
            artifact_backend = inspector.get("artifact_tokenizer_backend") or inspector.get("tokenizer_backend") or "unknown"
            artifact_exact = inspector.get("artifact_tokenizer_exact_counting")
            if artifact_exact is None:
                artifact_exact = inspector.get("tokenizer_exact_counting", False)
            inspector_backend = inspector.get("inspector_tokenizer_backend") or inspector.get("tokenizer_backend") or "unknown"
            inspector_exact = inspector.get("inspector_tokenizer_exact_counting")
            if inspector_exact is None:
                inspector_exact = inspector.get("tokenizer_exact_counting", False)
            c0, c1, c2, c3 = st.columns(4)
            c0.metric("Chunks", int(inspector.get("chunk_count", 0) or 0))
            c1.metric("Page Tokens", int(inspector.get("page_token_count", 0) or 0))
            c2.metric("Chunk Size", inspector.get("chunk_size_tokens") or "n/a")
            c3.metric("Overlap", inspector.get("chunk_overlap_tokens") or "n/a")
            st.caption(
                "Artifact tokenizer: "
                f"{artifact_backend} | "
                f"exact_counting={bool(artifact_exact)}"
            )
            st.caption(
                "Inspector tokenizer: "
                f"{inspector_backend} | "
                f"exact_counting={bool(inspector_exact)} | "
                f"segment_aware={bool(inspector.get('segment_aware_chunking', False))}"
            )
            st.markdown(
                "<span class='tok-legend tok-legend-prev'>prefix shared with previous chunk</span>"
                "<span class='tok-legend tok-legend-next'>suffix shared with next chunk</span>",
                unsafe_allow_html=True,
            )
            with st.expander("Page text", expanded=False):
                st.text(inspector.get("page_text") or "")

            for idx, chunk in enumerate(inspector.get("chunks", []), start=1):
                label = "table" if bool(chunk.get("is_table")) else "text"
                meta = (
                    f"Chunk {idx} | {chunk.get('chunk_id') or chunk.get('chunk_id_global') or 'unknown'} | "
                    f"{label} | tokens {chunk.get('token_start', 0)}-{chunk.get('token_end', 0)}"
                )
                extra = (
                    f"prev_overlap={int(chunk.get('overlap_prev_tokens', 0) or 0)} | "
                    f"next_overlap={int(chunk.get('overlap_next_tokens', 0) or 0)} | "
                    f"segment={chunk.get('segment_title') or 'n/a'}"
                )
                if chunk.get("section_title"):
                    extra += f" | section={chunk.get('section_title')}"
                if chunk.get("subsection_title"):
                    extra += f" | subsection={chunk.get('subsection_title')}"
                st.markdown(
                    "<div class='chunk-card'>"
                    f"<div><strong>{escape(meta)}</strong></div>"
                    f"<div class='chunk-meta'>{escape(extra)}</div>"
                    f"{_render_token_sequence_html(chunk.get('tokens', []), prefix_overlap=int(chunk.get('overlap_prev_tokens', 0) or 0), suffix_overlap=int(chunk.get('overlap_next_tokens', 0) or 0))}"
                    "</div>",
                    unsafe_allow_html=True,
                )

with tabs[3]:
    st.subheader("Failure Audit")
    fa_df = artifacts.get("failure_audit", pd.DataFrame())
    fa_path = artifacts.get("failure_audit_path", "")
    if fa_df.empty:
        st.warning(f"Missing or empty failure-audit CSV for `{doc_id}` under `{results_dir}`.")
    else:
        if "doc_id" in fa_df.columns and doc_id:
            filtered = fa_df[fa_df["doc_id"].astype(str) == str(doc_id)]
            if not filtered.empty:
                fa_df = filtered
        if fa_path:
            st.caption(f"Loaded: {fa_path}")
        qids = ["all"] + sorted([str(x) for x in fa_df["query_id"].dropna().unique().tolist()])
        selected_qid = st.selectbox("query_id", options=qids, index=0)
        expected_page_filter = st.text_input("expected_page (optional integer)")
        only_missing = st.checkbox("Show only rows where expected page is missing from top-10", value=False)

        view = fa_df.copy()
        if selected_qid != "all":
            view = view[view["query_id"].astype(str) == selected_qid]
        if expected_page_filter.strip().isdigit() and "expected_pages" in view.columns:
            p = int(expected_page_filter.strip())
            view = view[view["expected_pages"].astype(str).str.contains(rf"\\b{p}\\b", regex=True)]
        if only_missing and {"expected_pages", "pages"}.issubset(set(view.columns)):
            def _miss(row: pd.Series) -> bool:
                exp = set(_parse_pages_value(row.get("expected_pages")))
                got = set(_parse_pages_value(row.get("pages")))
                return bool(exp) and bool(exp.intersection(got) == set())

            view = view[view.apply(_miss, axis=1)]

        cols = [
            "query_id",
            "rank",
            "pages",
            "chunk_id",
            "is_table",
            "score",
            "snippet",
            "expected_pages",
        ]
        st.dataframe(view[[c for c in cols if c in view.columns]], use_container_width=True, hide_index=True)

    fp2c = artifacts.get("fp2_classified", pd.DataFrame())
    if not fp2c.empty:
        st.markdown("**FP2 classified summary**")
        if "fp2_bucket" in fp2c.columns:
            st.dataframe(
                fp2c.groupby("fp2_bucket").size().reset_index(name="count").sort_values("count", ascending=False),
                use_container_width=True,
                hide_index=True,
            )
        if "difficulty" in fp2c.columns:
            st.dataframe(
                fp2c.groupby("difficulty").size().reset_index(name="count").sort_values("count", ascending=False),
                use_container_width=True,
                hide_index=True,
            )
        if {"query_id", "rank1_page", "expected_pages"}.issubset(fp2c.columns):
            tmp = fp2c.copy()
            tmp["exp_first"] = tmp["expected_pages"].apply(lambda x: _parse_pages_value(x)[0] if _parse_pages_value(x) else None)
            top2 = (
                ((tmp.get("rank1_page") == tmp["exp_first"]) | (tmp.get("rank2_page") == tmp["exp_first"]))
                .fillna(False)
                .mean()
            )
            top3 = (
                (
                    (tmp.get("rank1_page") == tmp["exp_first"])
                    | (tmp.get("rank2_page") == tmp["exp_first"])
                    | (tmp.get("rank3_page") == tmp["exp_first"])
                )
                .fillna(False)
                .mean()
            )
            st.write({"correct_in_top2_pct": round(float(top2 * 100), 2), "correct_in_top3_pct": round(float(top3 * 100), 2)})

with tabs[4]:
    st.subheader("Embedding Diagnostics")
    if not results_dir.exists():
        st.error(f"Results directory not found: {results_dir}")
    elif not doc_id:
        st.info("Select a document.")
    else:
        live = _load_embedding_artifacts(str(DATA_ROOT), str(doc_id))
        if not bool(live.get("ok", False)):
            st.warning(str(live.get("error") or "Could not load embedding artifacts."))
        else:
            emb = live["emb"]
            meta = live["meta"]
            st.markdown("**Live Embedding Map**")
            c0, c1, c2, c3 = st.columns(4)
            method = c0.selectbox("Projection", options=["PCA", "t-SNE", "UMAP"], index=0)
            color_by = c1.selectbox(
                "Color by",
                options=["none", "section_title", "subsection_title", "is_table", "page_start"],
                index=1,
            )
            max_points = int(c2.slider("Max points", 100, 5000, 1200, 100))
            seed = int(c3.number_input("Seed", min_value=0, max_value=99999, value=42, step=1))
            show_centroids = st.checkbox("Show cluster centroids", value=True)
            show_outlines = st.checkbox("Show cluster outlines", value=False)

            try:
                take_idx, xy = _project_embeddings(emb, method=method, max_points=max_points, seed=seed)
                show_meta = meta.iloc[take_idx].reset_index(drop=True)

                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(7.2, 5.4))
                doc_label = _doc_label_for_plot(str(doc_id))
                method_upper = str(method).upper()
                ax.set_title(f"{method_upper} projection of chunk embeddings from {doc_label}")
                if str(method).strip().lower() == "pca":
                    ax.set_xlabel("Principal Component 1")
                    ax.set_ylabel("Principal Component 2")
                else:
                    ax.set_xlabel("Component 1")
                    ax.set_ylabel("Component 2")

                cb = str(color_by)
                if cb == "none" or cb not in show_meta.columns:
                    ax.scatter(xy[:, 0], xy[:, 1], s=9, alpha=0.55)
                elif cb in {"section_title", "subsection_title", "is_table"}:
                    series = show_meta[cb].astype(str).fillna("Unknown")
                    top_vals = series.value_counts().head(8).index.tolist()
                    display = series.where(series.isin(top_vals), other="Other")
                    fixed_colors = {
                        "PERFORMANCE REPORT": "#1f77b4",  # blue
                        "ACCOUNTABILITY REPORT": "#2ca02c",  # green
                        "FINANCIAL STATEMENTS": "#9467bd",  # purple
                        "DIRECTIONS BY THE SCOTTISH MINISTERS": "#e377c2",  # pink
                    }
                    categories = display.unique().tolist()
                    cmap = plt.get_cmap("tab10")
                    for idx_cat, val in enumerate(categories):
                        mask = display == val
                        xg = xy[mask, 0]
                        yg = xy[mask, 1]
                        color = fixed_colors.get(str(val).upper(), cmap(idx_cat % 10))
                        ax.scatter(xg, yg, s=10, alpha=0.6, label=str(val), color=color)
                        if show_centroids and len(xg) >= 2:
                            cx = float(np.mean(xg))
                            cy = float(np.mean(yg))
                            ax.scatter(
                                [cx],
                                [cy],
                                marker="X",
                                s=78,
                                color=color,
                                edgecolor="black",
                                linewidth=0.8,
                                zorder=4,
                            )
                        if show_outlines and len(xg) >= 3:
                            try:
                                from scipy.spatial import ConvexHull  # type: ignore

                                pts = np.column_stack((xg, yg))
                                hull = ConvexHull(pts)
                                loop = np.append(hull.vertices, hull.vertices[0])
                                ax.plot(pts[loop, 0], pts[loop, 1], linewidth=1.1, alpha=0.5)
                            except Exception:
                                # Keep plot robust when scipy is unavailable or points are degenerate.
                                pass
                    ax.legend(loc="best", fontsize=7)
                else:
                    vals = pd.to_numeric(show_meta[cb], errors="coerce")
                    mask = vals.notna().values
                    if mask.any():
                        sc = ax.scatter(xy[mask, 0], xy[mask, 1], s=10, c=vals[mask], cmap="viridis", alpha=0.65)
                        fig.colorbar(sc, ax=ax, label=cb)
                    else:
                        ax.scatter(xy[:, 0], xy[:, 1], s=9, alpha=0.55)

                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as e:
                st.warning(f"Live projection failed ({method}). Details: {type(e).__name__}: {e}")

            st.markdown("**Live Similarity Histogram**")
            sample_pairs = int(st.slider("Cosine pair sample size", 1000, 50000, 12000, 1000))
            try:
                cos_vals = _sample_cosine_pairs(emb, sample_size=sample_pairs, seed=seed)
                if len(cos_vals) > 0:
                    import matplotlib.pyplot as plt

                    fig_h, ax_h = plt.subplots(figsize=(7.2, 4.6))
                    ax_h.hist(cos_vals, bins=50, alpha=0.8)
                    ax_h.set_title(f"Random Pair Cosine Similarity: {doc_id}")
                    ax_h.set_xlabel("Cosine similarity")
                    ax_h.set_ylabel("Count")
                    st.pyplot(fig_h, use_container_width=True)
                    plt.close(fig_h)
                    st.caption(
                        f"mean={float(np.mean(cos_vals)):.4f} | std={float(np.std(cos_vals)):.4f} | "
                        f"p5={float(np.percentile(cos_vals, 5)):.4f} | p95={float(np.percentile(cos_vals, 95)):.4f}"
                    )
                else:
                    st.info("Not enough vectors for cosine sampling.")
            except Exception as e:
                st.warning(f"Could not compute similarity histogram. Details: {type(e).__name__}: {e}")

            st.markdown("---")
            st.markdown("**Static charts from RESULTS_DIR**")

        st.markdown("---")
        st.markdown("**WIZMAP (Interactive)**")
        st.caption("Launch an interactive embedding map from `data.ndjson` and `grid.json` URLs.")

        discovered_wizmap_dir = _discover_wizmap_dir(str(doc_id))
        default_wizmap_dir = discovered_wizmap_dir or (project_root() / "results" / "wizmap" / str(doc_id) / "searchable")
        source_csv_default = _resolve_wizmap_source_csv(str(doc_id), str(DATA_ROOT))
        source_csv_default_str = str(source_csv_default) if source_csv_default else str(
            project_root() / "results" / "wizmap" / f"{doc_id}_wizmap_umap.csv"
        )

        current_query_text = str(question or "").strip()
        g0, g1, g2, g3, g4 = st.columns([2.2, 1.25, 1.1, 1.1, 1.3])
        source_csv_input = g0.text_input(
            "Source UMAP CSV",
            value=source_csv_default_str,
            key=f"wizmap_source_csv_{doc_id}",
        )
        generate_include_chunk_text = g1.checkbox("Use full chunk text (search)", value=True)
        generate_use_groups = g2.checkbox("Use category groups", value=False)
        include_query_point = g3.checkbox("Plot current query", value=True)
        auto_refresh_query_point = g4.checkbox("Auto-refresh query point", value=True)
        do_generate = st.button("Generate/Refresh WIZMAP files", use_container_width=True)
        wizmap_dir = st.text_input(
            "WIZMAP file dir",
            value=str(default_wizmap_dir),
            key=f"wizmap_dir_{doc_id}",
        )

        active_source = Path(source_csv_input).expanduser()
        active_wizmap_dir = Path(wizmap_dir).expanduser()
        query_hash = hashlib.sha1(current_query_text.encode("utf-8")).hexdigest()[:12] if current_query_text else "empty"
        auto_sync_signature = "|".join(
            [
                str(doc_id),
                str(active_source),
                str(active_wizmap_dir),
                "1" if generate_include_chunk_text else "0",
                "1" if generate_use_groups else "0",
                "1" if include_query_point else "0",
                query_hash if include_query_point else "no-query",
            ]
        )
        wizmap_sync_state_key = f"wizmap_sync_signature_{doc_id}"
        should_auto_refresh = bool(
            auto_refresh_query_point
            and include_query_point
            and current_query_text
            and active_source.exists()
            and (
                st.session_state.get(wizmap_sync_state_key) != auto_sync_signature
                or not _wizmap_output_contains_query(active_wizmap_dir, str(doc_id))
            )
        )

        if do_generate or should_auto_refresh:
            src = active_source
            out = active_wizmap_dir
            if not src.exists():
                st.error(f"Source CSV not found: {src}")
            else:
                ok, msg = _generate_wizmap_files(
                    doc_id=str(doc_id),
                    source_csv=src,
                    output_dir=out,
                    include_chunk_text=bool(generate_include_chunk_text),
                    use_category_groups=bool(generate_use_groups),
                    query_text=current_query_text if include_query_point else None,
                    data_root_str=str(DATA_ROOT),
                )
                if ok:
                    if do_generate:
                        st.success(msg)
                    elif should_auto_refresh:
                        st.caption(f"Auto-refreshed WIZMAP for current query. {msg}")
                    st.session_state[f"wizmap_cache_buster_{doc_id}"] = str(time.time_ns())
                    st.session_state[wizmap_sync_state_key] = auto_sync_signature
                else:
                    st.error(msg)

        active_source_path = str(Path(source_csv_input).expanduser())
        source_exists = Path(active_source_path).exists()
        source_ok, source_status = _validate_wizmap_source_csv(
            doc_id=str(doc_id),
            source_csv=Path(active_source_path),
            data_root_str=str(DATA_ROOT),
        )
        st.caption(
            f"Active source CSV: `{active_source_path}` "
            f"({'found' if source_exists else 'missing'}) | Projection tag: **UMAP**"
        )
        if source_exists:
            if source_ok:
                st.caption(source_status)
            else:
                st.warning(source_status)
        if include_query_point:
            if current_query_text:
                st.caption("Current question will be appended as a live `Query` point in WizMap.")
            else:
                st.caption("Enter a question to add a live `Query` point to WizMap.")
        if str(doc_id) not in active_source_path:
            st.warning("Source CSV path does not include selected doc_id. Confirm this is intentional.")
        if str(doc_id) not in str(wizmap_dir):
            st.warning("WIZMAP file dir does not include selected doc_id. This can load a different document.")

        s0, s1 = st.columns([2, 1])
        default_port = _doc_default_port(str(doc_id))
        wizmap_host = s0.text_input("WIZMAP local host", value="127.0.0.1", key=f"wizmap_host_{doc_id}")
        wizmap_port = int(
            s1.number_input(
                "Port",
                min_value=1,
                max_value=65535,
                value=default_port,
                step=1,
                key=f"wizmap_port_{doc_id}",
            )
        )

        cache_buster_key = f"wizmap_cache_buster_{doc_id}"
        if cache_buster_key not in st.session_state:
            st.session_state[cache_buster_key] = str(time.time_ns())
        cache_buster = str(st.session_state[cache_buster_key])
        default_data_url = f"http://{wizmap_host}:{wizmap_port}/data.ndjson?v={cache_buster}"
        default_grid_url = f"http://{wizmap_host}:{wizmap_port}/grid.json?v={cache_buster}"
        data_url_input = st.text_input(
            "Data URL (data.ndjson)",
            value=default_data_url,
            key=f"wizmap_data_url_{doc_id}",
        )
        grid_url_input = st.text_input(
            "Grid URL (grid.json)",
            value=default_grid_url,
            key=f"wizmap_grid_url_{doc_id}",
        )
        wizmap_url = _build_wizmap_url(data_url_input, grid_url_input, state_key=f"{doc_id}_{wizmap_port}")

        if wizmap_url:
            st.success("Recommended: open WIZMAP in a new tab (Chrome).")
            st.link_button("Open WIZMAP in new tab", wizmap_url, use_container_width=False)
            st.text_input("Launch URL (copy this)", value=wizmap_url, key=f"wizmap_launch_url_{doc_id}")
            st.caption("For demos, Chrome new-tab mode is the most reliable option.")
            st.code(f'open -a "Google Chrome" "{wizmap_url}"', language="bash")
            st.caption("If WIZMAP shows `0 Data Points`, open `Contour` and enable all groups, or open in Incognito.")
            st.code(
                " ".join(
                    [
                        "python",
                        "scripts/serve_wizmap_local.py",
                        f"--dir \"{wizmap_dir}\"",
                        f"--host {wizmap_host}",
                        f"--port {wizmap_port}",
                    ]
                ),
                language="bash",
            )
            st.warning(
                "Embedded iframe can appear blank due to browser cross-origin restrictions. "
                "Use new-tab mode if that happens."
            )
            if st.checkbox("Embed WIZMAP here (experimental)", value=False):
                components.iframe(wizmap_url, height=760, scrolling=True)
        else:
            st.info("Provide both Data URL and Grid URL to launch WIZMAP.")
        pca_name = f"vector_pca_{doc_id}.png"
        hist_name = f"vector_similarity_hist_{doc_id}.png"
        pca_path = results_dir / pca_name
        hist_path = results_dir / hist_name
        show_missing_figures = st.toggle("Show missing figure sections", value=False)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**PCA**")
            if pca_path.exists():
                st.image(str(pca_path), use_container_width=True)
            elif show_missing_figures:
                st.warning(f"Missing image: {pca_path}")
        with c2:
            st.markdown("**Similarity Histogram**")
            if hist_path.exists():
                st.image(str(hist_path), use_container_width=True)
            elif show_missing_figures:
                st.warning(f"Missing image: {hist_path}")

        st.markdown("---")
        st.markdown("**Paper Figures: Random vs Retrieved Similarity**")
        fig_random_pairs = results_dir / f"vector_similarity_hist_chunk_pairs_{doc_id}.png"
        fig_overlay = results_dir / f"vector_similarity_random_vs_retrieved_{doc_id}.png"
        fig_top1 = results_dir / f"vector_similarity_hist_retrieved_top1_{doc_id}.png"
        fig_summary = results_dir / f"vector_similarity_random_vs_retrieved_summary_{doc_id}.csv"
        fig_rel_non = results_dir / f"vector_similarity_relevant_vs_nonrelevant_{doc_id}.png"
        fig_rel_non_summary = results_dir / f"vector_similarity_relevant_vs_nonrelevant_summary_{doc_id}.csv"

        p1, p2 = st.columns(2)
        with p1:
            st.markdown("**Random chunk-pair similarity**")
            if fig_random_pairs.exists():
                st.image(str(fig_random_pairs), use_container_width=True)
            elif show_missing_figures:
                st.info(f"Missing figure: {fig_random_pairs.name}")
        with p2:
            st.markdown("**Random vs retrieved similarity (overlay)**")
            if fig_overlay.exists():
                st.image(str(fig_overlay), use_container_width=True)
            elif show_missing_figures:
                st.info(f"Missing figure: {fig_overlay.name}")

        p3, p4 = st.columns(2)
        with p3:
            st.markdown("**Top-1 retrieved similarity**")
            if fig_top1.exists():
                st.image(str(fig_top1), use_container_width=True)
            elif show_missing_figures:
                st.info(f"Missing figure: {fig_top1.name}")
        with p4:
            st.markdown("**Summary metrics**")
            if fig_summary.exists():
                try:
                    df_sum = pd.read_csv(fig_summary)
                    st.dataframe(df_sum, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.warning(f"Could not load summary CSV: {type(e).__name__}: {e}")
            elif show_missing_figures:
                st.info(f"Missing summary: {fig_summary.name}")

        st.markdown("**Embedding behaviour: relevant vs non-relevant similarity**")
        q1, q2 = st.columns(2)
        with q1:
            if fig_rel_non.exists():
                st.image(str(fig_rel_non), use_container_width=True)
            elif show_missing_figures:
                st.info(f"Missing figure: {fig_rel_non.name}")
        with q2:
            if fig_rel_non_summary.exists():
                try:
                    rel_df = pd.read_csv(fig_rel_non_summary)
                    st.dataframe(rel_df, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.warning(f"Could not load relevance summary CSV: {type(e).__name__}: {e}")
            elif show_missing_figures:
                st.info(f"Missing summary: {fig_rel_non_summary.name}")

with tabs[5]:
    st.subheader("Run Info")
    if not doc_id:
        st.info("Select a document.")
    else:
        run_info = artifacts.get("run_info", {})
        metrics_state = artifacts.get("metrics_state", {}) if isinstance(artifacts, dict) else {}
        metrics_exists = bool(metrics_state.get("exists")) if isinstance(metrics_state, dict) else False
        metrics_path_text = str(metrics_state.get("path") or "")
        doc_dir = Path(str(stats.get("data_dir"))) if isinstance(stats, dict) and stats.get("data_dir") else None
        core_paths = [
            (doc_dir / "chunks.parquet") if doc_dir else None,
            (doc_dir / "chunk_meta.parquet") if doc_dir else None,
            (doc_dir / "faiss.index") if doc_dir else None,
        ]
        core_existing = [p for p in core_paths if isinstance(p, Path) and p.exists()]
        newest_core_mtime = max((p.stat().st_mtime for p in core_existing), default=0.0)
        metrics_mtime = 0.0
        if metrics_exists and metrics_path_text:
            try:
                metrics_mtime = Path(metrics_path_text).stat().st_mtime
            except Exception:
                metrics_mtime = 0.0
        is_metrics_stale = bool(metrics_exists and newest_core_mtime > metrics_mtime)

        if not metrics_exists:
            st.warning("No retrieval metrics file found for this document. Run retrieval evaluation to populate Run Info.")
        elif is_metrics_stale:
            st.warning("Retrieval metrics appear stale vs current pipeline artifacts (chunks/index newer than metrics).")
        else:
            st.success("Run Info is linked to current retrieval metrics artifact.")
        if metrics_path_text:
            st.caption(f"Metrics source: {metrics_path_text}")

        pipe = run_info.get("pipeline_settings", {}) if isinstance(run_info, dict) else {}
        eval_set = run_info.get("eval_set", {}) if isinstance(run_info, dict) else {}
        metrics_k = artifacts.get("metrics_by_k", {}) if isinstance(artifacts.get("metrics_by_k", {}), dict) else {}
        k1 = metrics_k.get("1", {})
        items = {
            "doc_id": pipe.get("doc_id") or doc_id,
            "year": pipe.get("report_year"),
            "chunk_size_tokens": pipe.get("chunk_size_tokens"),
            "chunk_overlap_tokens": pipe.get("chunk_overlap_tokens"),
            "segment_aware_chunking": pipe.get("segment_aware_chunking"),
            "retriever": run_info.get("method", "hybrid_rrf_dense_bm25"),
            "embedding_model": run_info.get("embedding_model"),
            "eval_set_sha1": eval_set.get("sha1"),
            "eval_query_count": eval_set.get("query_count"),
            "hit@1": k1.get("page_hit_rate_at_k"),
            "mrr@1": k1.get("mean_page_mrr_at_k"),
        }
        for key, value in items.items():
            st.write(f"- `{key}`: `{value}`")

with tabs[6]:
    st.subheader("Pipeline Architecture")
    components.html(_pipeline_demo_html(), height=940, scrolling=True)

with tabs[7]:
    st.subheader("System Metrics")
    if "system_metrics" not in st.session_state:
        st.session_state["system_metrics"] = None
    if st.button("Refresh Metrics", key="refresh_system_metrics"):
        try:
            st.session_state["system_metrics"] = api_get("/api/v1/metrics")
        except Exception as e:
            st.error(f"Failed to load system metrics: {e}")

    metrics_payload = st.session_state.get("system_metrics")
    if not isinstance(metrics_payload, dict):
        try:
            metrics_payload = api_get("/api/v1/metrics")
            st.session_state["system_metrics"] = metrics_payload
        except Exception as e:
            st.warning(f"Metrics unavailable: {e}")
            metrics_payload = {}

    gen_counts = metrics_payload.get("generation_counts", {}) if isinstance(metrics_payload, dict) else {}
    citation_counts = metrics_payload.get("citation_counts", {}) if isinstance(metrics_payload, dict) else {}
    derived = metrics_payload.get("derived", {}) if isinstance(metrics_payload, dict) else {}

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Gen Total", int(gen_counts.get("total", 0) or 0))
    c2.metric("Gen OK", int(gen_counts.get("ok", 0) or 0))
    c3.metric("Gen Skipped", int(gen_counts.get("skipped", 0) or 0))
    c4.metric("Insufficient", int(gen_counts.get("insufficient_evidence", 0) or 0))
    c5.metric("Gen Error", int(gen_counts.get("error", 0) or 0))

    d1, d2, d3 = st.columns(3)
    valid_rate = derived.get("citation_valid_rate")
    reject_rate = derived.get("citation_rejected_rate")
    avg_latency = derived.get("generation_avg_latency_ms")
    d1.metric("Citation Valid Rate", f"{float(valid_rate):.2%}" if valid_rate is not None else "n/a")
    d2.metric("Citation Reject Rate", f"{float(reject_rate):.2%}" if reject_rate is not None else "n/a")
    d3.metric("Avg Gen Latency (ms)", f"{float(avg_latency):.2f}" if avg_latency is not None else "n/a")

    st.markdown("**Citation Counters**")
    st.write(
        {
            "parsed_total": int(citation_counts.get("parsed_total", 0) or 0),
            "valid_total": int(citation_counts.get("valid_total", 0) or 0),
            "rejected_total": int(citation_counts.get("rejected_total", 0) or 0),
        }
    )

st.divider()
st.caption("Tip: start API with `uvicorn app.api.main:app --reload` and this UI with `streamlit run app/ui/streamlit_app.py`.")
