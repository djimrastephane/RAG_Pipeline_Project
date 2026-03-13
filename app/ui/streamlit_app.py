from __future__ import annotations

import base64
import ast
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


def _resolve_wizmap_source_csv(doc_id: str) -> Path | None:
    wiz_root = project_root() / "results" / "wizmap"
    if not wiz_root.exists():
        return None
    candidates = [
        wiz_root / f"{doc_id}_wizmap_umap.csv",
        wiz_root / f"{_slug(doc_id)}_wizmap_umap.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    globbed = sorted(wiz_root.glob(f"*{doc_id}*wizmap_umap.csv"))
    return globbed[0] if globbed else None


def _resolve_chunks_for_doc(doc_id: str) -> Path | None:
    roots = [
        project_root() / "data_processed" / doc_id,
        project_root() / "data_processed_toc_upgrade_5docs" / doc_id,
        project_root() / "data_processed_toc_upgrade_test" / doc_id,
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


def _generate_wizmap_files(
    doc_id: str,
    source_csv: Path,
    output_dir: Path,
    include_chunk_text: bool = True,
    use_category_groups: bool = False,
) -> tuple[bool, str]:
    try:
        import wizmap  # type: ignore
    except Exception as e:
        return False, f"Missing `wizmap` package: {type(e).__name__}: {e}"

    try:
        df = pd.read_csv(source_csv)
    except Exception as e:
        return False, f"Could not read source CSV: {type(e).__name__}: {e}"

    required = {"x", "y"}
    if not required.issubset(df.columns):
        return False, f"Source CSV missing required columns: {sorted(required)}"

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

    if include_chunk_text:
        chunks_path = _resolve_chunks_for_doc(doc_id)
        if chunks_path is not None:
            try:
                chunks = pd.read_parquet(chunks_path)
                join_cols = {}
                if "chunk_id_global" in chunks.columns:
                    join_cols["chunk_id_global"] = "id"
                if "chunk_text" in chunks.columns:
                    join_cols["chunk_text"] = "chunk_text_full"
                if join_cols:
                    tmp = chunks[list(join_cols.keys())].rename(columns=join_cols)
                    df = df.merge(tmp, on="id", how="left")
            except Exception:
                pass

    xs = pd.to_numeric(df["x"], errors="coerce").fillna(0.0).astype(float).tolist()
    ys = pd.to_numeric(df["y"], errors="coerce").fillna(0.0).astype(float).tolist()

    def _mk_text(r: pd.Series) -> str:
        body = str(r.get("chunk_text_full") or r.get("text") or "")
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
        return True, f"Wrote {output_dir / 'data.ndjson'} and {output_dir / 'grid.json'}"
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
    include_generated_answer = st.toggle("Generate answer (Local LLM)", value=False)
    with st.expander("Generation context controls (live)", expanded=False):
        gen_max_context_chunks = st.slider("Max context chunks", 1, 20, 5, 1)
        gen_max_context_chars = st.slider("Max context chars", 1000, 20000, 9000, 500)
        gen_max_chunk_chars = st.slider("Max chars per chunk", 200, 4000, 2200, 100)
        gen_timeout_seconds = st.slider("Generation timeout (sec)", 5, 180, 20, 5)

    if st.button("Run Search", disabled=(not doc_id or not question.strip())):
        try:
            payload = {
                "question": question.strip(),
                "k": int(k),
                "query_id": (query_id or None),
                "include_generated_answer": bool(include_generated_answer),
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
                f"bm25_w={retrieval_cfg.get('bm25_weight')}"
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
                title = (
                    f"Rank {r.get('rank')} | {_short_chunk_name(str(r.get('chunk_id')))} | "
                    f"p{_format_pages(r.get('pages', []))} | "
                    f"{'table' if bool(r.get('is_table')) else 'narrative'}"
                )
                with st.expander(title, expanded=False):
                    st.caption(
                        f"dense={r.get('dense_raw_score')} | bm25={r.get('bm25_raw_score')} | fused={r.get('score')}"
                    )
                    st.write((r.get("chunk_text", "") or "")[:500])
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
        source_csv_default = _resolve_wizmap_source_csv(str(doc_id))
        source_csv_default_str = str(source_csv_default) if source_csv_default else str(
            project_root() / "results" / "wizmap" / f"{doc_id}_wizmap_umap.csv"
        )

        g0, g1, g2, g3 = st.columns([2.3, 1.5, 1.3, 1.2])
        source_csv_input = g0.text_input(
            "Source UMAP CSV",
            value=source_csv_default_str,
            key=f"wizmap_source_csv_{doc_id}",
        )
        generate_include_chunk_text = g1.checkbox("Use full chunk text (search)", value=True)
        generate_use_groups = g2.checkbox("Use category groups", value=False)
        do_generate = g3.button("Generate/Refresh WIZMAP files", use_container_width=True)
        wizmap_dir = st.text_input(
            "WIZMAP file dir",
            value=str(default_wizmap_dir),
            key=f"wizmap_dir_{doc_id}",
        )

        if do_generate:
            src = Path(source_csv_input).expanduser()
            out = Path(wizmap_dir).expanduser()
            if not src.exists():
                st.error(f"Source CSV not found: {src}")
            else:
                ok, msg = _generate_wizmap_files(
                    doc_id=str(doc_id),
                    source_csv=src,
                    output_dir=out,
                    include_chunk_text=bool(generate_include_chunk_text),
                    use_category_groups=bool(generate_use_groups),
                )
                if ok:
                    st.success(msg)
                    st.session_state[f"wizmap_cache_buster_{doc_id}"] = str(time.time_ns())
                else:
                    st.error(msg)

        active_source_path = str(Path(source_csv_input).expanduser())
        source_exists = Path(active_source_path).exists()
        st.caption(
            f"Active source CSV: `{active_source_path}` "
            f"({'found' if source_exists else 'missing'}) | Projection tag: **UMAP**"
        )
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
                        "./.venv/bin/python",
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
