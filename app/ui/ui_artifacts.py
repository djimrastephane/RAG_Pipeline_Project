from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@st.cache_data(show_spinner=False)
def load_json(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def load_csv(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def retrieval_metrics_path(data_root: Path, doc_id: str) -> Path:
    doc_dir = data_root / doc_id
    preferred = doc_dir / "retrieval_metrics_hybrid.json"
    if preferred.exists():
        return preferred
    fallback = doc_dir / "retrieval_metrics.json"
    return fallback


def retrieval_results_path(data_root: Path, doc_id: str) -> Path:
    doc_dir = data_root / doc_id
    preferred = doc_dir / "retrieval_results_hybrid.json"
    if preferred.exists():
        return preferred
    fallback = doc_dir / "retrieval_results.json"
    return fallback


def run_info_from_metrics(data_root: Path, doc_id: str) -> dict[str, Any]:
    metrics = load_json(str(retrieval_metrics_path(data_root, doc_id)))
    return metrics.get("run_info", {}) if isinstance(metrics, dict) else {}


def metrics_by_k_from_metrics(data_root: Path, doc_id: str) -> dict[str, Any]:
    metrics = load_json(str(retrieval_metrics_path(data_root, doc_id)))
    out = metrics.get("metrics_by_k", {}) if isinstance(metrics, dict) else {}
    return out if isinstance(out, dict) else {}


def artifact_state(path: Path) -> dict[str, Any]:
    """Return lightweight status info for an artifact path."""
    if not path.exists():
        return {"exists": False, "path": str(path), "mtime_utc": None}
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except Exception:
        mtime = None
    return {"exists": True, "path": str(path), "mtime_utc": mtime}
