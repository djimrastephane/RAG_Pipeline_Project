"""
Build an assisted heading-label dataset from processed pages.

Outputs a CSV with:
- auto_label (heuristic pre-label)
- auto_confidence (0-1)
- review_priority (higher = review first)
- final_label (empty; human fills as 0/1)

Designed for human-in-the-loop labeling rounds.
"""

from __future__ import annotations

import argparse
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
src_path = repo_root / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from rag_pdf.headings import (
    is_global_boilerplate_heading,
    is_section_anchor_line,
    looks_like_heading_text_only,
    looks_like_lettered_subsection,
)


HEADING_KEYWORDS = {
    "report",
    "performance",
    "governance",
    "accountability",
    "financial",
    "remuneration",
    "staff",
    "statement",
    "overview",
    "risk",
    "sustainability",
    "targets",
    "compliance",
}

NUMBER_TOKEN_RE = re.compile(r"^(?:[A-Z](?:[.)])?|\d+(?:\.\d+){0,5})\s+")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build assisted heading-label seed CSV.")
    p.add_argument("--data-root", default="data_processed")
    p.add_argument("--doc-regex", default=r"^(Grampian|Shetland)-\d{4}-\d{4}$")
    p.add_argument("--max-docs", type=int, default=12)
    p.add_argument("--max-lines-per-doc", type=int, default=500)
    p.add_argument("--body-sample-per-page", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-csv",
        default="data_processed/labeling/heading_assisted_seed.csv",
    )
    return p.parse_args()


def _iter_top_lines(v: Any) -> list[dict[str, Any]]:
    if v is None:
        return []
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        out: list[dict[str, Any]] = []
        for x in v:
            if isinstance(x, dict):
                out.append(x)
            else:
                out.append({"text": str(x)})
        return out
    return [{"text": str(v)}]


def _iter_heading_candidates(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if str(x).strip()]
    s = str(v).strip()
    return [s] if s else []


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _uppercase_ratio(s: str) -> float:
    alpha = [c for c in s if c.isalpha()]
    if not alpha:
        return 0.0
    return float(sum(c.isupper() for c in alpha) / len(alpha))


def _digit_ratio(s: str) -> float:
    if not s:
        return 0.0
    return float(sum(c.isdigit() for c in s) / len(s))


def _contains_keyword(s: str) -> bool:
    toks = re.findall(r"[a-z]+", s.lower())
    return any(t in HEADING_KEYWORDS for t in toks)


def _auto_label_and_confidence(text: str, source: str, y_norm: float | None) -> tuple[int, float]:
    t = _norm_text(text)
    if not t:
        return 0, 0.0

    is_heading = looks_like_heading_text_only(t)
    is_sub = looks_like_lettered_subsection(t)
    is_anchor = is_section_anchor_line(t)
    is_global = is_global_boilerplate_heading(t)
    starts_num = bool(NUMBER_TOKEN_RE.match(t))
    upr = _uppercase_ratio(t)
    has_kw = _contains_keyword(t)
    ends_punct = t.endswith((".", ";"))

    label = 1 if (is_heading or is_sub or is_anchor) and not is_global else 0

    score = 0.0
    if label == 1:
        score += 0.55
    if is_heading:
        score += 0.10
    if is_sub:
        score += 0.10
    if is_anchor:
        score += 0.10
    if starts_num:
        score += 0.05
    if upr >= 0.65:
        score += 0.05
    if has_kw:
        score += 0.05
    if is_global:
        score -= 0.30
    if ends_punct:
        score -= 0.10
    if source == "heading_candidates":
        score += 0.03
    if y_norm is not None and y_norm <= 0.18:
        score += 0.05

    conf = max(0.01, min(0.99, score if label == 1 else 1.0 - score))
    return label, float(conf)


def _review_priority(conf: float, source: str, auto_label: int) -> float:
    # Prefer uncertain and high-impact cases.
    uncertainty = 1.0 - abs(conf - 0.5) * 2.0
    src_bonus = 0.08 if source == "heading_candidates" else 0.0
    pos_bonus = 0.04 if auto_label == 1 else 0.0
    return float(uncertainty + src_bonus + pos_bonus)


def main() -> None:
    args = parse_args()
    rng = random.Random(int(args.seed))
    data_root = Path(args.data_root).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    doc_re = re.compile(args.doc_regex)
    doc_dirs = sorted([d for d in data_root.iterdir() if d.is_dir() and doc_re.match(d.name)])
    if args.max_docs > 0:
        doc_dirs = doc_dirs[: args.max_docs]
    if not doc_dirs:
        raise FileNotFoundError(f"No matching doc folders under {data_root}")

    rows: list[dict[str, Any]] = []
    for doc_dir in doc_dirs:
        pages_path = doc_dir / "pages.parquet"
        if not pages_path.exists():
            continue
        pages = pd.read_parquet(pages_path)
        doc_rows: list[dict[str, Any]] = []
        seen: set[tuple[int, str]] = set()

        for _, r in pages.iterrows():
            page = int(r.get("page", 0) or 0)
            if page <= 0:
                continue
            page_h = float(r.get("page_height", 0.0) or 0.0)

            for ln in _iter_top_lines(r.get("top_lines")):
                txt = _norm_text(ln.get("text", ""))
                if not txt:
                    continue
                key = (page, txt.lower())
                if key in seen:
                    continue
                seen.add(key)
                y0 = ln.get("y0")
                y_norm = None
                if y0 is not None and page_h > 0:
                    try:
                        y_norm = float(y0) / page_h
                    except Exception:
                        y_norm = None
                auto_label, conf = _auto_label_and_confidence(txt, "top_lines", y_norm)
                doc_rows.append(
                    {
                        "doc_id": doc_dir.name,
                        "page": page,
                        "source": "top_lines",
                        "line_text": txt,
                        "y_norm": y_norm,
                        "char_len": len(txt),
                        "word_count": len(txt.split()),
                        "ends_punct": int(txt.endswith((".", ";"))),
                        "comma_count": txt.count(","),
                        "uppercase_ratio": _uppercase_ratio(txt),
                        "digit_ratio": _digit_ratio(txt),
                        "starts_number_token": int(bool(NUMBER_TOKEN_RE.match(txt))),
                        "contains_heading_keyword": int(_contains_keyword(txt)),
                        "auto_label": int(auto_label),
                        "auto_confidence": float(conf),
                    }
                )

            for txt in _iter_heading_candidates(r.get("heading_candidates")):
                txt = _norm_text(txt)
                if not txt:
                    continue
                key = (page, txt.lower())
                if key in seen:
                    continue
                seen.add(key)
                auto_label, conf = _auto_label_and_confidence(txt, "heading_candidates", None)
                doc_rows.append(
                    {
                        "doc_id": doc_dir.name,
                        "page": page,
                        "source": "heading_candidates",
                        "line_text": txt,
                        "y_norm": np.nan,
                        "char_len": len(txt),
                        "word_count": len(txt.split()),
                        "ends_punct": int(txt.endswith((".", ";"))),
                        "comma_count": txt.count(","),
                        "uppercase_ratio": _uppercase_ratio(txt),
                        "digit_ratio": _digit_ratio(txt),
                        "starts_number_token": int(bool(NUMBER_TOKEN_RE.match(txt))),
                        "contains_heading_keyword": int(_contains_keyword(txt)),
                        "auto_label": int(auto_label),
                        "auto_confidence": float(conf),
                    }
                )

            # Add small body-text negatives for calibration.
            clean = _norm_text(str(r.get("clean_text", "")))
            if clean:
                body_lines = [_norm_text(x) for x in clean.splitlines() if _norm_text(x)]
                if body_lines:
                    for txt in rng.sample(body_lines, k=min(int(args.body_sample_per_page), len(body_lines))):
                        key = (page, txt.lower())
                        if key in seen:
                            continue
                        seen.add(key)
                        auto_label, conf = _auto_label_and_confidence(txt, "body_sample", None)
                        doc_rows.append(
                            {
                                "doc_id": doc_dir.name,
                                "page": page,
                                "source": "body_sample",
                                "line_text": txt,
                                "y_norm": np.nan,
                                "char_len": len(txt),
                                "word_count": len(txt.split()),
                                "ends_punct": int(txt.endswith((".", ";"))),
                                "comma_count": txt.count(","),
                                "uppercase_ratio": _uppercase_ratio(txt),
                                "digit_ratio": _digit_ratio(txt),
                                "starts_number_token": int(bool(NUMBER_TOKEN_RE.match(txt))),
                                "contains_heading_keyword": int(_contains_keyword(txt)),
                                "auto_label": int(auto_label),
                                "auto_confidence": float(conf),
                            }
                        )

        # Keep highest-priority subset per doc.
        if not doc_rows:
            continue
        ddf = pd.DataFrame(doc_rows)
        ddf["review_priority"] = ddf.apply(
            lambda x: _review_priority(float(x["auto_confidence"]), str(x["source"]), int(x["auto_label"])),
            axis=1,
        )
        ddf = ddf.sort_values(["review_priority", "source", "page"], ascending=[False, True, True])
        ddf = ddf.head(int(args.max_lines_per_doc))
        rows.extend(ddf.to_dict(orient="records"))

    if not rows:
        raise RuntimeError("No rows generated.")

    out = pd.DataFrame(rows).reset_index(drop=True)
    out.insert(len(out.columns), "final_label", "")
    out.insert(len(out.columns), "review_notes", "")
    out = out.sort_values(["doc_id", "review_priority", "page"], ascending=[True, False, True]).reset_index(drop=True)
    out.to_csv(out_csv, index=False)

    summary = (
        out.groupby("doc_id", as_index=False)
        .agg(
            rows=("line_text", "count"),
            auto_heading_rate=("auto_label", "mean"),
            mean_conf=("auto_confidence", "mean"),
        )
        .sort_values("doc_id")
    )
    summary_csv = out_csv.with_name(out_csv.stem + "_summary.csv")
    summary.to_csv(summary_csv, index=False)

    print("Saved:", out_csv)
    print("Saved:", summary_csv)
    print("Rows:", len(out))
    print("Docs:", out["doc_id"].nunique())


if __name__ == "__main__":
    main()
