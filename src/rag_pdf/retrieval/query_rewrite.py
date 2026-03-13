from __future__ import annotations

from typing import Optional

from rag_pdf.question_router import route_question


REWRITE_SYNONYMS = {
    "deficit": ["overspend", "shortfall"],
    "surplus": ["underspend"],
    "staff costs": ["employee benefits", "remuneration", "pension costs"],
    "emissions": ["greenhouse gas", "carbon", "co2"],
    "integration joint board": ["ijb"],
    "ijb": ["integration joint board"],
}


def _expand_synonyms(question: str) -> str:
    """Expand key domain terms into inline synonyms for robust retrieval."""
    out = question
    q_lower = question.lower()
    for term, synonyms in REWRITE_SYNONYMS.items():
        if term in q_lower:
            out = f"{out} ({' / '.join(synonyms)})"
    return out


def _intent_hint_rewrite(question: str) -> Optional[str]:
    """Produce one intent-specific rewrite guided by question router output."""
    route = route_question(question)
    q = question.strip()
    if route.intent == "table_metric_staff_costs":
        return f"{q} in staff cost table (employee benefits, pension, remuneration)"
    if route.intent == "table_metric_emissions":
        return f"{q} in emissions table (target emissions, percentage change)"
    if route.intent.startswith("table_metric_"):
        return f"{q} in performance table by quarter"
    return None


def generate_query_rewrites(question: str) -> list[str]:
    """Return deterministic rewrite variants to improve semantic recall."""
    rewrites: list[str] = []
    expanded = _expand_synonyms(question)
    if expanded.strip().lower() != question.strip().lower():
        rewrites.append(expanded)

    intent_rewrite = _intent_hint_rewrite(question)
    if intent_rewrite and intent_rewrite.strip().lower() != question.strip().lower():
        rewrites.append(intent_rewrite)

    deduped: list[str] = []
    seen: set[str] = set()
    for rw in rewrites:
        key = rw.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(rw.strip())
    return deduped
