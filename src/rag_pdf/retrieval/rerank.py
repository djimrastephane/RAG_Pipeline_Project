from __future__ import annotations

import re
from dataclasses import dataclass


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is", "it", "of", "on",
    "or", "that", "the", "to", "was", "what", "when", "where", "which", "who", "why", "with", "year",
    "did", "do", "does", "were", "was", "had", "has", "have", "within", "between", "during", "reported",
    "report", "show", "shown", "stated", "figure", "value", "total",
}

NUMERIC_INTENT_PHRASES = (
    "how much",
    "how many",
    "what amount",
    "what proportion",
    "what percentage",
    "what percent",
    "what was the total",
    "what is the total",
    "number of",
    "amount of",
)

NUMERIC_INTENT_TOKENS = {
    "amount", "total", "value", "cost", "spend", "spending", "expenditure", "budget", "deficit", "surplus",
    "income", "assets", "liabilities", "cash", "equivalent", "equivalents", "rate", "ratio", "percent",
    "percentage", "proportion", "count", "number", "overspend", "underspend", "recurring", "nonrecurring",
}

TOKEN_CANONICAL_MAP = {
    "spent": "spend",
    "spending": "spend",
    "costs": "cost",
    "amounts": "amount",
    "totals": "total",
    "figures": "figure",
    "percentages": "percent",
    "percent": "percent",
    "percentage": "percent",
    "proportion": "percent",
    "proportions": "percent",
    "values": "value",
    "deficits": "deficit",
    "surpluses": "surplus",
    "assets": "asset",
    "liabilities": "liability",
    "equivalents": "equivalent",
    "overspent": "overspend",
    "under-spend": "underspend",
    "non-recurring": "nonrecurring",
    "one-off": "nonrecurring",
    "oneoff": "nonrecurring",
}


@dataclass(frozen=True)
class RerankConfig:
    """Configuration values used to compute lightweight lexical rerank boosts."""

    table_chunk_boost: float = 0.08
    entity_match_boost: float = 0.04
    numeric_density_boost: float = 0.03
    max_entity_matches: int = 4


def normalize_text(text: str) -> str:
    """Normalize free text for overlap and keyword checks."""
    return re.sub(r"\s+", " ", str(text or "").strip().lower().replace("’", "'"))


def _normalize_token(token: str) -> str:
    t = str(token or "").lower().strip()
    if not t:
        return ""
    t = TOKEN_CANONICAL_MAP.get(t, t)
    if t.endswith("ies") and len(t) > 4:
        t = f"{t[:-3]}y"
    elif t.endswith("es") and len(t) > 4 and not t.endswith("ses"):
        t = t[:-2]
    elif t.endswith("s") and len(t) > 3 and not t.endswith("ss"):
        t = t[:-1]
    return TOKEN_CANONICAL_MAP.get(t, t)


def _tokenize_normalized(text: str) -> list[str]:
    raw = re.findall(r"[a-z0-9][a-z0-9'%\-]{1,}", normalize_text(text))
    out: list[str] = []
    for tok in raw:
        t = _normalize_token(tok)
        if t:
            out.append(t)
    return out


def extract_query_entities(question: str) -> list[str]:
    """Extract stable query entities from the question using token-level filtering."""
    tokens = _tokenize_normalized(question)
    entities: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        if len(tok) < 2:
            continue
        if tok in STOPWORDS:
            continue
        if tok.isdigit() and len(tok) < 3:
            continue
        if tok not in seen:
            seen.add(tok)
            entities.append(tok)
    return entities


def query_overlap_boost(question: str, chunk_text: str, config: RerankConfig) -> float:
    """Return lexical overlap boost based on matched query entities in the chunk."""
    entities = extract_query_entities(question)
    if not entities:
        return 0.0

    chunk_tokens = set(_tokenize_normalized(chunk_text))
    if not chunk_tokens:
        return 0.0

    matches = 0
    for ent in entities:
        if ent and ent in chunk_tokens:
            matches += 1
            if matches >= config.max_entity_matches:
                break
    return float(matches) * float(config.entity_match_boost)


def numeric_question(question: str) -> bool:
    """Return True when question intent likely targets numeric/tabular evidence."""
    q = normalize_text(question)
    if any(p in q for p in NUMERIC_INTENT_PHRASES):
        return True
    if re.search(r"\bq[1-4]\b", q):
        return True
    if "%" in q or "£" in q:
        return True
    q_tokens = set(_tokenize_normalized(q))
    return bool(q_tokens.intersection(NUMERIC_INTENT_TOKENS))


def numeric_content_density(chunk_text: str) -> float:
    """Compute ratio of numeric-like tokens in chunk text."""
    text = normalize_text(chunk_text)
    if not text:
        return 0.0
    tokens = re.findall(r"[a-z0-9.,%-]+", text)
    if not tokens:
        return 0.0
    numeric = sum(1 for t in tokens if re.search(r"\d", t))
    return numeric / float(len(tokens))


def numeric_density_boost(question: str, chunk_text: str, config: RerankConfig) -> float:
    """Return numeric-density boost for value-seeking questions."""
    if not numeric_question(question):
        return 0.0
    density = numeric_content_density(chunk_text)
    if density < 0.10:
        return 0.0
    boost = float(config.numeric_density_boost)
    if density >= 0.20:
        boost += 0.5 * float(config.numeric_density_boost)
    q = normalize_text(question)
    if ("%" in q and "%" in chunk_text) or ("£" in q and "£" in chunk_text):
        boost += 0.5 * float(config.numeric_density_boost)
    return boost


def table_priority_boost(is_table_chunk: bool, route_intent: str, config: RerankConfig) -> float:
    """Return table-priority boost for routed table-metric questions."""
    if not is_table_chunk:
        return 0.0
    if str(route_intent or "").startswith("table_metric_"):
        return float(config.table_chunk_boost)
    return 0.0
