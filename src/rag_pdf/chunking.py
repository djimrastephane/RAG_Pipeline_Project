from __future__ import annotations
import re

try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None


def get_encoder():
    """Get tiktoken encoder for accurate token counting."""
    if tiktoken is None:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def require_encoder():
    """Return a tiktoken encoder or raise a clear error for preprocessing paths."""
    enc = get_encoder()
    if enc is None:
        raise RuntimeError(
            "tiktoken with cl100k_base is required for preprocessing. "
            "Install and verify tiktoken instead of using the word-based fallback."
        )
    return enc


def count_tokens(text: str, enc) -> int:
    """
    Count tokens in text.

    Uses tiktoken if available, otherwise estimates based on word count.
    """
    if enc is None:
        return max(1, int(len(text.split()) / 0.75))
    return len(enc.encode(text))


def chunk_text_by_tokens(
    text: str,
    chunk_tokens: int,
    overlap_tokens: int,
    enc,
) -> list[str]:
    """
    Split text into overlapping chunks by token count.

    Args:
        text: Text to chunk
        chunk_tokens: Target chunk size in tokens
        overlap_tokens: Overlap size in tokens
        enc: Tiktoken encoder (or None for word-based estimation)

    Returns:
        List of text chunks
    """
    text = text.strip()
    if not text:
        return []

    if enc is None:
        # Word-based fallback
        words = text.split()
        words_per_chunk = max(50, int(chunk_tokens * 0.75))
        words_overlap = max(10, int(overlap_tokens * 0.75))
        chunks = []
        start = 0
        while start < len(words):
            end = min(len(words), start + words_per_chunk)
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            if end == len(words):
                break
            start = max(0, end - words_overlap)
        return chunks

    # Token-based chunking
    toks = enc.encode(text)
    chunks = []
    start = 0
    while start < len(toks):
        end = min(len(toks), start + chunk_tokens)
        chunk = enc.decode(toks[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(toks):
            break
        start = max(0, end - overlap_tokens)
    return chunks


def split_text_for_segment_aware_chunking(text: str) -> list[tuple[str, str]]:
    """
    Split page text into segment-aware blocks before token chunking.

    Returns:
        List of (segment_title, segment_text).
    """
    t = str(text or "").strip()
    if not t:
        return []

    # Add synthetic boundaries for flattened OCR text where headings/entities appear inline.
    # This reduces mixed-entity chunks (e.g., multiple IJB entities in one chunk).
    for patt in (
        r"(\b\d+\.\d+\.\d+\.\d+\b)",
        r"(?i)(\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\s+Integration Joint Board\s*\(IJB\)\b)",
        r"(?i)(\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\s+IJB\b)",
        r"(?i)(\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\s+IJB\s+reported\b)",
        r"(?i)(\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\s+Integration Joint Board\s*\(IJB\)\s+reported\b)",
    ):
        t = re.sub(patt, r"\n\1", t)

    lines = [ln.strip() for ln in re.split(r"\r?\n+", t) if ln.strip()]
    if not lines:
        return [("segment_000", t)]

    section_number = re.compile(r"^\d+(?:\.\d+){1,5}\b")
    ijb_heading = re.compile(r"(?i)\b(?:integration\s+joint\s+board|ijb)\b")
    ijb_entity_clause = re.compile(
        r"(?i)^[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\s+"
        r"(?:integration\s+joint\s+board\s*\(ijb\)|ijb)\b"
    )
    uppercase_heading = re.compile(r"^[A-Z][A-Z0-9 ,/&()\-]{8,}$")

    segments: list[tuple[str, list[str]]] = []
    cur_title = "segment_000"
    cur_lines: list[str] = []

    def _push() -> None:
        nonlocal cur_lines, cur_title
        if not cur_lines:
            return
        body = " ".join(cur_lines).strip()
        if body:
            segments.append((cur_title, cur_lines.copy()))
        cur_lines = []

    for i, ln in enumerate(lines):
        is_boundary = False
        if section_number.match(ln):
            is_boundary = True
        elif ijb_heading.search(ln):
            is_boundary = True
        elif ijb_entity_clause.match(ln):
            is_boundary = True
        elif uppercase_heading.match(ln) and len(ln.split()) <= 14:
            is_boundary = True

        if is_boundary:
            _push()
            # Keep segment title compact and deterministic.
            title = re.sub(r"\s+", " ", ln)[:96]
            cur_title = title if title else f"segment_{i:03d}"
        cur_lines.append(ln)

    _push()
    if not segments:
        return [("segment_000", t)]

    out: list[tuple[str, str]] = []
    for title, seg_lines in segments:
        body = " ".join(seg_lines).strip()
        if body:
            out.append((title, body))
    return out if out else [("segment_000", t)]
