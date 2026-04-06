from __future__ import annotations
import re
from dataclasses import dataclass

try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None


@dataclass(frozen=True)
class SegmentBlock:
    title: str
    text: str
    boundary_type: str
    segment_has_search_hit: bool = False


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


def split_text_for_segment_aware_chunking(text: str) -> list[SegmentBlock]:
    """
    Split page text into segment-aware blocks before token chunking.

    Returns:
        List of segment blocks.
    """
    from .config import DEFAULT_CONFIG

    return split_text_for_segment_aware_chunking_with_patterns(
        text,
        insert_patterns=tuple(DEFAULT_CONFIG.SEGMENT_BOUNDARY_INSERT_PATTERNS),
        boundary_match_patterns=tuple(DEFAULT_CONFIG.SEGMENT_BOUNDARY_MATCH_PATTERNS),
        boundary_search_patterns=tuple(DEFAULT_CONFIG.SEGMENT_BOUNDARY_SEARCH_PATTERNS),
        uppercase_heading_pattern=str(DEFAULT_CONFIG.SEGMENT_UPPERCASE_HEADING_PATTERN),
        uppercase_heading_max_words=int(DEFAULT_CONFIG.SEGMENT_UPPERCASE_HEADING_MAX_WORDS),
    )


def split_text_for_segment_aware_chunking_with_patterns(
    text: str,
    *,
    insert_patterns: tuple[str, ...] = (),
    boundary_match_patterns: tuple[str, ...] = (),
    boundary_search_patterns: tuple[str, ...] = (),
    uppercase_heading_pattern: str = r"^[A-Z][A-Z0-9 ,/&()\-]{8,}$",
    uppercase_heading_max_words: int = 14,
) -> list[SegmentBlock]:
    """
    Split page text into segment-aware blocks before token chunking.

    The caller controls which regexes create synthetic line breaks and which
    line-level regexes count as segment boundaries.
    """
    t = str(text or "").strip()
    if not t:
        return []

    # Add synthetic boundaries for flattened OCR text where headings/entities appear inline.
    for patt in insert_patterns:
        t = re.sub(patt, r"\n\1", t)

    lines = [ln.strip() for ln in re.split(r"\r?\n+", t) if ln.strip()]
    if not lines:
        return [SegmentBlock(title="segment_000", text=t, boundary_type="CONTINUATION", segment_has_search_hit=False)]

    boundary_matchers = [re.compile(patt) for patt in boundary_match_patterns]
    boundary_searchers = [re.compile(patt) for patt in boundary_search_patterns]
    insert_matchers = [re.compile(patt) for patt in insert_patterns]
    uppercase_heading = re.compile(uppercase_heading_pattern)

    segments: list[SegmentBlock] = []
    cur_title = "segment_000"
    cur_lines: list[str] = []
    cur_boundary_type = "CONTINUATION"
    cur_has_search_hit = False

    def _push() -> None:
        nonlocal cur_lines, cur_title, cur_boundary_type, cur_has_search_hit
        if not cur_lines:
            return
        body = " ".join(cur_lines).strip()
        if body:
            segments.append(
                SegmentBlock(
                    title=cur_title,
                    text=body,
                    boundary_type=cur_boundary_type,
                    segment_has_search_hit=bool(cur_has_search_hit),
                )
            )
        cur_lines = []
        cur_has_search_hit = False

    for i, ln in enumerate(lines):
        is_boundary = any(patt.match(ln) for patt in boundary_matchers)
        boundary_type = "MATCH" if is_boundary else "CONTINUATION"
        if is_boundary and any(patt.match(ln) for patt in insert_matchers):
            boundary_type = "INSERT"
        if not is_boundary and uppercase_heading.match(ln) and len(ln.split()) <= uppercase_heading_max_words:
            is_boundary = True
            boundary_type = "MATCH"
        line_has_search_hit = any(patt.search(ln) for patt in boundary_searchers)

        if is_boundary:
            _push()
            # Keep segment title compact and deterministic.
            title = re.sub(r"\s+", " ", ln)[:96]
            cur_title = title if title else f"segment_{i:03d}"
            cur_boundary_type = boundary_type
        cur_lines.append(ln)
        cur_has_search_hit = bool(cur_has_search_hit or line_has_search_hit)

    _push()
    if not segments:
        return [SegmentBlock(title="segment_000", text=t, boundary_type="CONTINUATION", segment_has_search_hit=False)]
    return segments
