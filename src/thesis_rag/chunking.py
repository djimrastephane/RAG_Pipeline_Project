from __future__ import annotations

from dataclasses import dataclass

try:
    import tiktoken
except Exception:
    tiktoken = None


@dataclass(slots=True)
class TokenChunk:
    text: str
    token_count: int


def get_encoder():
    if tiktoken is None:
        raise RuntimeError("tiktoken is required for deterministic token chunking.")
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, encoder) -> int:
    return len(encoder.encode(text))


def chunk_text(text: str, chunk_size: int, overlap: int, encoder) -> list[TokenChunk]:
    stripped = text.strip()
    if not stripped:
        return []
    if overlap >= chunk_size:
        raise ValueError("chunk overlap must be smaller than chunk size")
    tokens = encoder.encode(stripped)
    chunks: list[TokenChunk] = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + chunk_size)
        chunk_text_value = encoder.decode(tokens[start:end]).strip()
        if chunk_text_value:
            chunks.append(TokenChunk(text=chunk_text_value, token_count=end - start))
        if end == len(tokens):
            break
        start = end - overlap
    return chunks
