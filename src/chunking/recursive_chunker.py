from __future__ import annotations

from src.chunking.fixed_chunker import chunk_text


def recursive_chunk_text(text: str, chunk_size: int = 256, overlap: int = 50) -> list[dict]:
    return chunk_text(text, chunk_size=chunk_size, overlap=overlap)
