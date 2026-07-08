from __future__ import annotations

from typing import Any

from src.datasets.schema import QASample


def chunk_text(text: str, chunk_size: int = 256, overlap: int = 50) -> list[dict[str, Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    tokens = text.split()
    if not tokens:
        return []

    chunks: list[dict[str, Any]] = []
    step = chunk_size - overlap
    for start in range(0, len(tokens), step):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        if chunk_tokens:
            chunks.append({"text": " ".join(chunk_tokens), "start_token": start, "end_token": end})
        if end == len(tokens):
            break
    return chunks


def chunk_sample_contexts(sample: QASample, chunk_size: int = 256, overlap: int = 50) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for context in sample.contexts:
        doc_id = str(context["doc_id"])
        title = str(context.get("title", doc_id))
        doc_chunks = chunk_text(str(context.get("text", "")), chunk_size=chunk_size, overlap=overlap)
        for index, chunk in enumerate(doc_chunks):
            chunks.append(
                {
                    "sample_id": sample.sample_id,
                    "doc_id": doc_id,
                    "title": title,
                    "chunk_id": f"{doc_id}::{index}",
                    "chunk_index": index,
                    **chunk,
                }
            )
    return chunks
