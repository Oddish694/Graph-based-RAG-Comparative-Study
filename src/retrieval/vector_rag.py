from __future__ import annotations

from typing import Any, Iterable

from src.indexing.vector_index import HashingEmbeddingModel, VectorIndex


class VectorRAGRetriever:
    def __init__(self, index: VectorIndex):
        self.index = index

    @classmethod
    def from_chunks(
        cls,
        chunks: Iterable[dict[str, Any]],
        embedding_model: Any | None = None,
    ) -> "VectorRAGRetriever":
        index = VectorIndex(embedding_model or HashingEmbeddingModel())
        index.add(chunks)
        return cls(index)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self.index.search(query, k=top_k)
