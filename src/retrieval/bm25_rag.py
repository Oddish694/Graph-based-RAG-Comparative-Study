from __future__ import annotations

from typing import Any, Iterable

from src.indexing.bm25_index import BM25Index


class BM25RAGRetriever:
    def __init__(self, index: BM25Index):
        self.index = index

    @classmethod
    def from_chunks(cls, chunks: Iterable[dict[str, Any]]) -> "BM25RAGRetriever":
        index = BM25Index()
        index.add(chunks)
        return cls(index)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        results = self.index.search(query, k=top_k)
        for row in results:
            row["retriever_source"] = "bm25"
        return results
