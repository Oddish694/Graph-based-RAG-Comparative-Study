from __future__ import annotations

from typing import Any, Iterable

from src.indexing.bm25_index import BM25Index
from src.indexing.vector_index import HashingEmbeddingModel, VectorIndex


class HybridRAGRetriever:
    def __init__(
        self,
        bm25_index: BM25Index,
        vector_index: VectorIndex,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        fusion: str = "weighted",
        rrf_k: int = 60,
    ):
        self.bm25_index = bm25_index
        self.vector_index = vector_index
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.fusion = fusion
        self.rrf_k = rrf_k

    @classmethod
    def from_chunks(
        cls,
        chunks: Iterable[dict[str, Any]],
        embedding_model: Any | None = None,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        fusion: str = "weighted",
        rrf_k: int = 60,
    ) -> "HybridRAGRetriever":
        chunk_list = [dict(chunk) for chunk in chunks]
        bm25_index = BM25Index()
        bm25_index.add(chunk_list)
        vector_index = VectorIndex(embedding_model or HashingEmbeddingModel())
        vector_index.add(chunk_list)
        return cls(
            bm25_index=bm25_index,
            vector_index=vector_index,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
            fusion=fusion,
            rrf_k=rrf_k,
        )

    def retrieve(self, query: str, top_k: int = 5, candidate_k: int | None = None) -> list[dict[str, Any]]:
        candidate_limit = candidate_k or max(top_k * 4, top_k)
        bm25_results = self.bm25_index.search(query, k=candidate_limit)
        dense_results = self.vector_index.search(query, k=candidate_limit)
        if self.fusion == "rrf":
            return self._reciprocal_rank_fusion(bm25_results, dense_results, top_k)
        if self.fusion == "weighted":
            return self._weighted_fusion(bm25_results, dense_results, top_k)
        raise ValueError(f"Unsupported hybrid fusion strategy: {self.fusion}")

    def _weighted_fusion(
        self,
        bm25_results: list[dict[str, Any]],
        dense_results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        bm25_scores = _normalize_scores(bm25_results)
        dense_scores = _normalize_scores(dense_results)
        merged = _merge_by_chunk_id(bm25_results, dense_results)
        for chunk_id, row in merged.items():
            row["bm25_score"] = bm25_scores.get(chunk_id, 0.0)
            row["dense_score"] = dense_scores.get(chunk_id, 0.0)
            row["score"] = self.bm25_weight * row["bm25_score"] + self.dense_weight * row["dense_score"]
            row["retriever_source"] = "hybrid"
        return sorted(merged.values(), key=lambda item: item["score"], reverse=True)[:top_k]

    def _reciprocal_rank_fusion(
        self,
        bm25_results: list[dict[str, Any]],
        dense_results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        merged = _merge_by_chunk_id(bm25_results, dense_results)
        bm25_ranks = {row["chunk_id"]: rank for rank, row in enumerate(bm25_results, start=1)}
        dense_ranks = {row["chunk_id"]: rank for rank, row in enumerate(dense_results, start=1)}
        for chunk_id, row in merged.items():
            bm25_score = 1.0 / (self.rrf_k + bm25_ranks[chunk_id]) if chunk_id in bm25_ranks else 0.0
            dense_score = 1.0 / (self.rrf_k + dense_ranks[chunk_id]) if chunk_id in dense_ranks else 0.0
            row["bm25_score"] = bm25_score
            row["dense_score"] = dense_score
            row["score"] = bm25_score + dense_score
            row["retriever_source"] = "hybrid"
        return sorted(merged.values(), key=lambda item: item["score"], reverse=True)[:top_k]


def _normalize_scores(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {}
    raw_scores = [float(row.get("score", 0.0)) for row in results]
    min_score = min(raw_scores)
    max_score = max(raw_scores)
    if max_score == min_score:
        return {row["chunk_id"]: 1.0 for row in results}
    return {
        row["chunk_id"]: (float(row.get("score", 0.0)) - min_score) / (max_score - min_score)
        for row in results
    }


def _merge_by_chunk_id(*result_sets: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for results in result_sets:
        for row in results:
            chunk_id = row["chunk_id"]
            if chunk_id not in merged:
                merged[chunk_id] = {key: value for key, value in row.items() if key != "score"}
    return merged
