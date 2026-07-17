from __future__ import annotations

from typing import Any, Iterable

from src.graph.entity_extractor import SimpleEntityExtractor
from src.graph.graph_builder import build_graph_index
from src.indexing.graph_index import GraphIndex
from src.indexing.vector_index import HashingEmbeddingModel
from src.retrieval.hybrid_rag import HybridRAGRetriever


class GraphRAGStyleRetriever:
    def __init__(
        self,
        seed_retriever: HybridRAGRetriever,
        graph_index: GraphIndex,
        entity_extractor: SimpleEntityExtractor | None = None,
        seed_top_k: int = 10,
        expansion_depth: int = 1,
        max_neighbors_per_entity: int = 10,
        seed_weight: float = 0.7,
        graph_weight: float = 0.3,
    ):
        self.seed_retriever = seed_retriever
        self.graph_index = graph_index
        self.entity_extractor = entity_extractor or SimpleEntityExtractor()
        self.seed_top_k = seed_top_k
        self.expansion_depth = expansion_depth
        self.max_neighbors_per_entity = max_neighbors_per_entity
        self.seed_weight = seed_weight
        self.graph_weight = graph_weight

    @classmethod
    def from_chunks(
        cls,
        chunks: Iterable[dict[str, Any]],
        embedding_model: Any | None = None,
        seed_top_k: int = 10,
        expansion_depth: int = 1,
        max_neighbors_per_entity: int = 10,
        seed_weight: float = 0.7,
        graph_weight: float = 0.3,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        fusion: str = "weighted",
        rrf_k: int = 60,
    ) -> "GraphRAGStyleRetriever":
        chunk_list = [dict(chunk) for chunk in chunks]
        extractor = SimpleEntityExtractor()
        graph_index = build_graph_index(chunk_list, extractor)
        seed_retriever = HybridRAGRetriever.from_chunks(
            chunk_list,
            embedding_model=embedding_model or HashingEmbeddingModel(),
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
            fusion=fusion,
            rrf_k=rrf_k,
        )
        return cls(
            seed_retriever=seed_retriever,
            graph_index=graph_index,
            entity_extractor=extractor,
            seed_top_k=seed_top_k,
            expansion_depth=expansion_depth,
            max_neighbors_per_entity=max_neighbors_per_entity,
            seed_weight=seed_weight,
            graph_weight=graph_weight,
        )

    def retrieve(self, query: str, top_k: int = 5, candidate_k: int | None = None) -> list[dict[str, Any]]:
        seed_limit = candidate_k or max(self.seed_top_k, top_k)
        seed_results = self.seed_retriever.retrieve(query, top_k=seed_limit, candidate_k=seed_limit)
        query_entities = set(self.entity_extractor.extract(query))
        seed_chunk_ids = [str(row.get("chunk_id")) for row in seed_results if row.get("chunk_id")]
        seed_entities = set()
        for chunk_id in seed_chunk_ids[: self.seed_top_k]:
            seed_entities.update(self.graph_index.chunk_to_entities.get(chunk_id, set()))

        graph_candidates_by_entity = self.graph_index.expand_from_entities(
            query_entities | seed_entities,
            depth=self.expansion_depth,
            max_neighbors_per_entity=self.max_neighbors_per_entity,
        )
        graph_candidates_by_seed = self.graph_index.expand_from_chunks(
            seed_chunk_ids[: self.seed_top_k],
            depth=self.expansion_depth,
            max_neighbors_per_entity=self.max_neighbors_per_entity,
        )

        seed_scores = _normalize_scores(seed_results)
        merged = _merge_by_chunk_id(seed_results, graph_candidates_by_entity, graph_candidates_by_seed)
        focus_entities = query_entities | seed_entities
        for chunk_id, row in merged.items():
            chunk_entities = self.graph_index.chunk_to_entities.get(chunk_id, set())
            graph_score = _graph_score(chunk_entities, focus_entities)
            seed_score = seed_scores.get(chunk_id, 0.0)
            row["seed_score"] = seed_score
            row["graph_score"] = graph_score
            row["score"] = self.seed_weight * seed_score + self.graph_weight * graph_score
            row["retriever_source"] = "graph_rag_style"
            row["matched_entities"] = sorted(chunk_entities & focus_entities)
        return sorted(merged.values(), key=lambda item: item["score"], reverse=True)[:top_k]


def _graph_score(chunk_entities: set[str], focus_entities: set[str]) -> float:
    if not chunk_entities or not focus_entities:
        return 0.0
    overlap = chunk_entities & focus_entities
    return len(overlap) / len(focus_entities)


def _normalize_scores(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {}
    raw_scores = [float(row.get("score", 0.0)) for row in results]
    min_score = min(raw_scores)
    max_score = max(raw_scores)
    if max_score == min_score:
        return {str(row["chunk_id"]): 1.0 for row in results if row.get("chunk_id")}
    return {
        str(row["chunk_id"]): (float(row.get("score", 0.0)) - min_score) / (max_score - min_score)
        for row in results
        if row.get("chunk_id")
    }


def _merge_by_chunk_id(*result_sets: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for results in result_sets:
        for row in results:
            chunk_id = str(row["chunk_id"])
            if chunk_id not in merged:
                merged[chunk_id] = {key: value for key, value in row.items() if key != "score"}
    return merged
