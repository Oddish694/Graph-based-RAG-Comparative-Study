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
        max_seed_entities: int = 25,
        seed_weight: float = 0.7,
        graph_weight: float = 0.3,
        query_entity_weight: float = 0.45,
        seed_entity_weight: float = 0.25,
        expansion_weight: float = 0.30,
        distance_decay: float = 0.5,
    ):
        self.seed_retriever = seed_retriever
        self.graph_index = graph_index
        self.entity_extractor = entity_extractor or SimpleEntityExtractor()
        self.seed_top_k = seed_top_k
        self.expansion_depth = expansion_depth
        self.max_neighbors_per_entity = max_neighbors_per_entity
        self.max_seed_entities = max_seed_entities
        self.seed_weight = seed_weight
        self.graph_weight = graph_weight
        self.query_entity_weight = query_entity_weight
        self.seed_entity_weight = seed_entity_weight
        self.expansion_weight = expansion_weight
        self.distance_decay = distance_decay

    @classmethod
    def from_chunks(
        cls,
        chunks: Iterable[dict[str, Any]],
        embedding_model: Any | None = None,
        seed_top_k: int = 10,
        expansion_depth: int = 1,
        max_neighbors_per_entity: int = 10,
        max_seed_entities: int = 25,
        seed_weight: float = 0.7,
        graph_weight: float = 0.3,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        fusion: str = "weighted",
        rrf_k: int = 60,
        query_entity_weight: float = 0.45,
        seed_entity_weight: float = 0.25,
        expansion_weight: float = 0.30,
        distance_decay: float = 0.5,
        include_aliases: bool = True,
        alias_policy: str = "conservative",
        min_alias_token_length: int = 5,
    ) -> "GraphRAGStyleRetriever":
        chunk_list = [dict(chunk) for chunk in chunks]
        extractor = SimpleEntityExtractor(
            include_aliases=include_aliases,
            alias_policy=alias_policy,
            min_alias_token_length=min_alias_token_length,
        )
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
            max_seed_entities=max_seed_entities,
            seed_weight=seed_weight,
            graph_weight=graph_weight,
            query_entity_weight=query_entity_weight,
            seed_entity_weight=seed_entity_weight,
            expansion_weight=expansion_weight,
            distance_decay=distance_decay,
        )

    def retrieve(self, query: str, top_k: int = 5, candidate_k: int | None = None) -> list[dict[str, Any]]:
        seed_limit = candidate_k or max(self.seed_top_k, top_k)
        seed_results = self.seed_retriever.retrieve(query, top_k=seed_limit, candidate_k=seed_limit)
        query_entities = set(self.entity_extractor.extract(query))
        seed_chunk_ids = [str(row.get("chunk_id")) for row in seed_results if row.get("chunk_id")]
        seed_entities = set()
        for chunk_id in seed_chunk_ids[: self.seed_top_k]:
            seed_entities.update(self.graph_index.chunk_to_entities.get(chunk_id, set()))
        seed_entities = set(_top_weighted_entities(seed_entities, self.graph_index, self.max_seed_entities))

        graph_candidates_by_entity = self.graph_index.expand_from_entities(
            query_entities | seed_entities,
            depth=self.expansion_depth,
            max_neighbors_per_entity=self.max_neighbors_per_entity,
        )
        query_entity_paths = self.graph_index.shortest_entity_paths(
            query_entities,
            depth=self.expansion_depth,
            max_neighbors_per_entity=self.max_neighbors_per_entity,
        )
        seed_entity_paths = self.graph_index.shortest_entity_paths(
            seed_entities,
            depth=self.expansion_depth,
            max_neighbors_per_entity=self.max_neighbors_per_entity,
        )
        query_entity_distances = {entity: len(path) - 1 for entity, path in query_entity_paths.items()}
        seed_entity_distances = {entity: len(path) - 1 for entity, path in seed_entity_paths.items()}

        seed_scores = _normalize_scores(seed_results)
        merged = _merge_by_chunk_id(seed_results, graph_candidates_by_entity)
        focus_entities = query_entities | seed_entities
        for chunk_id, row in merged.items():
            chunk_entities = self.graph_index.chunk_to_entities.get(chunk_id, set())
            query_entity_score = _weighted_overlap_score(chunk_entities, query_entities, self.graph_index)
            seed_entity_score = _weighted_overlap_score(chunk_entities, seed_entities, self.graph_index)
            query_proximity_score = _distance_proximity_score(
                chunk_entities,
                query_entity_distances,
                self.graph_index,
                self.distance_decay,
            )
            seed_proximity_score = _distance_proximity_score(
                chunk_entities,
                seed_entity_distances,
                self.graph_index,
                self.distance_decay,
            )
            expansion_score = max(query_proximity_score, seed_proximity_score)
            graph_score = (
                self.query_entity_weight * query_entity_score
                + self.seed_entity_weight * seed_entity_score
                + self.expansion_weight * expansion_score
            )
            seed_score = seed_scores.get(chunk_id, 0.0)
            row["seed_score"] = seed_score
            row["graph_score"] = graph_score
            row["query_entity_score"] = query_entity_score
            row["seed_entity_score"] = seed_entity_score
            row["graph_proximity_score"] = expansion_score
            row["score"] = self.seed_weight * seed_score + self.graph_weight * graph_score
            row["retriever_source"] = "graph_rag_style"
            row["matched_entities"] = sorted(chunk_entities & focus_entities)
            row["reachable_entities"] = sorted(chunk_entities & (set(query_entity_distances) | set(seed_entity_distances)))
            row["graph_paths"] = _best_graph_paths(
                chunk_entities,
                query_entity_paths,
                seed_entity_paths,
                self.graph_index,
            )
        return sorted(merged.values(), key=lambda item: item["score"], reverse=True)[:top_k]


def _graph_score(chunk_entities: set[str], focus_entities: set[str]) -> float:
    if not chunk_entities or not focus_entities:
        return 0.0
    overlap = chunk_entities & focus_entities
    return len(overlap) / len(focus_entities)


def _top_weighted_entities(entities: set[str], graph_index: GraphIndex, limit: int) -> list[str]:
    if limit <= 0:
        return []
    return sorted(entities, key=lambda entity: (-graph_index.entity_weight(entity), entity))[:limit]


def _weighted_overlap_score(chunk_entities: set[str], focus_entities: set[str], graph_index: GraphIndex) -> float:
    if not chunk_entities or not focus_entities:
        return 0.0
    denominator = sum(graph_index.entity_weight(entity) for entity in focus_entities)
    if denominator <= 0.0:
        return 0.0
    numerator = sum(graph_index.entity_weight(entity) for entity in chunk_entities & focus_entities)
    return numerator / denominator


def _distance_proximity_score(
    chunk_entities: set[str],
    entity_distances: dict[str, int],
    graph_index: GraphIndex,
    distance_decay: float,
) -> float:
    if not chunk_entities or not entity_distances:
        return 0.0
    start_weight = sum(graph_index.entity_weight(entity) for entity, distance in entity_distances.items() if distance == 0)
    denominator = start_weight if start_weight > 0.0 else sum(graph_index.entity_weight(entity) for entity in entity_distances)
    if denominator <= 0.0:
        return 0.0
    numerator = 0.0
    for entity in chunk_entities & set(entity_distances):
        numerator += graph_index.entity_weight(entity) * (distance_decay ** entity_distances[entity])
    return min(1.0, numerator / denominator)


def _best_graph_paths(
    chunk_entities: set[str],
    query_entity_paths: dict[str, list[str]],
    seed_entity_paths: dict[str, list[str]],
    graph_index: GraphIndex,
    limit: int = 3,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for source_type, paths in (("query", query_entity_paths), ("seed", seed_entity_paths)):
        for entity in chunk_entities & set(paths):
            path = paths[entity]
            candidates.append(
                {
                    "source": source_type,
                    "target_entity": entity,
                    "distance": len(path) - 1,
                    "path": path,
                    "path_edges": _path_edges(path, graph_index),
                }
            )
    return sorted(candidates, key=lambda item: (item["distance"], item["source"], item["target_entity"]))[:limit]


def _path_edges(path: list[str], graph_index: GraphIndex) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for left, right in zip(path, path[1:]):
        edges.append(
            {
                "left": left,
                "right": right,
                "weight": graph_index.edge_weight(left, right),
                "evidence": graph_index.edge_evidence(left, right, limit=2),
            }
        )
    return edges


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
