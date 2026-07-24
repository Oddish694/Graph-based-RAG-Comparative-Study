from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Sequence
from itertools import combinations
import math
from typing import Any, Iterable

from src.graph.entity_extractor import normalize_entity


class GraphIndex:
    def __init__(self):
        self.chunks_by_id: dict[str, dict[str, Any]] = {}
        self.entity_to_chunk_ids: dict[str, set[str]] = defaultdict(set)
        self.chunk_to_entities: dict[str, set[str]] = defaultdict(set)
        self.entity_adjacency: dict[str, dict[str, float]] = defaultdict(dict)
        self.edge_sources: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
        self._entity_weight_cache: dict[str, float] = {}

    def add_chunk(
        self,
        chunk: dict[str, Any],
        entities: Iterable[str],
        entity_groups: Sequence[Iterable[str]] | None = None,
    ) -> None:
        chunk_id = str(chunk["chunk_id"])
        normalized_entities = {normalize_entity(entity) for entity in entities if normalize_entity(entity)}
        self._entity_weight_cache.clear()
        self.chunks_by_id[chunk_id] = dict(chunk)
        self.chunk_to_entities[chunk_id].update(normalized_entities)
        for entity in normalized_entities:
            self.entity_to_chunk_ids[entity].add(chunk_id)
            self.entity_adjacency.setdefault(entity, {})
        if entity_groups is None:
            entity_groups = [normalized_entities]
        for group in entity_groups:
            self.add_entity_group_edges(chunk_id, group)

    def add_entity_group_edges(self, chunk_id: str, entities: Iterable[str], weight: float = 1.0) -> None:
        normalized_entities = sorted({normalize_entity(entity) for entity in entities if normalize_entity(entity)})
        if len(normalized_entities) < 2:
            return
        group_weight = weight / max(1, len(normalized_entities) - 1)
        for left, right in combinations(normalized_entities, 2):
            self.entity_adjacency[left][right] = self.entity_adjacency[left].get(right, 0.0) + group_weight
            self.entity_adjacency[right][left] = self.entity_adjacency[right].get(left, 0.0) + group_weight
            edge_key = _edge_key(left, right)
            self.edge_sources[edge_key][chunk_id] = self.edge_sources[edge_key].get(chunk_id, 0.0) + group_weight

    def get_chunks_for_entities(self, entities: Iterable[str]) -> list[dict[str, Any]]:
        chunk_ids: set[str] = set()
        for entity in entities:
            chunk_ids.update(self.entity_to_chunk_ids.get(normalize_entity(entity), set()))
        return [dict(self.chunks_by_id[chunk_id]) for chunk_id in sorted(chunk_ids)]

    def neighbors(self, entity: str) -> set[str]:
        return set(self.entity_adjacency.get(normalize_entity(entity), {}))

    def edge_weight(self, left: str, right: str) -> float:
        return self.entity_adjacency.get(normalize_entity(left), {}).get(normalize_entity(right), 0.0)

    def edge_evidence(self, left: str, right: str, limit: int = 3) -> list[dict[str, Any]]:
        edge_key = _edge_key(normalize_entity(left), normalize_entity(right))
        source_weights = self.edge_sources.get(edge_key, {})
        ranked_sources = sorted(source_weights.items(), key=lambda item: (-item[1], item[0]))[:limit]
        evidence: list[dict[str, Any]] = []
        for chunk_id, weight in ranked_sources:
            chunk = self.chunks_by_id.get(chunk_id, {})
            evidence.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": chunk.get("doc_id", ""),
                    "title": chunk.get("title", ""),
                    "weight": weight,
                }
            )
        return evidence

    def entity_weight(self, entity: str) -> float:
        normalized = normalize_entity(entity)
        if normalized in self._entity_weight_cache:
            return self._entity_weight_cache[normalized]
        frequency = len(self.entity_to_chunk_ids.get(normalized, set()))
        if frequency <= 0:
            return 0.0
        weight = 1.0 / math.sqrt(frequency)
        self._entity_weight_cache[normalized] = weight
        return weight

    def shortest_entity_distances(
        self,
        entities: Iterable[str],
        depth: int = 1,
        max_neighbors_per_entity: int = 10,
    ) -> dict[str, int]:
        paths = self.shortest_entity_paths(
            entities,
            depth=depth,
            max_neighbors_per_entity=max_neighbors_per_entity,
        )
        return {entity: len(path) - 1 for entity, path in paths.items()}

    def shortest_entity_paths(
        self,
        entities: Iterable[str],
        depth: int = 1,
        max_neighbors_per_entity: int = 10,
    ) -> dict[str, list[str]]:
        start_entities = [normalize_entity(entity) for entity in entities if normalize_entity(entity)]
        distances: dict[str, int] = {entity: 0 for entity in start_entities}
        paths: dict[str, list[str]] = {entity: [entity] for entity in start_entities}
        queue: deque[tuple[str, int]] = deque((entity, 0) for entity in start_entities)

        while queue:
            entity, distance = queue.popleft()
            if distance >= depth:
                continue
            neighbors = sorted(
                self.entity_adjacency.get(entity, {}),
                key=lambda neighbor: (-self.edge_weight(entity, neighbor), -self.entity_weight(neighbor), neighbor),
            )[:max_neighbors_per_entity]
            for neighbor in neighbors:
                if neighbor not in distances:
                    distances[neighbor] = distance + 1
                    paths[neighbor] = paths[entity] + [neighbor]
                    queue.append((neighbor, distance + 1))
        return paths

    def expand_from_entities(
        self,
        entities: Iterable[str],
        depth: int = 1,
        max_neighbors_per_entity: int = 10,
    ) -> list[dict[str, Any]]:
        distances = self.shortest_entity_distances(
            entities,
            depth=depth,
            max_neighbors_per_entity=max_neighbors_per_entity,
        )
        return self.get_chunks_for_entities(distances)

    def expand_from_chunks(
        self,
        chunk_ids: Iterable[str],
        depth: int = 1,
        max_neighbors_per_entity: int = 10,
    ) -> list[dict[str, Any]]:
        entities: set[str] = set()
        for chunk_id in chunk_ids:
            entities.update(self.chunk_to_entities.get(str(chunk_id), set()))
        return self.expand_from_entities(entities, depth=depth, max_neighbors_per_entity=max_neighbors_per_entity)

    @property
    def num_entities(self) -> int:
        return len(self.entity_to_chunk_ids)

    @property
    def num_edges(self) -> int:
        return sum(len(neighbors) for neighbors in self.entity_adjacency.values()) // 2

    def __len__(self) -> int:
        return len(self.chunks_by_id)


def _edge_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))
