from __future__ import annotations

from collections import defaultdict, deque
from itertools import combinations
from typing import Any, Iterable

from src.graph.entity_extractor import normalize_entity


class GraphIndex:
    def __init__(self):
        self.chunks_by_id: dict[str, dict[str, Any]] = {}
        self.entity_to_chunk_ids: dict[str, set[str]] = defaultdict(set)
        self.chunk_to_entities: dict[str, set[str]] = defaultdict(set)
        self.entity_adjacency: dict[str, set[str]] = defaultdict(set)

    def add_chunk(self, chunk: dict[str, Any], entities: Iterable[str]) -> None:
        chunk_id = str(chunk["chunk_id"])
        normalized_entities = {normalize_entity(entity) for entity in entities if normalize_entity(entity)}
        self.chunks_by_id[chunk_id] = dict(chunk)
        self.chunk_to_entities[chunk_id].update(normalized_entities)
        for entity in normalized_entities:
            self.entity_to_chunk_ids[entity].add(chunk_id)
            self.entity_adjacency.setdefault(entity, set())
        for left, right in combinations(sorted(normalized_entities), 2):
            self.entity_adjacency[left].add(right)
            self.entity_adjacency[right].add(left)

    def get_chunks_for_entities(self, entities: Iterable[str]) -> list[dict[str, Any]]:
        chunk_ids: set[str] = set()
        for entity in entities:
            chunk_ids.update(self.entity_to_chunk_ids.get(normalize_entity(entity), set()))
        return [dict(self.chunks_by_id[chunk_id]) for chunk_id in sorted(chunk_ids)]

    def neighbors(self, entity: str) -> set[str]:
        return set(self.entity_adjacency.get(normalize_entity(entity), set()))

    def expand_from_entities(
        self,
        entities: Iterable[str],
        depth: int = 1,
        max_neighbors_per_entity: int = 10,
    ) -> list[dict[str, Any]]:
        start_entities = [normalize_entity(entity) for entity in entities if normalize_entity(entity)]
        visited: set[str] = set(start_entities)
        queue: deque[tuple[str, int]] = deque((entity, 0) for entity in start_entities)

        while queue:
            entity, distance = queue.popleft()
            if distance >= depth:
                continue
            neighbors = sorted(self.entity_adjacency.get(entity, set()))[:max_neighbors_per_entity]
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

        return self.get_chunks_for_entities(visited)

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
