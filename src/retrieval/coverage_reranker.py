from __future__ import annotations

from typing import Any


class CoverageAwareReranker:
    def __init__(
        self,
        coverage_weight: float = 0.15,
        entity_coverage_weight: float = 0.10,
        relevance_weight: float = 1.0,
    ):
        self.coverage_weight = coverage_weight
        self.entity_coverage_weight = entity_coverage_weight
        self.relevance_weight = relevance_weight

    def rerank(self, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0 or not candidates:
            return []

        remaining = [dict(candidate) for candidate in candidates]
        selected: list[dict[str, Any]] = []
        seen_docs: set[str] = set()
        seen_entities: set[str] = set()

        while remaining and len(selected) < top_k:
            scored = [
                (self._score(candidate, seen_docs, seen_entities), index, candidate)
                for index, candidate in enumerate(remaining)
            ]
            scored.sort(key=lambda item: item[0], reverse=True)
            _, index, best = scored[0]
            best = dict(best)
            coverage_score = self._coverage_score(best, seen_docs, seen_entities)
            best["coverage_score"] = coverage_score
            best["rerank_score"] = self._score(best, seen_docs, seen_entities)
            selected.append(best)
            if best.get("doc_id"):
                seen_docs.add(str(best["doc_id"]))
            seen_entities.update(str(entity) for entity in best.get("matched_entities", []) if entity)
            remaining.pop(index)

        return selected

    def _score(self, candidate: dict[str, Any], seen_docs: set[str], seen_entities: set[str]) -> float:
        base_score = float(candidate.get("score", 0.0))
        coverage_score = self._coverage_score(candidate, seen_docs, seen_entities)
        return self.relevance_weight * base_score + coverage_score

    def _coverage_score(self, candidate: dict[str, Any], seen_docs: set[str], seen_entities: set[str]) -> float:
        doc_score = 0.0
        doc_id = candidate.get("doc_id")
        if self.coverage_weight > 0.0 and doc_id and str(doc_id) not in seen_docs:
            doc_score = self.coverage_weight

        entity_score = 0.0
        entities = {str(entity) for entity in candidate.get("matched_entities", []) if entity}
        if self.entity_coverage_weight > 0.0 and entities:
            novel_entities = entities - seen_entities
            entity_score = self.entity_coverage_weight * (len(novel_entities) / len(entities))

        return doc_score + entity_score
