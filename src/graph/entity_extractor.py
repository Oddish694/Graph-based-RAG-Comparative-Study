from __future__ import annotations

import re
from typing import Iterable


CAPITALIZED_PHRASE_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9_]+)(?:\s+(?:[A-Z][A-Za-z0-9_]+))*\b")
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

STOP_ENTITIES = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
}


def normalize_entity(entity: str) -> str:
    tokens = TOKEN_RE.findall(entity.lower())
    return " ".join(tokens)


class SimpleEntityExtractor:
    def __init__(self, min_token_length: int = 2):
        self.min_token_length = min_token_length

    def extract(self, text: str) -> list[str]:
        entities: list[str] = []
        seen: set[str] = set()
        for match in CAPITALIZED_PHRASE_RE.finditer(text):
            entity = normalize_entity(match.group(0))
            if self._keep(entity) and entity not in seen:
                entities.append(entity)
                seen.add(entity)
        return entities

    def extract_many(self, texts: Iterable[str]) -> list[str]:
        entities: list[str] = []
        seen: set[str] = set()
        for text in texts:
            for entity in self.extract(text):
                if entity not in seen:
                    entities.append(entity)
                    seen.add(entity)
        return entities

    def _keep(self, entity: str) -> bool:
        if not entity or entity in STOP_ENTITIES:
            return False
        tokens = entity.split()
        if len(tokens) == 1:
            token = tokens[0]
            return len(token) >= self.min_token_length and token not in STOP_ENTITIES
        return any(token not in STOP_ENTITIES and len(token) >= self.min_token_length for token in tokens)
