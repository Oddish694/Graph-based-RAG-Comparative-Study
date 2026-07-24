from __future__ import annotations

import re
from typing import Iterable


CAPITALIZED_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9_'-]+)(?:\s+(?:of|and|the|for|in|on|de|la|van|von|[A-Z][A-Za-z0-9_'-]+))*\b"
)
ACRONYM_RE = re.compile(r"\b(?:[A-Z]\.){2,}|\b[A-Z]{2,}[A-Z0-9-]*\b")
QUOTED_PHRASE_RE = re.compile(r"[\"']([^\"']{3,80})[\"']")
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
    "with",
}

GENERIC_ALIAS_TAILS = {
    "album",
    "airport",
    "area",
    "association",
    "award",
    "book",
    "bridge",
    "building",
    "city",
    "college",
    "company",
    "computer",
    "county",
    "engine",
    "entity",
    "film",
    "group",
    "institute",
    "island",
    "lake",
    "league",
    "mountain",
    "museum",
    "park",
    "party",
    "river",
    "school",
    "series",
    "station",
    "team",
    "theatre",
    "university",
}


def normalize_entity(entity: str) -> str:
    tokens = TOKEN_RE.findall(entity.lower())
    while tokens and tokens[0] in {"a", "an", "the"}:
        tokens.pop(0)
    while tokens and tokens[-1] in STOP_ENTITIES:
        tokens.pop()
    return " ".join(tokens)


class SimpleEntityExtractor:
    def __init__(self, min_token_length: int = 2, include_aliases: bool = True):
        self.min_token_length = min_token_length
        self.include_aliases = include_aliases

    def extract(self, text: str) -> list[str]:
        entities: list[str] = []
        seen: set[str] = set()
        for raw_entity in self._candidate_entities(text):
            for entity in self._expand_aliases(normalize_entity(raw_entity)):
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

    def _candidate_entities(self, text: str) -> list[str]:
        candidates: list[str] = []
        seen_spans: set[tuple[int, int]] = set()
        for pattern in (ACRONYM_RE, CAPITALIZED_PHRASE_RE, QUOTED_PHRASE_RE):
            for match in pattern.finditer(text):
                span = match.span()
                if span in seen_spans:
                    continue
                raw = match.group(1) if pattern is QUOTED_PHRASE_RE else match.group(0)
                candidates.append(raw)
                seen_spans.add(span)
        return candidates

    def _expand_aliases(self, entity: str) -> list[str]:
        if not entity:
            return []
        aliases = [entity]
        tokens = entity.split()
        if not self.include_aliases or len(tokens) < 2:
            return aliases
        last_token = tokens[-1]
        if len(tokens) == 2 and len(last_token) >= 4 and last_token not in STOP_ENTITIES | GENERIC_ALIAS_TAILS:
            aliases.append(last_token)
        acronym = "".join(token[0] for token in tokens if token and token not in STOP_ENTITIES)
        if len(acronym) >= 2:
            aliases.append(acronym)
        return aliases
