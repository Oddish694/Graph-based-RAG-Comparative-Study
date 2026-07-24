from __future__ import annotations

import re

from src.graph.entity_extractor import SimpleEntityExtractor
from src.indexing.graph_index import GraphIndex


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def build_graph_index(
    chunks: list[dict],
    entity_extractor: SimpleEntityExtractor | None = None,
) -> GraphIndex:
    extractor = entity_extractor or SimpleEntityExtractor()
    graph = GraphIndex()
    for chunk in chunks:
        title = str(chunk.get("title", ""))
        text = str(chunk.get("text", ""))
        title_entities = extractor.extract(title)
        sentence_groups: list[list[str]] = []
        if len(title_entities) >= 2:
            sentence_groups.append(title_entities)
        for sentence in split_sentences(text):
            sentence_entities = extractor.extract(sentence)
            if sentence_entities:
                sentence_groups.append(title_entities + sentence_entities)
        entities = extractor.extract_many([title, text])
        graph.add_chunk(dict(chunk), entities, entity_groups=sentence_groups)
    return graph


def split_sentences(text: str) -> list[str]:
    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(text) if sentence.strip()]
    return sentences or ([text.strip()] if text.strip() else [])
