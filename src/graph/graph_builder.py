from __future__ import annotations

from src.graph.entity_extractor import SimpleEntityExtractor
from src.indexing.graph_index import GraphIndex


def build_graph_index(
    chunks: list[dict],
    entity_extractor: SimpleEntityExtractor | None = None,
) -> GraphIndex:
    extractor = entity_extractor or SimpleEntityExtractor()
    graph = GraphIndex()
    for chunk in chunks:
        title = str(chunk.get("title", ""))
        text = str(chunk.get("text", ""))
        entities = extractor.extract_many([title, text])
        graph.add_chunk(dict(chunk), entities)
    return graph
