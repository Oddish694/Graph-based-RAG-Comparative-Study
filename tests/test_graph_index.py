import unittest

from src.graph.entity_extractor import SimpleEntityExtractor
from src.graph.graph_builder import build_graph_index


class GraphIndexTest(unittest.TestCase):
    def test_build_graph_index_maps_entities_to_chunks(self):
        chunks = [
            {"chunk_id": "ada::0", "doc_id": "Ada", "title": "Ada", "text": "Ada Lovelace studied the Analytical Engine."},
            {"chunk_id": "babbage::0", "doc_id": "Babbage", "title": "Babbage", "text": "Charles Babbage designed the Analytical Engine."},
            {"chunk_id": "noise::0", "doc_id": "Noise", "title": "Noise", "text": "Paris is a city."},
        ]

        graph = build_graph_index(chunks, SimpleEntityExtractor())

        results = graph.get_chunks_for_entities(["analytical engine"])
        self.assertEqual({row["chunk_id"] for row in results}, {"ada::0", "babbage::0"})
        self.assertIn("charles babbage", graph.neighbors("analytical engine"))
        self.assertGreaterEqual(graph.num_entities, 3)

    def test_expand_from_chunks_returns_entity_neighbors(self):
        chunks = [
            {"chunk_id": "a::0", "doc_id": "A", "title": "A", "text": "Alpha Entity connects to Bridge Entity."},
            {"chunk_id": "b::0", "doc_id": "B", "title": "B", "text": "Bridge Entity connects to Target Entity."},
        ]
        graph = build_graph_index(chunks, SimpleEntityExtractor())

        expanded = graph.expand_from_chunks(["a::0"], depth=1, max_neighbors_per_entity=5)

        self.assertIn("b::0", {row["chunk_id"] for row in expanded})

    def test_graph_index_tracks_entity_weight_and_distances(self):
        chunks = [
            {"chunk_id": "a::0", "doc_id": "A", "title": "A", "text": "Rare Entity connects Common Entity."},
            {"chunk_id": "b::0", "doc_id": "B", "title": "B", "text": "Common Entity connects Bridge Entity."},
            {"chunk_id": "c::0", "doc_id": "C", "title": "C", "text": "Common Entity connects Target Entity."},
        ]
        graph = build_graph_index(chunks, SimpleEntityExtractor())

        self.assertGreater(graph.entity_weight("rare entity"), graph.entity_weight("common entity"))
        distances = graph.shortest_entity_distances(["rare entity"], depth=2, max_neighbors_per_entity=10)
        self.assertEqual(distances["rare entity"], 0)
        self.assertIn("bridge entity", distances)

    def test_sentence_level_edges_do_not_connect_unrelated_sentences(self):
        chunks = [
            {
                "chunk_id": "doc::0",
                "doc_id": "Doc",
                "title": "Doc",
                "text": "Alpha Entity appears here. Target Entity appears later.",
            }
        ]
        graph = build_graph_index(chunks, SimpleEntityExtractor())

        self.assertEqual(graph.edge_weight("alpha entity", "target entity"), 0.0)

    def test_weighted_edges_keep_evidence_sources(self):
        chunks = [
            {
                "chunk_id": "doc::0",
                "doc_id": "Doc",
                "title": "Doc",
                "text": "Alpha Entity connects Bridge Entity.",
            }
        ]
        graph = build_graph_index(chunks, SimpleEntityExtractor())

        self.assertGreater(graph.edge_weight("alpha entity", "bridge entity"), 0.0)
        evidence = graph.edge_evidence("alpha entity", "bridge entity")
        self.assertEqual(evidence[0]["chunk_id"], "doc::0")


if __name__ == "__main__":
    unittest.main()
