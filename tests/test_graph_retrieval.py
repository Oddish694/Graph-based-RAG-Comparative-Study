import unittest

from src.indexing.vector_index import HashingEmbeddingModel
from src.retrieval.graph_rag_style import GraphRAGStyleRetriever


class GraphRetrievalTest(unittest.TestCase):
    def test_graph_retriever_adds_graph_expanded_candidates(self):
        chunks = [
            {
                "chunk_id": "seed::0",
                "doc_id": "Seed",
                "title": "Seed",
                "text": "Ada Lovelace wrote notes about the Analytical Engine.",
            },
            {
                "chunk_id": "target::0",
                "doc_id": "Target",
                "title": "Target",
                "text": "Charles Babbage designed the Analytical Engine in London.",
            },
            {
                "chunk_id": "noise::0",
                "doc_id": "Noise",
                "title": "Noise",
                "text": "The ocean contains salt water.",
            },
        ]
        retriever = GraphRAGStyleRetriever.from_chunks(
            chunks,
            embedding_model=HashingEmbeddingModel(dimensions=64),
            seed_top_k=1,
            graph_weight=1.0,
            seed_weight=0.1,
        )

        results = retriever.retrieve("Who designed the Analytical Engine?", top_k=2, candidate_k=2)

        self.assertIn("Target", [row["doc_id"] for row in results])
        self.assertTrue(all(row["retriever_source"] == "graph_rag_style" for row in results))
        self.assertTrue(any(row.get("graph_score", 0.0) > 0.0 for row in results))

    def test_graph_retriever_exposes_explainable_graph_score_components(self):
        chunks = [
            {
                "chunk_id": "seed::0",
                "doc_id": "Seed",
                "title": "Seed",
                "text": "Ada Lovelace studied the Analytical Engine.",
            },
            {
                "chunk_id": "target::0",
                "doc_id": "Target",
                "title": "Target",
                "text": "Charles Babbage designed the Analytical Engine.",
            },
        ]
        retriever = GraphRAGStyleRetriever.from_chunks(
            chunks,
            embedding_model=HashingEmbeddingModel(dimensions=32),
            graph_weight=1.0,
            seed_weight=0.0,
        )

        results = retriever.retrieve("Who designed the Analytical Engine?", top_k=2, candidate_k=2)

        self.assertTrue(any(row.get("query_entity_score", 0.0) > 0.0 for row in results))
        self.assertTrue(any(row.get("graph_proximity_score", 0.0) > 0.0 for row in results))
        self.assertTrue(all("reachable_entities" in row for row in results))
        self.assertTrue(all("graph_paths" in row for row in results))

    def test_graph_retriever_records_path_edges(self):
        chunks = [
            {"chunk_id": "a::0", "doc_id": "A", "title": "A", "text": "Alpha Entity connects Bridge Entity."},
            {"chunk_id": "b::0", "doc_id": "B", "title": "B", "text": "Bridge Entity connects Target Entity."},
        ]
        retriever = GraphRAGStyleRetriever.from_chunks(
            chunks,
            embedding_model=HashingEmbeddingModel(dimensions=32),
            expansion_depth=1,
            graph_weight=1.0,
            seed_weight=0.0,
        )

        results = retriever.retrieve("Alpha Entity", top_k=2, candidate_k=2)

        path_rows = [row for row in results if row.get("graph_paths")]
        self.assertTrue(path_rows)
        self.assertIn("path_edges", path_rows[0]["graph_paths"][0])


if __name__ == "__main__":
    unittest.main()
