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


if __name__ == "__main__":
    unittest.main()
