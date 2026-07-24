import unittest

from src.indexing.vector_index import HashingEmbeddingModel
from src.retrieval.improved_lightrag import ImprovedLightRAGRetriever


class ImprovedLightRAGTest(unittest.TestCase):
    def test_improved_lightrag_returns_reranked_graph_candidates(self):
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
        retriever = ImprovedLightRAGRetriever.from_chunks(
            chunks,
            embedding_model=HashingEmbeddingModel(dimensions=64),
            seed_top_k=2,
            expansion_depth=1,
            candidate_pool_size=5,
            coverage_weight=0.2,
        )

        results = retriever.retrieve("Who designed the Analytical Engine?", top_k=2, candidate_k=5)

        self.assertLessEqual(len(results), 2)
        self.assertTrue(all(row["retriever_source"] == "improved_lightrag" for row in results))
        self.assertTrue(any(row["doc_id"] == "Target" for row in results))
        self.assertIn("rerank_score", results[0])
        self.assertIn("coverage_score", results[0])

    def test_improved_lightrag_can_disable_coverage_reranking(self):
        chunks = [
            {"chunk_id": "a::0", "doc_id": "A", "title": "A", "text": "Alpha Entity connects Bridge Entity."},
            {"chunk_id": "b::0", "doc_id": "B", "title": "B", "text": "Bridge Entity connects Target Entity."},
        ]
        retriever = ImprovedLightRAGRetriever.from_chunks(
            chunks,
            embedding_model=HashingEmbeddingModel(dimensions=32),
            use_coverage_reranking=False,
        )

        results = retriever.retrieve("Alpha Entity", top_k=2)

        self.assertTrue(all(row["retriever_source"] == "improved_lightrag" for row in results))


if __name__ == "__main__":
    unittest.main()
