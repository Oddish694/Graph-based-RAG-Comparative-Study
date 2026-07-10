import unittest

from src.indexing.vector_index import HashingEmbeddingModel
from src.retrieval.hybrid_rag import HybridRAGRetriever


class HybridRetrievalTest(unittest.TestCase):
    def test_hybrid_retriever_fuses_bm25_and_dense_scores(self):
        chunks = [
            {"chunk_id": "lexical::0", "doc_id": "Lexical", "text": "rareterm exact evidence"},
            {"chunk_id": "semantic::0", "doc_id": "Semantic", "text": "evidence about related facts"},
            {"chunk_id": "noise::0", "doc_id": "Noise", "text": "unrelated text"},
        ]
        retriever = HybridRAGRetriever.from_chunks(
            chunks,
            embedding_model=HashingEmbeddingModel(dimensions=64),
            bm25_weight=0.8,
            dense_weight=0.2,
        )

        results = retriever.retrieve("rareterm", top_k=2)

        self.assertEqual(results[0]["doc_id"], "Lexical")
        self.assertLessEqual(len(results), 2)
        self.assertEqual(results[0]["retriever_source"], "hybrid")
        self.assertIn("bm25_score", results[0])
        self.assertIn("dense_score", results[0])

    def test_hybrid_retriever_supports_reciprocal_rank_fusion(self):
        chunks = [
            {"chunk_id": "a::0", "doc_id": "A", "text": "alpha beta"},
            {"chunk_id": "b::0", "doc_id": "B", "text": "gamma delta"},
        ]
        retriever = HybridRAGRetriever.from_chunks(
            chunks,
            embedding_model=HashingEmbeddingModel(dimensions=32),
            fusion="rrf",
        )

        results = retriever.retrieve("alpha", top_k=5)

        self.assertLessEqual(len(results), 2)
        self.assertTrue(all(row["retriever_source"] == "hybrid" for row in results))


if __name__ == "__main__":
    unittest.main()
