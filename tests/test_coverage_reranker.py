import unittest

from src.retrieval.coverage_reranker import CoverageAwareReranker


class CoverageRerankerTest(unittest.TestCase):
    def test_reranker_promotes_new_documents_when_scores_are_close(self):
        candidates = [
            {"chunk_id": "a::0", "doc_id": "A", "score": 1.00, "matched_entities": ["alpha"]},
            {"chunk_id": "a::1", "doc_id": "A", "score": 0.99, "matched_entities": ["alpha"]},
            {"chunk_id": "b::0", "doc_id": "B", "score": 0.98, "matched_entities": ["beta"]},
        ]
        reranker = CoverageAwareReranker(coverage_weight=0.5, entity_coverage_weight=0.5)

        results = reranker.rerank(candidates, top_k=2)

        self.assertEqual([row["doc_id"] for row in results], ["A", "B"])
        self.assertIn("coverage_score", results[1])
        self.assertIn("rerank_score", results[1])

    def test_reranker_keeps_base_order_when_coverage_is_disabled(self):
        candidates = [
            {"chunk_id": "a::0", "doc_id": "A", "score": 1.00, "matched_entities": ["alpha"]},
            {"chunk_id": "b::0", "doc_id": "B", "score": 0.50, "matched_entities": ["beta"]},
        ]
        reranker = CoverageAwareReranker(coverage_weight=0.0, entity_coverage_weight=0.0)

        results = reranker.rerank(candidates, top_k=2)

        self.assertEqual([row["chunk_id"] for row in results], ["a::0", "b::0"])
        self.assertEqual(results[0]["coverage_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
