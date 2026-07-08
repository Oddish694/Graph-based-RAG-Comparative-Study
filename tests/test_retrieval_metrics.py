import unittest

from src.evaluation.retrieval_metrics import evaluate_query, mean_metrics, mrr_at_k, recall_at_k


class RetrievalMetricsTest(unittest.TestCase):
    def test_recall_at_k_counts_unique_gold_documents_retrieved(self):
        retrieved = [
            {"doc_id": "A", "chunk_id": "A::0"},
            {"doc_id": "B", "chunk_id": "B::0"},
            {"doc_id": "A", "chunk_id": "A::1"},
        ]
        gold = [{"doc_id": "A"}, {"doc_id": "C"}]

        self.assertEqual(recall_at_k(retrieved, gold, k=3), 0.5)

    def test_mrr_at_k_uses_rank_of_first_relevant_result(self):
        retrieved = [
            {"doc_id": "X", "chunk_id": "X::0"},
            {"doc_id": "B", "chunk_id": "B::0"},
            {"doc_id": "C", "chunk_id": "C::0"},
        ]
        gold = [{"doc_id": "B"}, {"doc_id": "C"}]

        self.assertEqual(mrr_at_k(retrieved, gold, k=3), 0.5)

    def test_evaluate_query_returns_zero_when_gold_is_empty(self):
        metrics = evaluate_query([{"doc_id": "A"}], [], k_values=[1, 3])

        self.assertEqual(metrics["recall@1"], 0.0)
        self.assertEqual(metrics["mrr@3"], 0.0)
        self.assertEqual(metrics["hit_rate@3"], 0.0)

    def test_mean_metrics_averages_named_metric_rows(self):
        rows = [
            {"recall@5": 1.0, "mrr@5": 0.5},
            {"recall@5": 0.0, "mrr@5": 1.0},
        ]

        self.assertEqual(mean_metrics(rows), {"recall@5": 0.5, "mrr@5": 0.75})


if __name__ == "__main__":
    unittest.main()
