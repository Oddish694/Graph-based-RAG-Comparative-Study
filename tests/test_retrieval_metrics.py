import unittest

from src.evaluation.retrieval_metrics import (
    evaluate_query,
    evidence_hit_count_at_k,
    full_evidence_recall_at_k,
    mean_metrics,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    retrieved_context_tokens_at_k,
)


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

    def test_precision_at_k_counts_unique_relevant_results_over_k(self):
        retrieved = [
            {"doc_id": "A", "chunk_id": "A::0"},
            {"doc_id": "A", "chunk_id": "A::1"},
            {"doc_id": "B", "chunk_id": "B::0"},
            {"doc_id": "Noise", "chunk_id": "Noise::0"},
        ]
        gold = [{"doc_id": "A"}, {"doc_id": "B"}]

        self.assertAlmostEqual(precision_at_k(retrieved, gold, k=4), 0.5)

    def test_evidence_hit_count_and_full_evidence_recall(self):
        retrieved = [
            {"doc_id": "A", "chunk_id": "A::0"},
            {"doc_id": "Noise", "chunk_id": "Noise::0"},
            {"doc_id": "B", "chunk_id": "B::0"},
        ]
        gold = [{"doc_id": "A"}, {"doc_id": "B"}]

        self.assertEqual(evidence_hit_count_at_k(retrieved, gold, k=2), 1.0)
        self.assertEqual(full_evidence_recall_at_k(retrieved, gold, k=2), 0.0)
        self.assertEqual(full_evidence_recall_at_k(retrieved, gold, k=3), 1.0)

    def test_ndcg_rewards_relevant_documents_ranked_earlier(self):
        better = [
            {"doc_id": "A", "chunk_id": "A::0"},
            {"doc_id": "Noise", "chunk_id": "Noise::0"},
            {"doc_id": "B", "chunk_id": "B::0"},
        ]
        worse = [
            {"doc_id": "Noise", "chunk_id": "Noise::0"},
            {"doc_id": "A", "chunk_id": "A::0"},
            {"doc_id": "B", "chunk_id": "B::0"},
        ]
        gold = [{"doc_id": "A"}, {"doc_id": "B"}]

        self.assertGreater(ndcg_at_k(better, gold, k=3), ndcg_at_k(worse, gold, k=3))

    def test_retrieved_context_tokens_at_k_sums_simple_word_counts(self):
        retrieved = [
            {"text": "one two three"},
            {"text": "four five"},
            {"text": "ignored"},
        ]

        self.assertEqual(retrieved_context_tokens_at_k(retrieved, k=2), 5.0)

    def test_evaluate_query_returns_all_phase2_5_metrics(self):
        retrieved = [
            {"doc_id": "A", "chunk_id": "A::0", "text": "alpha beta"},
            {"doc_id": "B", "chunk_id": "B::0", "text": "gamma"},
        ]
        gold = [{"doc_id": "A"}, {"doc_id": "B"}]

        metrics = evaluate_query(retrieved, gold, k_values=[1, 2])

        self.assertEqual(metrics["recall@2"], 1.0)
        self.assertEqual(metrics["precision@2"], 1.0)
        self.assertEqual(metrics["evidence_hit_count@2"], 2.0)
        self.assertEqual(metrics["full_evidence_recall@2"], 1.0)
        self.assertEqual(metrics["retrieved_context_tokens@2"], 3.0)
        self.assertIn("ndcg@2", metrics)

    def test_evaluate_query_returns_zero_when_gold_is_empty(self):
        metrics = evaluate_query([{"doc_id": "A", "text": "some context"}], [], k_values=[1, 3])

        self.assertEqual(metrics["recall@1"], 0.0)
        self.assertEqual(metrics["mrr@3"], 0.0)
        self.assertEqual(metrics["hit_rate@3"], 0.0)
        self.assertEqual(metrics["precision@1"], 0.0)
        self.assertEqual(metrics["ndcg@1"], 0.0)
        self.assertEqual(metrics["full_evidence_recall@1"], 0.0)
        self.assertEqual(metrics["retrieved_context_tokens@1"], 2.0)

    def test_mean_metrics_averages_named_metric_rows(self):
        rows = [
            {"recall@5": 1.0, "mrr@5": 0.5},
            {"recall@5": 0.0, "mrr@5": 1.0},
        ]

        self.assertEqual(mean_metrics(rows), {"recall@5": 0.5, "mrr@5": 0.75})


if __name__ == "__main__":
    unittest.main()
