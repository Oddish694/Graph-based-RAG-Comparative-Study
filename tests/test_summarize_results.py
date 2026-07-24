import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.summarize_results import summarize_aggregate_metrics


class SummarizeResultsTest(unittest.TestCase):
    def test_summarizes_aggregate_metrics_to_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result_dir = root / "phase5_variant"
            result_dir.mkdir()
            (result_dir / "aggregate_metrics.json").write_text(
                json.dumps(
                    {
                        "retriever": "improved_lightrag",
                        "recall@5": 0.8,
                        "full_evidence_recall@5": 0.6,
                        "avg_retrieval_latency_seconds": 0.1,
                    }
                ),
                encoding="utf-8",
            )
            output_path = root / "ablation_table.csv"

            rows = summarize_aggregate_metrics([result_dir], output_path)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["variant"], "phase5_variant")
            self.assertEqual(rows[0]["recall@5"], 0.8)
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
