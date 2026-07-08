import json
import tempfile
import unittest
from pathlib import Path

from src.run_experiment import run_phase1_vector_rag


class RunExperimentTest(unittest.TestCase):
    def test_phase1_runner_writes_per_query_and_aggregate_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "hotpotqa_small.jsonl"
            output_dir = root / "results"
            sample = {
                "sample_id": "s1",
                "question": "Where was Ada born?",
                "answer": "Paris",
                "contexts": [
                    {"doc_id": "Ada", "title": "Ada", "text": "Ada was born in Paris."},
                    {"doc_id": "Noise", "title": "Noise", "text": "The ocean is blue."},
                ],
                "supporting_facts": [{"doc_id": "Ada", "sent_id": 0}],
            }
            dataset_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            config = {
                "dataset_path": str(dataset_path),
                "output_dir": str(output_dir),
                "chunking": {"chunk_size": 32, "overlap": 0},
                "retrieval": {"top_k": 3},
                "embedding": {"type": "hashing", "dimensions": 64},
                "metrics": {"k_values": [1, 3]},
            }

            summary = run_phase1_vector_rag(config)

            self.assertTrue((output_dir / "per_query_results.csv").exists())
            self.assertTrue((output_dir / "aggregate_metrics.json").exists())
            self.assertIn("recall@3", summary)
            self.assertGreaterEqual(summary["recall@3"], 0.0)


if __name__ == "__main__":
    unittest.main()
