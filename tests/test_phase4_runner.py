import json
import tempfile
import unittest
from pathlib import Path

from src.run_experiment import run_retrieval_experiment


class Phase4RunnerTest(unittest.TestCase):
    def test_phase4_improved_lightrag_runner_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "hotpotqa_small.jsonl"
            output_dir = root / "phase4_results"
            sample = {
                "sample_id": "s1",
                "question": "Who designed the Analytical Engine?",
                "answer": "Charles Babbage",
                "contexts": [
                    {"doc_id": "Ada", "title": "Ada", "text": "Ada Lovelace wrote notes about the Analytical Engine."},
                    {"doc_id": "Babbage", "title": "Babbage", "text": "Charles Babbage designed the Analytical Engine."},
                ],
                "supporting_facts": [{"doc_id": "Babbage", "sent_id": 0}],
            }
            dataset_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            config = {
                "dataset_path": str(dataset_path),
                "output_dir": str(output_dir),
                "chunking": {"chunk_size": 32, "overlap": 0},
                "embedding": {"type": "hashing", "dimensions": 64},
                "retrieval": {
                    "method": "improved_lightrag",
                    "top_k": 3,
                    "candidate_k": 6,
                    "seed_top_k": 2,
                    "coverage_weight": 0.1,
                },
                "metrics": {"k_values": [1, 3]},
            }

            summary = run_retrieval_experiment(config)

            self.assertTrue((output_dir / "per_query_results.csv").exists())
            self.assertTrue((output_dir / "aggregate_metrics.json").exists())
            self.assertEqual(summary["retriever"], "improved_lightrag")
            self.assertIn("graph_num_entities", summary)
            self.assertIn("full_evidence_recall@3", summary)


if __name__ == "__main__":
    unittest.main()
