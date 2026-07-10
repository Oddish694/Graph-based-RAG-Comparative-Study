import json
import tempfile
import unittest
from pathlib import Path

from src.run_experiment import run_phase2_hybrid_rag


class Phase2RunnerTest(unittest.TestCase):
    def test_phase2_hybrid_runner_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "hotpotqa_small.jsonl"
            output_dir = root / "phase2_results"
            sample = {
                "sample_id": "s1",
                "question": "Which engine did Babbage design?",
                "answer": "Analytical Engine",
                "contexts": [
                    {"doc_id": "Babbage", "title": "Babbage", "text": "Charles Babbage designed the Analytical Engine."},
                    {"doc_id": "Noise", "title": "Noise", "text": "Ada Lovelace was born in London."},
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
                    "method": "hybrid",
                    "top_k": 3,
                    "bm25_weight": 0.7,
                    "dense_weight": 0.3,
                    "fusion": "weighted",
                },
                "metrics": {"k_values": [1, 3]},
            }

            summary = run_phase2_hybrid_rag(config)

            self.assertTrue((output_dir / "per_query_results.csv").exists())
            self.assertTrue((output_dir / "aggregate_metrics.json").exists())
            self.assertIn("recall@3", summary)
            self.assertIn("avg_retrieval_latency_seconds", summary)


if __name__ == "__main__":
    unittest.main()
