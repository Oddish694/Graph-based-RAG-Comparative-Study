import json
import tempfile
import unittest
from pathlib import Path

from src.retrieval.lightrag_runner import (
    LightRAGControlledRetriever,
    LightRAGResultAdapter,
    LightRAGUnavailableError,
)
from src.run_experiment import run_retrieval_experiment


class LightRAGAdapterTest(unittest.TestCase):
    def test_adapter_normalizes_external_result_schema(self):
        adapter = LightRAGResultAdapter(
            chunk_lookup={
                "c1": {
                    "chunk_id": "c1",
                    "doc_id": "Ada",
                    "text": "Ada Lovelace studied the Analytical Engine.",
                }
            }
        )

        rows = adapter.normalize(
            {"contexts": [{"chunk_id": "c1", "score": 0.9}, {"doc_id": "Babbage", "content": "Babbage built machines."}]},
            top_k=2,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["chunk_id"], "c1")
        self.assertEqual(rows[0]["doc_id"], "Ada")
        self.assertEqual(rows[0]["score"], 0.9)
        self.assertEqual(rows[1]["doc_id"], "Babbage")
        self.assertEqual(rows[1]["retriever_source"], "lightrag_external")

    def test_local_compat_backend_returns_project_retrieval_schema(self):
        chunks = [
            {"chunk_id": "c1", "doc_id": "Ada", "text": "Ada Lovelace wrote about the Analytical Engine."},
            {"chunk_id": "c2", "doc_id": "Noise", "text": "The ocean is blue."},
        ]
        retriever = LightRAGControlledRetriever.from_chunks(chunks, backend="local_compat")

        rows = retriever.retrieve("Who wrote about the Analytical Engine?", top_k=1)

        self.assertEqual(len(rows), 1)
        self.assertIn("chunk_id", rows[0])
        self.assertIn("doc_id", rows[0])
        self.assertIn("score", rows[0])
        self.assertEqual(rows[0]["retriever_source"], "lightrag_local_compat")

    def test_external_backend_requires_optional_lightrag_package(self):
        chunks = [{"chunk_id": "c1", "doc_id": "Ada", "text": "Ada Lovelace."}]

        with self.assertRaises(LightRAGUnavailableError):
            LightRAGControlledRetriever.from_chunks(chunks, backend="external")


class Phase45RunnerTest(unittest.TestCase):
    def test_phase45_lightrag_runner_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "hotpotqa_small.jsonl"
            output_dir = root / "phase45_results"
            sample = {
                "sample_id": "s1",
                "question": "Who wrote about the Analytical Engine?",
                "answer": "Ada Lovelace",
                "contexts": [
                    {"doc_id": "Ada", "title": "Ada", "text": "Ada Lovelace wrote notes about the Analytical Engine."},
                    {"doc_id": "Noise", "title": "Noise", "text": "The ocean is blue."},
                ],
                "supporting_facts": [{"doc_id": "Ada", "sent_id": 0}],
            }
            dataset_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            config = {
                "dataset_path": str(dataset_path),
                "output_dir": str(output_dir),
                "chunking": {"chunk_size": 32, "overlap": 0},
                "embedding": {"type": "hashing", "dimensions": 64},
                "retrieval": {"method": "lightrag", "backend": "local_compat", "top_k": 3, "candidate_k": 6},
                "metrics": {"k_values": [1, 3]},
            }

            summary = run_retrieval_experiment(config)

            self.assertTrue((output_dir / "per_query_results.csv").exists())
            self.assertTrue((output_dir / "aggregate_metrics.json").exists())
            self.assertEqual(summary["retriever"], "lightrag")
            self.assertIn("recall@3", summary)


if __name__ == "__main__":
    unittest.main()
