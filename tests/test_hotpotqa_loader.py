import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.datasets.hotpotqa_loader import (
    DEFAULT_HOTPOTQA_DATASET,
    load_cached_jsonl,
    load_hotpotqa_small,
    normalize_hotpot_record,
    save_jsonl,
)
from src.datasets.schema import QASample


class HotpotQALoaderTest(unittest.TestCase):
    def test_normalize_hotpot_record_keeps_question_answer_contexts_and_supporting_facts(self):
        record = {
            "id": "sample-1",
            "question": "Where was the founder of Example Corp born?",
            "answer": "Paris",
            "context": [
                ["Example Corp", ["Example Corp was founded by Ada.", "It is a technology firm."]],
                ["Ada", ["Ada was born in Paris."]],
            ],
            "supporting_facts": [["Example Corp", 0], ["Ada", 0]],
        }

        sample = normalize_hotpot_record(record)

        self.assertIsInstance(sample, QASample)
        self.assertEqual(sample.sample_id, "sample-1")
        self.assertEqual(len(sample.contexts), 2)
        self.assertEqual(sample.supporting_facts[0]["doc_id"], "Example Corp")

    def test_jsonl_cache_round_trip(self):
        sample = QASample(
            sample_id="s1",
            question="Question?",
            answer="Answer",
            contexts=[{"doc_id": "Doc", "title": "Doc", "text": "Some context."}],
            supporting_facts=[{"doc_id": "Doc", "sent_id": 0}],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hotpotqa_small.jsonl"
            save_jsonl([sample], path)

            raw = path.read_text(encoding="utf-8").strip()
            self.assertEqual(json.loads(raw)["sample_id"], "s1")
            loaded = load_cached_jsonl(path)

        self.assertEqual(loaded, [sample])

    def test_loader_uses_namespaced_hotpotqa_dataset_by_default(self):
        fake_dataset = [
            {
                "id": "s1",
                "question": "Q?",
                "answer": "A",
                "context": [["Doc", ["Evidence."]]],
                "supporting_facts": [["Doc", 0]],
            }
        ]
        load_dataset = Mock(return_value=fake_dataset)
        fake_module = type("DatasetsModule", (), {"load_dataset": load_dataset})

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("sys.modules", {"datasets": fake_module}):
                samples = load_hotpotqa_small(Path(tmpdir) / "cache.jsonl", sample_size=1)

        load_dataset.assert_called_once_with(DEFAULT_HOTPOTQA_DATASET, "distractor", split="validation")
        self.assertEqual(samples[0].sample_id, "s1")


if __name__ == "__main__":
    unittest.main()
