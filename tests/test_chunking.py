import unittest

from src.chunking.fixed_chunker import chunk_sample_contexts
from src.datasets.schema import QASample


class ChunkingTest(unittest.TestCase):
    def test_fixed_chunker_creates_stable_non_empty_chunks(self):
        sample = QASample(
            sample_id="s1",
            question="Question?",
            answer="Answer",
            contexts=[
                {
                    "doc_id": "Doc A",
                    "title": "Doc A",
                    "text": "one two three four five six seven",
                }
            ],
            supporting_facts=[],
        )

        chunks = chunk_sample_contexts(sample, chunk_size=3, overlap=1)

        self.assertEqual([chunk["chunk_id"] for chunk in chunks], ["Doc A::0", "Doc A::1", "Doc A::2"])
        self.assertTrue(all(chunk["text"] for chunk in chunks))
        self.assertEqual(chunks[1]["text"], "three four five")


if __name__ == "__main__":
    unittest.main()
