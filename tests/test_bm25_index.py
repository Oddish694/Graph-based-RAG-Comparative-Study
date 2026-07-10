import unittest

from src.indexing.bm25_index import BM25Index


class BM25IndexTest(unittest.TestCase):
    def test_bm25_ranks_exact_keyword_match_first(self):
        chunks = [
            {"chunk_id": "ada::0", "doc_id": "Ada", "text": "Ada Lovelace wrote notes."},
            {"chunk_id": "babbage::0", "doc_id": "Babbage", "text": "Charles Babbage designed the Analytical Engine."},
            {"chunk_id": "noise::0", "doc_id": "Noise", "text": "Paris is a city."},
        ]
        index = BM25Index()
        index.add(chunks)

        results = index.search("analytical engine", k=2)

        self.assertEqual(results[0]["doc_id"], "Babbage")
        self.assertLessEqual(len(results), 2)
        self.assertTrue({"chunk_id", "doc_id", "text", "score"}.issubset(results[0]))

    def test_bm25_returns_empty_for_blank_query(self):
        index = BM25Index()
        index.add([{"chunk_id": "a::0", "doc_id": "A", "text": "some text"}])

        self.assertEqual(index.search("", k=5), [])


if __name__ == "__main__":
    unittest.main()
