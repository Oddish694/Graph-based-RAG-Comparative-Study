import unittest

from src.indexing.vector_index import HashingEmbeddingModel, VectorIndex
from src.retrieval.vector_rag import VectorRAGRetriever


class VectorRetrievalTest(unittest.TestCase):
    def test_vector_index_returns_top_k_with_scores_and_metadata(self):
        chunks = [
            {"chunk_id": "doc1::0", "doc_id": "doc1", "text": "Paris is the capital of France."},
            {"chunk_id": "doc2::0", "doc_id": "doc2", "text": "The Pacific Ocean is large."},
            {"chunk_id": "doc3::0", "doc_id": "doc3", "text": "France has many museums in Paris."},
        ]
        index = VectorIndex(HashingEmbeddingModel(dimensions=64))
        index.add(chunks)

        results = index.search("capital city of France", k=2)

        self.assertLessEqual(len(results), 2)
        self.assertTrue(all({"chunk_id", "doc_id", "text", "score"}.issubset(row) for row in results))
        self.assertEqual(results[0]["doc_id"], "doc1")

    def test_vector_rag_retriever_preserves_index_metadata(self):
        chunks = [
            {"chunk_id": "alpha::0", "doc_id": "alpha", "title": "Alpha", "text": "graph retrieval evidence"},
            {"chunk_id": "beta::0", "doc_id": "beta", "title": "Beta", "text": "unrelated text"},
        ]
        retriever = VectorRAGRetriever.from_chunks(chunks, embedding_model=HashingEmbeddingModel(dimensions=32))

        results = retriever.retrieve("graph evidence", top_k=5)

        self.assertEqual(results[0]["chunk_id"], "alpha::0")
        self.assertEqual(results[0]["title"], "Alpha")
        self.assertLessEqual(len(results), 5)


if __name__ == "__main__":
    unittest.main()
