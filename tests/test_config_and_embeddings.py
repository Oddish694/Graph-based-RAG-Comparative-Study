import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.indexing.vector_index import SentenceTransformerEmbeddingModel
from src.run_experiment import build_embedding_model, load_config


class FakeSentenceTransformer:
    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = device

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        return np.array([[3.0, 4.0] for _ in texts], dtype=np.float32)


class ConfigAndEmbeddingsTest(unittest.TestCase):
    def test_load_config_accepts_real_yaml_syntax(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "phase1.yaml"
            path.write_text(
                "dataset_path: data/processed/hotpotqa_small.jsonl\n"
                "embedding:\n"
                "  type: hashing\n"
                "  dimensions: 32\n",
                encoding="utf-8",
            )

            config = load_config(path)

        self.assertEqual(config["dataset_path"], "data/processed/hotpotqa_small.jsonl")
        self.assertEqual(config["embedding"]["dimensions"], 32)

    def test_build_embedding_model_supports_sentence_transformers(self):
        with patch.dict("sys.modules", {"sentence_transformers": type("M", (), {"SentenceTransformer": FakeSentenceTransformer})}):
            model = build_embedding_model(
                {
                    "type": "sentence_transformers",
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "device": "cpu",
                }
            )

        self.assertIsInstance(model, SentenceTransformerEmbeddingModel)

    def test_sentence_transformer_adapter_normalizes_encoded_vectors(self):
        with patch.dict("sys.modules", {"sentence_transformers": type("M", (), {"SentenceTransformer": FakeSentenceTransformer})}):
            model = SentenceTransformerEmbeddingModel("fake-model", device="cpu")
            vectors = model.encode(["hello"])

        self.assertEqual(vectors.shape, (1, 2))
        self.assertAlmostEqual(float(np.linalg.norm(vectors[0])), 1.0)


if __name__ == "__main__":
    unittest.main()
