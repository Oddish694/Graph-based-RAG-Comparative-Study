from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Iterable

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class HashingEmbeddingModel:
    def __init__(self, dimensions: int = 384):
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        vectors = [self._encode_one(text) for text in texts]
        if not vectors:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return np.vstack(vectors).astype(np.float32)

    def _encode_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token in TOKEN_RE.findall(text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(float(np.dot(vector, vector)))
        if norm:
            vector /= norm
        return vector


class SentenceTransformerEmbeddingModel:
    def __init__(self, model_name: str, device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install project requirements "
                "or set embedding.type to 'hashing' for offline smoke tests."
            ) from exc

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        sample = self.encode(["dimension probe"])
        self.dimensions = int(sample.shape[1])

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)


class VectorIndex:
    def __init__(self, embedding_model: Any | None = None):
        self.embedding_model = embedding_model or HashingEmbeddingModel()
        self._chunks: list[dict[str, Any]] = []
        self._vectors = np.zeros((0, self.embedding_model.dimensions), dtype=np.float32)

    def add(self, chunks: Iterable[dict[str, Any]]) -> None:
        new_chunks = [dict(chunk) for chunk in chunks if str(chunk.get("text", "")).strip()]
        if not new_chunks:
            return
        new_vectors = self.embedding_model.encode([chunk["text"] for chunk in new_chunks])
        self._chunks.extend(new_chunks)
        if self._vectors.size == 0:
            self._vectors = new_vectors
        else:
            self._vectors = np.vstack([self._vectors, new_vectors])

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        if k <= 0 or not self._chunks:
            return []
        query_vector = self.embedding_model.encode([query])[0]
        scores = self._vectors @ query_vector
        ranked_indices = np.argsort(-scores)[:k]
        results: list[dict[str, Any]] = []
        for index in ranked_indices:
            chunk = dict(self._chunks[int(index)])
            chunk["score"] = float(scores[int(index)])
            results.append(chunk)
        return results

    def __len__(self) -> int:
        return len(self._chunks)
