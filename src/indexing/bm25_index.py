from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._chunks: list[dict[str, Any]] = []
        self._term_frequencies: list[Counter[str]] = []
        self._document_frequencies: Counter[str] = Counter()
        self._doc_lengths: list[int] = []
        self._avg_doc_length = 0.0

    def add(self, chunks: Iterable[dict[str, Any]]) -> None:
        for chunk in chunks:
            text = str(chunk.get("text", "")).strip()
            if not text:
                continue
            tokens = tokenize(text)
            if not tokens:
                continue
            term_counts = Counter(tokens)
            self._chunks.append(dict(chunk))
            self._term_frequencies.append(term_counts)
            self._doc_lengths.append(len(tokens))
            self._document_frequencies.update(term_counts.keys())

        if self._doc_lengths:
            self._avg_doc_length = sum(self._doc_lengths) / len(self._doc_lengths)

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        query_terms = tokenize(query)
        if k <= 0 or not query_terms or not self._chunks:
            return []

        scored: list[tuple[float, int]] = []
        for index, term_counts in enumerate(self._term_frequencies):
            score = self._score(query_terms, term_counts, self._doc_lengths[index])
            if score > 0.0:
                scored.append((score, index))

        scored.sort(key=lambda item: item[0], reverse=True)
        results: list[dict[str, Any]] = []
        for score, index in scored[:k]:
            chunk = dict(self._chunks[index])
            chunk["score"] = float(score)
            results.append(chunk)
        return results

    def _score(self, query_terms: list[str], term_counts: Counter[str], doc_length: int) -> float:
        score = 0.0
        corpus_size = len(self._chunks)
        for term in query_terms:
            term_frequency = term_counts.get(term, 0)
            if term_frequency == 0:
                continue
            document_frequency = self._document_frequencies.get(term, 0)
            idf = math.log(1 + (corpus_size - document_frequency + 0.5) / (document_frequency + 0.5))
            denominator = term_frequency + self.k1 * (
                1 - self.b + self.b * doc_length / (self._avg_doc_length or 1.0)
            )
            score += idf * (term_frequency * (self.k1 + 1)) / denominator
        return score

    def __len__(self) -> int:
        return len(self._chunks)
