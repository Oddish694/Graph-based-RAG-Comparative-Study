from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol

from src.indexing.vector_index import HashingEmbeddingModel
from src.retrieval.hybrid_rag import HybridRAGRetriever


class LightRAGExternalRunner(Protocol):
    def retrieve(self, query: str, top_k: int) -> Any:
        ...


class LightRAGUnavailableError(RuntimeError):
    pass


@dataclass
class LightRAGResultAdapter:
    chunk_lookup: dict[str, dict[str, Any]]

    def normalize(self, raw_results: Any, top_k: int) -> list[dict[str, Any]]:
        if raw_results is None:
            return []
        if isinstance(raw_results, dict):
            candidates = raw_results.get("results") or raw_results.get("contexts") or raw_results.get("chunks") or []
        else:
            candidates = raw_results

        normalized: list[dict[str, Any]] = []
        for rank, candidate in enumerate(candidates, start=1):
            row = self._normalize_one(candidate, rank)
            if row is not None:
                normalized.append(row)
            if len(normalized) >= top_k:
                break
        return normalized

    def _normalize_one(self, candidate: Any, rank: int) -> dict[str, Any] | None:
        if isinstance(candidate, str):
            return {
                "chunk_id": f"external_rank_{rank}",
                "doc_id": "",
                "text": candidate,
                "score": 1.0 / rank,
                "retriever_source": "lightrag_external",
                "lightrag_rank": rank,
            }
        if not isinstance(candidate, dict):
            return None

        chunk_id = _first_present(candidate, ["chunk_id", "id", "source_id", "doc_id"])
        chunk_id = str(chunk_id) if chunk_id is not None else f"external_rank_{rank}"
        source_chunk = self.chunk_lookup.get(chunk_id, {})
        doc_id = _first_present(candidate, ["doc_id", "document_id", "source_doc_id", "title"])
        text = _first_present(candidate, ["text", "content", "chunk", "context"])
        score = _first_present(candidate, ["score", "similarity", "distance", "rank_score"])

        row = dict(source_chunk)
        row.update(
            {
                "chunk_id": chunk_id,
                "doc_id": str(doc_id if doc_id is not None else source_chunk.get("doc_id", "")),
                "text": str(text if text is not None else source_chunk.get("text", "")),
                "score": float(score) if score is not None else 1.0 / rank,
                "retriever_source": "lightrag_external",
                "lightrag_rank": rank,
            }
        )
        return row


class LightRAGControlledRetriever:
    def __init__(
        self,
        adapter: LightRAGResultAdapter,
        backend: str = "local_compat",
        local_retriever: HybridRAGRetriever | None = None,
        external_runner: LightRAGExternalRunner | None = None,
    ):
        self.adapter = adapter
        self.backend = backend
        self.local_retriever = local_retriever
        self.external_runner = external_runner

    @classmethod
    def from_chunks(
        cls,
        chunks: Iterable[dict[str, Any]],
        embedding_model: Any | None = None,
        backend: str = "local_compat",
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        fusion: str = "weighted",
        rrf_k: int = 60,
        external_runner: LightRAGExternalRunner | None = None,
    ) -> "LightRAGControlledRetriever":
        chunk_list = [dict(chunk) for chunk in chunks]
        chunk_lookup = {str(chunk["chunk_id"]): chunk for chunk in chunk_list if chunk.get("chunk_id")}
        adapter = LightRAGResultAdapter(chunk_lookup=chunk_lookup)
        backend = backend.lower()

        if backend == "local_compat":
            local_retriever = HybridRAGRetriever.from_chunks(
                chunk_list,
                embedding_model=embedding_model or HashingEmbeddingModel(),
                bm25_weight=bm25_weight,
                dense_weight=dense_weight,
                fusion=fusion,
                rrf_k=rrf_k,
            )
            return cls(adapter=adapter, backend=backend, local_retriever=local_retriever)

        if backend == "external":
            _ensure_lightrag_available()
            if external_runner is None:
                raise LightRAGUnavailableError(
                    "LightRAG is installed, but no external_runner was provided. "
                    "Provide a runner that returns retrievable contexts with chunk_id/doc_id metadata."
                )
            return cls(adapter=adapter, backend=backend, external_runner=external_runner)

        raise ValueError(f"Unsupported LightRAG backend: {backend}")

    def retrieve(self, query: str, top_k: int = 5, candidate_k: int | None = None) -> list[dict[str, Any]]:
        if self.backend == "local_compat":
            candidate_limit = candidate_k or max(top_k * 4, top_k)
            results = self.local_retriever.retrieve(query, top_k=top_k, candidate_k=candidate_limit)
            for rank, row in enumerate(results, start=1):
                row["retriever_source"] = "lightrag_local_compat"
                row["lightrag_backend"] = self.backend
                row["lightrag_rank"] = rank
            return results

        if self.backend == "external":
            raw_results = self.external_runner.retrieve(query, top_k=top_k)
            results = self.adapter.normalize(raw_results, top_k=top_k)
            for row in results:
                row["lightrag_backend"] = self.backend
            return results

        raise ValueError(f"Unsupported LightRAG backend: {self.backend}")


def _ensure_lightrag_available() -> None:
    try:
        import lightrag  # noqa: F401
    except ImportError as exc:
        raise LightRAGUnavailableError(
            "The optional LightRAG package is not installed. Install lightrag-hku and provide "
            "a metadata-preserving runner before using backend='external'."
        ) from exc


def _first_present(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None
