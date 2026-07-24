from __future__ import annotations

from typing import Any, Iterable

from src.indexing.vector_index import HashingEmbeddingModel
from src.retrieval.coverage_reranker import CoverageAwareReranker
from src.retrieval.graph_rag_style import GraphRAGStyleRetriever
from src.retrieval.hybrid_rag import HybridRAGRetriever


class ImprovedLightRAGRetriever:
    def __init__(
        self,
        graph_retriever: GraphRAGStyleRetriever | None = None,
        seed_retriever: HybridRAGRetriever | None = None,
        reranker: CoverageAwareReranker | None = None,
        candidate_pool_size: int = 40,
        use_graph_expansion: bool = True,
        use_coverage_reranking: bool = True,
    ):
        self.graph_retriever = graph_retriever
        self.seed_retriever = seed_retriever
        self.reranker = reranker or CoverageAwareReranker()
        self.candidate_pool_size = candidate_pool_size
        self.use_graph_expansion = use_graph_expansion
        self.use_coverage_reranking = use_coverage_reranking
        self.graph_index = graph_retriever.graph_index if graph_retriever is not None else None

    @classmethod
    def from_chunks(
        cls,
        chunks: Iterable[dict[str, Any]],
        embedding_model: Any | None = None,
        seed_top_k: int = 10,
        expansion_depth: int = 1,
        max_neighbors_per_entity: int = 10,
        max_seed_entities: int = 25,
        seed_weight: float = 0.7,
        graph_weight: float = 0.3,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        fusion: str = "weighted",
        rrf_k: int = 60,
        query_entity_weight: float = 0.45,
        seed_entity_weight: float = 0.25,
        expansion_weight: float = 0.30,
        distance_decay: float = 0.5,
        candidate_pool_size: int = 40,
        coverage_weight: float = 0.15,
        entity_coverage_weight: float = 0.10,
        relevance_weight: float = 1.0,
        use_graph_expansion: bool = True,
        use_coverage_reranking: bool = True,
    ) -> "ImprovedLightRAGRetriever":
        chunk_list = [dict(chunk) for chunk in chunks]
        model = embedding_model or HashingEmbeddingModel()
        graph_retriever = None
        seed_retriever = None
        if use_graph_expansion:
            graph_retriever = GraphRAGStyleRetriever.from_chunks(
                chunk_list,
                embedding_model=model,
                seed_top_k=seed_top_k,
                expansion_depth=expansion_depth,
                max_neighbors_per_entity=max_neighbors_per_entity,
                max_seed_entities=max_seed_entities,
                seed_weight=seed_weight,
                graph_weight=graph_weight,
                bm25_weight=bm25_weight,
                dense_weight=dense_weight,
                fusion=fusion,
                rrf_k=rrf_k,
                query_entity_weight=query_entity_weight,
                seed_entity_weight=seed_entity_weight,
                expansion_weight=expansion_weight,
                distance_decay=distance_decay,
            )
        else:
            seed_retriever = HybridRAGRetriever.from_chunks(
                chunk_list,
                embedding_model=model,
                bm25_weight=bm25_weight,
                dense_weight=dense_weight,
                fusion=fusion,
                rrf_k=rrf_k,
            )
        reranker = CoverageAwareReranker(
            coverage_weight=coverage_weight,
            entity_coverage_weight=entity_coverage_weight,
            relevance_weight=relevance_weight,
        )
        return cls(
            graph_retriever=graph_retriever,
            seed_retriever=seed_retriever,
            reranker=reranker,
            candidate_pool_size=candidate_pool_size,
            use_graph_expansion=use_graph_expansion,
            use_coverage_reranking=use_coverage_reranking,
        )

    def retrieve(self, query: str, top_k: int = 5, candidate_k: int | None = None) -> list[dict[str, Any]]:
        pool_size = candidate_k or self.candidate_pool_size or max(top_k * 4, top_k)
        if self.use_graph_expansion:
            candidates = self.graph_retriever.retrieve(query, top_k=pool_size, candidate_k=pool_size)
        else:
            candidates = self.seed_retriever.retrieve(query, top_k=pool_size, candidate_k=pool_size)
            for row in candidates:
                row.setdefault("matched_entities", [])
                row.setdefault("graph_score", 0.0)
                row.setdefault("seed_score", row.get("score", 0.0))

        if self.use_coverage_reranking:
            results = self.reranker.rerank(candidates, top_k=top_k)
        else:
            results = [dict(candidate) for candidate in candidates[:top_k]]
            for row in results:
                row.setdefault("coverage_score", 0.0)
                row.setdefault("rerank_score", row.get("score", 0.0))

        for row in results:
            row["retriever_source"] = "improved_lightrag"
        return results
