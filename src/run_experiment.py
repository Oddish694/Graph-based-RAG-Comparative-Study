from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from src.chunking.fixed_chunker import chunk_sample_contexts
from src.datasets.hotpotqa_loader import load_cached_jsonl, load_hotpotqa_small
from src.evaluation.retrieval_metrics import evaluate_query, mean_metrics
from src.indexing.vector_index import HashingEmbeddingModel, SentenceTransformerEmbeddingModel
from src.retrieval.bm25_rag import BM25RAGRetriever
from src.retrieval.graph_rag_style import GraphRAGStyleRetriever
from src.retrieval.hybrid_rag import HybridRAGRetriever
from src.retrieval.improved_lightrag import ImprovedLightRAGRetriever
from src.retrieval.vector_rag import VectorRAGRetriever


def run_phase1_vector_rag(config: dict[str, Any]) -> dict[str, Any]:
    return run_retrieval_experiment(config, default_method="vector", default_output_dir="results/phase1_vector_rag")


def run_phase2_hybrid_rag(config: dict[str, Any]) -> dict[str, Any]:
    return run_retrieval_experiment(config, default_method="hybrid", default_output_dir="results/phase2_hybrid_rag")


def run_retrieval_experiment(
    config: dict[str, Any],
    default_method: str = "vector",
    default_output_dir: str = "results/retrieval_experiment",
) -> dict[str, Any]:
    dataset_path = Path(config["dataset_path"])
    output_dir = Path(config.get("output_dir", default_output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = _load_samples(config, dataset_path)

    chunk_config = config.get("chunking", {})
    chunk_size = int(chunk_config.get("chunk_size", 256))
    overlap = int(chunk_config.get("overlap", 50))
    chunks: list[dict[str, Any]] = []
    for sample in samples:
        chunks.extend(chunk_sample_contexts(sample, chunk_size=chunk_size, overlap=overlap))

    retrieval_config = config.get("retrieval", {})
    method = str(retrieval_config.get("method", default_method)).lower()
    embedding_model = None
    if method in {"vector", "dense", "hybrid", "graph_rag_style", "graph", "improved_lightrag"}:
        embedding_model = build_embedding_model(config.get("embedding", {}))

    index_start = time.perf_counter()
    retriever = build_retriever(method, chunks, retrieval_config, embedding_model)
    index_time_seconds = time.perf_counter() - index_start

    top_k = int(retrieval_config.get("top_k", 5))
    candidate_k_value = retrieval_config.get("candidate_k")
    candidate_k = int(candidate_k_value) if candidate_k_value is not None else None
    k_values = [int(value) for value in config.get("metrics", {}).get("k_values", [top_k])]

    metric_rows: list[dict[str, float]] = []
    per_query_rows: list[dict[str, Any]] = []
    for sample in samples:
        query_start = time.perf_counter()
        if method in {"hybrid", "graph_rag_style", "graph", "improved_lightrag"}:
            retrieved = retriever.retrieve(sample.question, top_k=top_k, candidate_k=candidate_k)
        else:
            retrieved = retriever.retrieve(sample.question, top_k=top_k)
        latency_seconds = time.perf_counter() - query_start
        metrics = evaluate_query(retrieved, sample.supporting_facts, k_values=k_values)
        metric_rows.append(metrics)
        per_query_rows.append(
            {
                "sample_id": sample.sample_id,
                "retriever": method,
                "question": sample.question,
                "answer": sample.answer,
                "retrieved_doc_ids": json.dumps([row.get("doc_id") for row in retrieved], ensure_ascii=False),
                "retrieved_chunk_ids": json.dumps([row.get("chunk_id") for row in retrieved], ensure_ascii=False),
                "retrieved_scores": json.dumps([row.get("score") for row in retrieved], ensure_ascii=False),
                "retrieval_latency_seconds": latency_seconds,
                **metrics,
            }
        )

    aggregate = mean_metrics(metric_rows)
    aggregate["retriever"] = method
    aggregate["index_time_seconds"] = index_time_seconds
    aggregate["num_samples"] = float(len(samples))
    aggregate["num_chunks"] = float(len(chunks))
    if hasattr(retriever, "graph_index"):
        aggregate["graph_num_entities"] = float(retriever.graph_index.num_entities)
        aggregate["graph_num_edges"] = float(retriever.graph_index.num_edges)
    aggregate["avg_retrieval_latency_seconds"] = (
        sum(row["retrieval_latency_seconds"] for row in per_query_rows) / len(per_query_rows)
        if per_query_rows
        else 0.0
    )

    _write_csv(output_dir / "per_query_results.csv", per_query_rows)
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return aggregate


def _load_samples(config: dict[str, Any], dataset_path: Path) -> list[Any]:
    if dataset_path.exists():
        return load_cached_jsonl(dataset_path)
    return load_hotpotqa_small(
        cache_path=dataset_path,
        sample_size=int(config.get("sample_size", 100)),
        split=str(config.get("split", "validation")),
        seed=int(config.get("seed", 42)),
        dataset_name=str(config.get("dataset_name", "hotpotqa/hotpot_qa")),
        dataset_config=str(config.get("dataset_config", "distractor")),
    )


def build_retriever(
    method: str,
    chunks: list[dict[str, Any]],
    retrieval_config: dict[str, Any],
    embedding_model: Any | None = None,
) -> Any:
    if method in {"vector", "dense"}:
        return VectorRAGRetriever.from_chunks(chunks, embedding_model=embedding_model)
    if method == "bm25":
        return BM25RAGRetriever.from_chunks(chunks)
    if method == "hybrid":
        return HybridRAGRetriever.from_chunks(
            chunks,
            embedding_model=embedding_model,
            bm25_weight=float(retrieval_config.get("bm25_weight", 0.5)),
            dense_weight=float(retrieval_config.get("dense_weight", 0.5)),
            fusion=str(retrieval_config.get("fusion", "weighted")).lower(),
            rrf_k=int(retrieval_config.get("rrf_k", 60)),
        )
    if method in {"graph_rag_style", "graph"}:
        return GraphRAGStyleRetriever.from_chunks(
            chunks,
            embedding_model=embedding_model,
            seed_top_k=int(retrieval_config.get("seed_top_k", 10)),
            expansion_depth=int(retrieval_config.get("expansion_depth", 1)),
            max_neighbors_per_entity=int(retrieval_config.get("max_neighbors_per_entity", 10)),
            seed_weight=float(retrieval_config.get("seed_weight", 0.7)),
            graph_weight=float(retrieval_config.get("graph_weight", 0.3)),
            bm25_weight=float(retrieval_config.get("bm25_weight", 0.5)),
            dense_weight=float(retrieval_config.get("dense_weight", 0.5)),
            fusion=str(retrieval_config.get("fusion", "weighted")).lower(),
            rrf_k=int(retrieval_config.get("rrf_k", 60)),
        )
    if method == "improved_lightrag":
        return ImprovedLightRAGRetriever.from_chunks(
            chunks,
            embedding_model=embedding_model,
            seed_top_k=int(retrieval_config.get("seed_top_k", 10)),
            expansion_depth=int(retrieval_config.get("expansion_depth", 1)),
            max_neighbors_per_entity=int(retrieval_config.get("max_neighbors_per_entity", 10)),
            seed_weight=float(retrieval_config.get("seed_weight", 0.7)),
            graph_weight=float(retrieval_config.get("graph_weight", 0.3)),
            bm25_weight=float(retrieval_config.get("bm25_weight", 0.5)),
            dense_weight=float(retrieval_config.get("dense_weight", 0.5)),
            fusion=str(retrieval_config.get("fusion", "weighted")).lower(),
            rrf_k=int(retrieval_config.get("rrf_k", 60)),
            candidate_pool_size=int(
                retrieval_config.get("candidate_pool_size", retrieval_config.get("candidate_k", 40))
            ),
            coverage_weight=float(retrieval_config.get("coverage_weight", 0.15)),
            entity_coverage_weight=float(retrieval_config.get("entity_coverage_weight", 0.10)),
            relevance_weight=float(retrieval_config.get("relevance_weight", 1.0)),
            use_graph_expansion=_as_bool(retrieval_config.get("use_graph_expansion", True)),
            use_coverage_reranking=_as_bool(retrieval_config.get("use_coverage_reranking", True)),
        )
    raise ValueError(f"Unsupported retrieval method: {method}")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def build_embedding_model(config: dict[str, Any]) -> Any:
    embedding_type = str(config.get("type", "hashing")).lower()
    if embedding_type in {"hashing", "hash"}:
        return HashingEmbeddingModel(dimensions=int(config.get("dimensions", 384)))
    if embedding_type in {"sentence_transformers", "sentence-transformer"}:
        return SentenceTransformerEmbeddingModel(
            model_name=str(config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")),
            device=config.get("device"),
        )
    raise ValueError(f"Unsupported embedding type: {embedding_type}")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to read YAML config files.") from exc
        loaded = yaml.safe_load(text)
        return loaded or {}
    return json.loads(text)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG retrieval evaluation.")
    parser.add_argument("--config", required=True, help="Path to the experiment config file.")
    args = parser.parse_args()
    config = load_config(args.config)
    method = str(config.get("retrieval", {}).get("method", "vector")).lower()
    if method == "hybrid":
        summary = run_phase2_hybrid_rag(config)
    else:
        summary = run_retrieval_experiment(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()





