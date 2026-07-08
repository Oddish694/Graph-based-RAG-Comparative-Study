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
from src.retrieval.vector_rag import VectorRAGRetriever


def run_phase1_vector_rag(config: dict[str, Any]) -> dict[str, float]:
    dataset_path = Path(config["dataset_path"])
    output_dir = Path(config.get("output_dir", "results/phase1_vector_rag"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset_path.exists():
        samples = load_cached_jsonl(dataset_path)
    else:
        samples = load_hotpotqa_small(
            cache_path=dataset_path,
            sample_size=int(config.get("sample_size", 100)),
            split=str(config.get("split", "validation")),
            seed=int(config.get("seed", 42)),
        )

    chunk_config = config.get("chunking", {})
    chunk_size = int(chunk_config.get("chunk_size", 256))
    overlap = int(chunk_config.get("overlap", 50))
    chunks = []
    for sample in samples:
        chunks.extend(chunk_sample_contexts(sample, chunk_size=chunk_size, overlap=overlap))

    embedding_model = build_embedding_model(config.get("embedding", {}))
    index_start = time.perf_counter()
    retriever = VectorRAGRetriever.from_chunks(chunks, embedding_model=embedding_model)
    index_time_seconds = time.perf_counter() - index_start

    retrieval_config = config.get("retrieval", {})
    top_k = int(retrieval_config.get("top_k", 5))
    k_values = [int(value) for value in config.get("metrics", {}).get("k_values", [top_k])]

    metric_rows: list[dict[str, float]] = []
    per_query_rows: list[dict[str, Any]] = []
    for sample in samples:
        query_start = time.perf_counter()
        retrieved = retriever.retrieve(sample.question, top_k=top_k)
        latency_seconds = time.perf_counter() - query_start
        metrics = evaluate_query(retrieved, sample.supporting_facts, k_values=k_values)
        metric_rows.append(metrics)
        per_query_rows.append(
            {
                "sample_id": sample.sample_id,
                "question": sample.question,
                "answer": sample.answer,
                "retrieved_doc_ids": json.dumps([row.get("doc_id") for row in retrieved], ensure_ascii=False),
                "retrieved_chunk_ids": json.dumps([row.get("chunk_id") for row in retrieved], ensure_ascii=False),
                "retrieval_latency_seconds": latency_seconds,
                **metrics,
            }
        )

    aggregate = mean_metrics(metric_rows)
    aggregate["index_time_seconds"] = index_time_seconds
    aggregate["num_samples"] = float(len(samples))
    aggregate["num_chunks"] = float(len(chunks))
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
    parser = argparse.ArgumentParser(description="Run Phase 1 Vector RAG retrieval evaluation.")
    parser.add_argument("--config", required=True, help="Path to the experiment config file.")
    args = parser.parse_args()
    summary = run_phase1_vector_rag(load_config(args.config))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
