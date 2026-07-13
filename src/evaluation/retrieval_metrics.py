from __future__ import annotations

import math
from typing import Any


def recall_at_k(retrieved: list[dict[str, Any]], gold: list[dict[str, Any]], k: int) -> float:
    gold_labels = _gold_labels(gold)
    if not gold_labels:
        return 0.0
    retrieved_labels = _retrieved_labels(retrieved[:k], gold)
    return len(gold_labels & retrieved_labels) / len(gold_labels)


def precision_at_k(retrieved: list[dict[str, Any]], gold: list[dict[str, Any]], k: int) -> float:
    if k <= 0:
        return 0.0
    gold_labels = _gold_labels(gold)
    if not gold_labels:
        return 0.0
    retrieved_labels = _retrieved_labels(retrieved[:k], gold)
    return len(gold_labels & retrieved_labels) / k


def mrr_at_k(retrieved: list[dict[str, Any]], gold: list[dict[str, Any]], k: int) -> float:
    gold_labels = _gold_labels(gold)
    if not gold_labels:
        return 0.0
    for rank, item in enumerate(retrieved[:k], start=1):
        if _item_label(item, gold) in gold_labels:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[dict[str, Any]], gold: list[dict[str, Any]], k: int) -> float:
    gold_labels = _gold_labels(gold)
    if k <= 0 or not gold_labels:
        return 0.0

    seen_labels: set[str] = set()
    dcg = 0.0
    for rank, item in enumerate(retrieved[:k], start=1):
        label = _item_label(item, gold)
        if label in gold_labels and label not in seen_labels:
            dcg += 1.0 / math.log2(rank + 1)
            seen_labels.add(label)

    ideal_hits = min(len(gold_labels), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def hit_rate_at_k(retrieved: list[dict[str, Any]], gold: list[dict[str, Any]], k: int) -> float:
    return 1.0 if mrr_at_k(retrieved, gold, k) > 0.0 else 0.0


def evidence_hit_count_at_k(retrieved: list[dict[str, Any]], gold: list[dict[str, Any]], k: int) -> float:
    gold_labels = _gold_labels(gold)
    if not gold_labels:
        return 0.0
    retrieved_labels = _retrieved_labels(retrieved[:k], gold)
    return float(len(gold_labels & retrieved_labels))


def full_evidence_recall_at_k(retrieved: list[dict[str, Any]], gold: list[dict[str, Any]], k: int) -> float:
    gold_labels = _gold_labels(gold)
    if not gold_labels:
        return 0.0
    retrieved_labels = _retrieved_labels(retrieved[:k], gold)
    return 1.0 if gold_labels.issubset(retrieved_labels) else 0.0


def retrieved_context_tokens_at_k(retrieved: list[dict[str, Any]], k: int) -> float:
    if k <= 0:
        return 0.0
    return float(sum(len(str(item.get("text", "")).split()) for item in retrieved[:k]))


def evaluate_query(
    retrieved: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    k_values: list[int],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(retrieved, gold, k)
        metrics[f"precision@{k}"] = precision_at_k(retrieved, gold, k)
        metrics[f"mrr@{k}"] = mrr_at_k(retrieved, gold, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved, gold, k)
        metrics[f"hit_rate@{k}"] = hit_rate_at_k(retrieved, gold, k)
        metrics[f"evidence_hit_count@{k}"] = evidence_hit_count_at_k(retrieved, gold, k)
        metrics[f"full_evidence_recall@{k}"] = full_evidence_recall_at_k(retrieved, gold, k)
        metrics[f"retrieved_context_tokens@{k}"] = retrieved_context_tokens_at_k(retrieved, k)
    return metrics


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {key: sum(float(row.get(key, 0.0)) for row in rows) / len(rows) for key in keys}


def _gold_labels(gold: list[dict[str, Any]]) -> set[str]:
    labels = set()
    for item in gold:
        if item.get("chunk_id"):
            labels.add(f"chunk:{item['chunk_id']}")
        elif item.get("doc_id"):
            labels.add(f"doc:{item['doc_id']}")
    return labels


def _retrieved_labels(retrieved: list[dict[str, Any]], gold: list[dict[str, Any]]) -> set[str]:
    return {_item_label(item, gold) for item in retrieved}


def _item_label(item: dict[str, Any], gold: list[dict[str, Any]]) -> str:
    gold_uses_chunk = any(fact.get("chunk_id") for fact in gold)
    if gold_uses_chunk and item.get("chunk_id"):
        return f"chunk:{item['chunk_id']}"
    return f"doc:{item.get('doc_id')}"
