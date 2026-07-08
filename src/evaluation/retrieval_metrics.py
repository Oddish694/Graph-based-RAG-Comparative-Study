from __future__ import annotations

from typing import Any


def recall_at_k(retrieved: list[dict[str, Any]], gold: list[dict[str, Any]], k: int) -> float:
    gold_labels = _gold_labels(gold)
    if not gold_labels:
        return 0.0
    retrieved_labels = _retrieved_labels(retrieved[:k], gold)
    return len(gold_labels & retrieved_labels) / len(gold_labels)


def mrr_at_k(retrieved: list[dict[str, Any]], gold: list[dict[str, Any]], k: int) -> float:
    gold_labels = _gold_labels(gold)
    if not gold_labels:
        return 0.0
    for rank, item in enumerate(retrieved[:k], start=1):
        if _item_label(item, gold) in gold_labels:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(retrieved: list[dict[str, Any]], gold: list[dict[str, Any]], k: int) -> float:
    return 1.0 if mrr_at_k(retrieved, gold, k) > 0.0 else 0.0


def evaluate_query(
    retrieved: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    k_values: list[int],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(retrieved, gold, k)
        metrics[f"mrr@{k}"] = mrr_at_k(retrieved, gold, k)
        metrics[f"hit_rate@{k}"] = hit_rate_at_k(retrieved, gold, k)
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
