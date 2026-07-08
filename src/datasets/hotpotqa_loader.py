from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

from src.datasets.schema import QASample


DEFAULT_HOTPOTQA_DATASET = "hotpotqa/hotpot_qa"
DEFAULT_HOTPOTQA_CONFIG = "distractor"


def normalize_hotpot_record(record: dict[str, Any]) -> QASample:
    sample_id = str(record.get("id") or record.get("_id") or record.get("sample_id"))
    contexts = _normalize_contexts(record.get("context", []))
    supporting_facts = _normalize_supporting_facts(record.get("supporting_facts", []))
    return QASample(
        sample_id=sample_id,
        question=str(record.get("question", "")).strip(),
        answer=str(record.get("answer", "")).strip(),
        contexts=contexts,
        supporting_facts=supporting_facts,
    )


def _normalize_contexts(raw_context: Any) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    if isinstance(raw_context, dict):
        titles = raw_context.get("title", [])
        sentence_groups = raw_context.get("sentences", [])
        raw_context = list(zip(titles, sentence_groups))

    for index, item in enumerate(raw_context or []):
        if isinstance(item, dict):
            title = str(item.get("title") or item.get("doc_id") or f"doc-{index}")
            sentences = item.get("sentences") or item.get("text") or ""
        else:
            title = str(item[0])
            sentences = item[1] if len(item) > 1 else ""

        if isinstance(sentences, list):
            text = " ".join(str(sentence).strip() for sentence in sentences if str(sentence).strip())
        else:
            text = str(sentences).strip()

        if text:
            contexts.append({"doc_id": title, "title": title, "text": text})
    return contexts


def _normalize_supporting_facts(raw_facts: Any) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    if isinstance(raw_facts, dict):
        titles = raw_facts.get("title", [])
        sent_ids = raw_facts.get("sent_id", [])
        raw_facts = list(zip(titles, sent_ids))

    for item in raw_facts or []:
        if isinstance(item, dict):
            doc_id = str(item.get("doc_id") or item.get("title"))
            sent_id = item.get("sent_id")
        else:
            doc_id = str(item[0])
            sent_id = item[1] if len(item) > 1 else None
        facts.append({"doc_id": doc_id, "sent_id": sent_id})
    return facts


def save_jsonl(samples: Iterable[QASample], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")


def load_cached_jsonl(path: str | Path) -> list[QASample]:
    input_path = Path(path)
    samples: list[QASample] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                samples.append(QASample.from_dict(json.loads(line)))
    return samples


def load_hotpotqa_small(
    cache_path: str | Path,
    sample_size: int = 100,
    split: str = "validation",
    seed: int = 42,
    force_download: bool = False,
    dataset_name: str = DEFAULT_HOTPOTQA_DATASET,
    dataset_config: str = DEFAULT_HOTPOTQA_CONFIG,
) -> list[QASample]:
    cache = Path(cache_path)
    if cache.exists() and not force_download:
        return load_cached_jsonl(cache)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face datasets is not installed and no cache file exists. "
            "Install datasets or provide data/processed/hotpotqa_small.jsonl."
        ) from exc

    dataset = load_dataset(dataset_name, dataset_config, split=split)
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    selected = [normalize_hotpot_record(dataset[index]) for index in indices[:sample_size]]
    save_jsonl(selected, cache)
    return selected
