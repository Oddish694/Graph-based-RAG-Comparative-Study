from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_FIELDS = [
    "variant",
    "retriever",
    "recall@5",
    "precision@5",
    "ndcg@5",
    "mrr@5",
    "full_evidence_recall@5",
    "recall@10",
    "ndcg@10",
    "full_evidence_recall@10",
    "avg_retrieval_latency_seconds",
    "index_time_seconds",
    "graph_num_entities",
    "graph_num_edges",
]


def summarize_aggregate_metrics(result_dirs: list[Path], output_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result_dir in result_dirs:
        aggregate_path = result_dir / "aggregate_metrics.json"
        if not aggregate_path.exists():
            continue
        metrics = json.loads(aggregate_path.read_text(encoding="utf-8"))
        row = {"variant": result_dir.name}
        for field in DEFAULT_FIELDS[1:]:
            row[field] = metrics.get(field, "")
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEFAULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize aggregate retrieval metrics into a CSV table.")
    parser.add_argument("--output", default="results/ablation_table.csv", help="Output CSV path.")
    parser.add_argument("result_dirs", nargs="+", help="Result directories containing aggregate_metrics.json.")
    args = parser.parse_args()
    rows = summarize_aggregate_metrics([Path(path) for path in args.result_dirs], Path(args.output))
    print(json.dumps({"output": args.output, "num_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
