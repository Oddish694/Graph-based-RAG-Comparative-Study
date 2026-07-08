from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(eq=True)
class QASample:
    sample_id: str
    question: str
    answer: str
    contexts: list[dict[str, Any]]
    supporting_facts: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QASample":
        return cls(
            sample_id=str(data["sample_id"]),
            question=str(data["question"]),
            answer=str(data["answer"]),
            contexts=list(data.get("contexts", [])),
            supporting_facts=list(data.get("supporting_facts", [])),
        )
