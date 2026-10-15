"""Run the retrieval eval over one split and gate it against a committed baseline.

``medleaf eval --split dev`` sends each question, unfiltered, through the same
:class:`~medleaf.retrieval.service.RetrievalService` as ``/v1/search`` and writes
``eval/results/<stamp>-<split>.json``. Questions with no gold chunks (not in corpus)
are kept in the file but not scored: they test refusal, which is a later milestone.
A search failure aborts the run; it is never scored as "nothing found".
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from medleaf.eval.metrics import gold_by_medicine, hit_rank, score

#: SPEC §13: the PR gate fails when Recall@5 drops by more than this many questions.
MAX_RECALL5_DROP = 1
TOP_K = 10


class EvalQuestion(BaseModel):
    """One line of ``eval/datasets/*.jsonl`` (SPEC §11 fields plus split and medicines)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    split: Literal["dev", "test"]
    lang: Literal["en", "el"]
    question: str = Field(min_length=1)
    category: str
    medicines: list[str]
    gold_chunk_ids: list[str]
    expected_label: Literal["info", "personal", "emergency"] = "info"
    must_refuse: bool = False
    notes: str = ""


def load_questions(path: Path, split: str | None = None) -> list[EvalQuestion]:
    """Parse a JSONL dataset, optionally keeping one split."""
    questions = [
        EvalQuestion.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [q for q in questions if split is None or q.split == split]


def evaluate(
    questions: Sequence[EvalQuestion], search: Callable[[str], list[str]]
) -> dict[str, Any]:
    """Score ``questions`` with ``search`` (query -> ranked chunk IDs)."""
    rows = []
    for question in questions:
        if not question.gold_chunk_ids:
            continue
        retrieved = search(question.question)
        rank = hit_rank(retrieved, gold_by_medicine(question.gold_chunk_ids))
        rows.append((question, rank, retrieved))

    def grouped(key: Callable[[EvalQuestion], str]) -> dict[str, dict[str, object]]:
        groups: dict[str, list[int | None]] = {}
        for question, rank, _ in rows:
            groups.setdefault(key(question), []).append(rank)
        return {name: score(ranks) for name, ranks in sorted(groups.items())}

    return {
        "questions": len(questions),
        "unscored_no_gold": len(questions) - len(rows),
        "overall": score([rank for _, rank, _ in rows]),
        "by_lang": grouped(lambda q: q.lang),
        "by_category": grouped(lambda q: q.category),
        "per_question": [
            {"id": q.id, "hit_rank": rank, "retrieved": retrieved} for q, rank, retrieved in rows
        ],
    }


def gate(result: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """Failure messages for the PR gate; empty means the result passes."""
    now, base = result["overall"], baseline["overall"]
    if now["total"] != base["total"]:
        return [
            f"scored question count changed ({base['total']} -> {now['total']}); "
            "update eval/baseline.json in its own PR"
        ]
    drop = base["recall_at_5"]["hits"] - now["recall_at_5"]["hits"]
    if drop > MAX_RECALL5_DROP:
        return [
            f"Recall@5 dropped by {drop} questions "
            f"({base['recall_at_5']['hits']} -> {now['recall_at_5']['hits']} of {now['total']}); "
            f"the gate allows {MAX_RECALL5_DROP}"
        ]
    return []


def write_result(result: dict[str, Any], out_dir: Path, split: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{split}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
