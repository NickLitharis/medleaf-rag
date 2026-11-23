"""The retrieval question set: ``eval/datasets/*.jsonl`` (one question per line)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EvalQuestion(BaseModel):
    """One line of ``eval/datasets/*.jsonl``."""

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
