"""M3: retrieval metrics, the PR gate and the eval dataset's integrity (no Qdrant, no model)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from medleaf.eval.metrics import gold_by_medicine, hit_rank, score
from medleaf.eval.runner import evaluate, gate, load_questions

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "eval" / "datasets" / "retrieval-v1.jsonl"
SPLITS = ROOT / "eval" / "splits.json"
PARSED = ROOT / "data" / "parsed"
GOLD_ID = re.compile(r"^[a-z0-9-]+:(smpc|leaflet):[0-9.]+$")


def test_hit_rank_needs_every_named_medicine() -> None:
    gold = gold_by_medicine(["a:smpc:4.3", "a:leaflet:2", "b:smpc:6.4"])
    assert hit_rank(["a:smpc:4.3", "x:smpc:1", "b:smpc:6.4"], gold) == 3
    assert hit_rank(["a:smpc:4.3", "a:leaflet:2"], gold) is None


def test_hit_rank_matches_parts_of_a_gold_section() -> None:
    gold = gold_by_medicine(["a:smpc:4.8"])
    assert hit_rank(["a:smpc:4.5#1", "a:smpc:4.8#3"], gold) == 2
    # 4.8 must not match 4.81 or 4.8.1-style neighbours by prefix.
    assert hit_rank(["a:smpc:4.81"], gold) is None


def test_score_counts_and_mrr() -> None:
    result = score([1, 4, 7, None])
    assert result["recall_at_5"] == {"hits": 2, "rate": 0.5}
    assert result["recall_at_10"] == {"hits": 3, "rate": 0.75}
    assert result["mrr_at_10"] == round((1 + 1 / 4 + 1 / 7) / 4, 4)


def test_evaluate_skips_questions_without_gold(tmp_path: Path) -> None:
    rows = [
        {"id": "q1", "split": "dev", "lang": "en", "question": "storage of a",
         "category": "single_section", "medicines": ["a"], "gold_chunk_ids": ["a:smpc:6.4"]},
        {"id": "q2", "split": "dev", "lang": "el", "question": "paracetamol",
         "category": "not_in_corpus", "medicines": [], "gold_chunk_ids": [], "must_refuse": True},
    ]
    path = tmp_path / "d.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    result = evaluate(load_questions(path, "dev"), lambda query: ["a:smpc:6.4"])
    assert result["unscored_no_gold"] == 1
    assert result["overall"]["recall_at_5"]["hits"] == 1
    assert set(result["by_lang"]) == {"en"}


def _result(hits: int, total: int = 50) -> dict[str, object]:
    return {"overall": {"total": total, "recall_at_5": {"hits": hits}}}


def test_gate_allows_a_one_question_drop_and_fails_on_two() -> None:
    assert gate(_result(40), _result(40)) == []
    assert gate(_result(39), _result(40)) == []
    assert gate(_result(38), _result(40))
    assert gate(_result(45, total=51), _result(40))  # dataset changed: baseline must move


def test_dataset_is_consistent_with_the_splits() -> None:
    raw = json.loads(SPLITS.read_text(encoding="utf-8"))
    splits = {k: v for k, v in raw.items() if k[0] != "_"}
    questions = load_questions(DATASET)
    assert len({q.id for q in questions}) == len(questions)
    for question in questions:
        assert all(GOLD_ID.match(g) for g in question.gold_chunk_ids), question.id
        assert sorted(gold_by_medicine(question.gold_chunk_ids)) == question.medicines, question.id
        assert {splits[m] for m in question.medicines} <= {question.split}, question.id
        assert question.gold_chunk_ids or question.must_refuse, question.id
    for split in ("dev", "test"):
        langs = [q.lang for q in questions if q.split == split]
        assert abs(langs.count("en") - langs.count("el")) <= len(langs) * 0.1, split


@pytest.mark.skipif(not PARSED.is_dir(), reason="parsed corpus not present (data/ is gitignored)")
def test_every_gold_section_exists_in_the_parsed_corpus() -> None:
    sections: set[str] = set()
    for path in PARSED.glob("*_en.jsonl"):
        for section in json.loads(path.read_text(encoding="utf-8"))["sections"]:
            sections.add(section["chunk_id"].split("#", 1)[0])
    missing = {g for q in load_questions(DATASET) for g in q.gold_chunk_ids} - sections
    assert not missing
