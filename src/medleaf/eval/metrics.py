"""Retrieval metrics from SPEC §11.

Gold IDs name a whole section (`<slug>:<doc>:<section>`). A question is a hit at k when,
for **every** medicine it names, at least one of that
medicine's gold chunk IDs is in the top k. Chunk IDs are shared by EN and EL, so a Greek
question can be satisfied by an English chunk. ``hit_rank`` is the rank at which the last
named medicine is first covered: Recall@k is ``hit_rank <= k`` and MRR@10 is its inverse,
so the two metrics can never disagree about a question.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence


def gold_by_medicine(gold_chunk_ids: Iterable[str]) -> dict[str, set[str]]:
    """Group gold IDs (``<slug>:<doc>:<section>``) by their medicine slug."""
    grouped: dict[str, set[str]] = {}
    for chunk_id in gold_chunk_ids:
        grouped.setdefault(chunk_id.split(":", 1)[0], set()).add(chunk_id)
    return grouped


def hit_rank(retrieved: Sequence[str], gold: Mapping[str, set[str]]) -> int | None:
    """1-based rank at which every medicine has a gold chunk, or None if never."""
    if not gold:
        return None
    missing = set(gold)
    for rank, chunk_id in enumerate(retrieved, start=1):
        # Long sections are indexed as parts (``eliquis:smpc:4.8#3``); gold names the section.
        section_id = chunk_id.split("#", 1)[0]
        medicine = section_id.split(":", 1)[0]
        if section_id in gold.get(medicine, set()):
            missing.discard(medicine)
        if not missing:
            return rank
    return None


def score(ranks: Sequence[int | None]) -> dict[str, object]:
    """Recall@5, Recall@10 and MRR@10 with the counts behind them."""
    total = len(ranks)
    at5 = sum(1 for rank in ranks if rank is not None and rank <= 5)
    at10 = sum(1 for rank in ranks if rank is not None and rank <= 10)
    mrr = sum(1 / rank for rank in ranks if rank is not None and rank <= 10)
    return {
        "total": total,
        "recall_at_5": {"hits": at5, "rate": round(at5 / total, 4) if total else 0.0},
        "recall_at_10": {"hits": at10, "rate": round(at10 / total, 4) if total else 0.0},
        "mrr_at_10": round(mrr / total, 4) if total else 0.0,
    }
