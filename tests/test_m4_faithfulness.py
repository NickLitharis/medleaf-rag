"""Fixed-context numerical quote regressions, not a general fact checker.

Opt in to six real-provider cases with MEDLEAF_LIVE_FAITHFULNESS=1. Normal runs
use only the attributed fixture and deliberately incorrect answers, with no network.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from medleaf.config import Settings
from medleaf.generation.answer import (
    CitationFilter,
    LLMError,
    NumericalGroundingError,
    build_messages,
    parse_quote_selection,
    quote_selection_format,
    render_numerical_quotes,
    stream_chat,
)
from medleaf.retrieval.service import SearchHit

DATA = json.loads(
    (Path(__file__).parents[1] / "eval" / "answer-faithfulness.json").read_text(encoding="utf-8")
)
CASES = DATA["cases"]
# This gate covers the percentages and mg doses in these fixtures, not all medical units.
CLINICAL_NUMBER = re.compile(r"\d[\d.,]*\s*(?:%|mg\b)")


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _assert_grounded(answer: str, case: dict[str, Any], *, requires_numbers: bool) -> None:
    """Every percentage/mg claim must occur inside a complete copied source sentence.

    Merely finding all expected numbers would miss the observed association swap.
    A correct quotation also cannot excuse a contradictory numerical paraphrase.
    This assertion measures the narrow quote contract, not general answer accuracy.
    """
    normalized = _normalize(answer)
    assert "[Ozempic SmPC 4.8]" in normalized, f"Missing source citation: {answer}"
    remaining = normalized
    for sentence in case["source_sentences"]:
        remaining = remaining.replace(_normalize(sentence), "")
    assert not CLINICAL_NUMBER.search(remaining), f"Unquoted clinical figure: {answer}"
    if requires_numbers:
        assert _normalize(case["expected_quote"]) in normalized, f"Missing source quote: {answer}"


def _source(case: dict[str, Any]) -> SearchHit:
    return SearchHit(
        rank=1, score=1.0, point_id=case["id"], chunk_id=case["chunk_id"],
        medicine="ozempic", lang=case["lang"], doc="smpc", section="4.8",
        title="Description of selected adverse reactions",
        text="\n".join(case["source_sentences"]), source_url=case["source_url"],
        revision_date=None,
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_numerical_rules_preserve_source_context(case: dict[str, Any]) -> None:
    messages = build_messages(case["questions"][0]["text"], [_source(case)], case["lang"])
    assert "quote the complete source sentence verbatim" in messages[0]["content"]
    assert "in its original language" in messages[0]["content"]
    assert _source(case).text in messages[1]["content"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_numeric_gate_accepts_exact_quotes_and_typographic_spaces(case: dict[str, Any]) -> None:
    answer = f'> {case["expected_quote"]} [Ozempic SmPC 4.8]'
    _assert_grounded(answer.replace(" ", "\u202f"), case, requires_numbers=True)


def test_numeric_gate_rejects_observed_swap_even_with_a_correct_quote() -> None:
    case = CASES[1]
    swapped = case["swapped_answer"]
    for answer in (swapped, case["expected_quote"] + swapped, f'[{swapped}]'):
        with pytest.raises(AssertionError, match="Unquoted clinical figure"):
            _assert_grounded(answer, case, requires_numbers=True)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_numeric_gate_does_not_reward_missing_answers(case: dict[str, Any]) -> None:
    with pytest.raises(AssertionError, match="Missing source citation"):
        _assert_grounded("", case, requires_numbers=True)
    qualitative = "Nausea is reported. [Ozempic SmPC 4.8]"
    _assert_grounded(qualitative, case, requires_numbers=False)
    with pytest.raises(AssertionError, match="Missing source quote"):
        _assert_grounded(qualitative, case, requires_numbers=True)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_source_quote_references_preserve_the_full_numeric_sentence(case: dict[str, Any]) -> None:
    source = _source(case)
    messages = build_messages("nausea", [source], case["lang"])
    assert f'[quote:2] {case["expected_quote"]}' in messages[1]["content"]
    rendered = render_numerical_quotes("Reported nausea: [quote:2]", [source])
    _assert_grounded(rendered, case, requires_numbers=True)
    assert case["expected_quote"] in rendered


@pytest.mark.parametrize(
    "answer",
    ["17% at 1 mg. [Ozempic SmPC 4.8]", "[quote:2] and 17% at 1 mg.",
     "[quote:999]", "A numbered answer: 1. Nausea [Ozempic SmPC 4.8]"],
)
def test_render_rejects_raw_numbers_and_unknown_references(answer: str) -> None:
    with pytest.raises(NumericalGroundingError):
        render_numerical_quotes(answer, [_source(CASES[1])])


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_native_selection_preserves_expected_quote(case: dict[str, Any]) -> None:
    source = _source(case)
    schema = quote_selection_format([source])["json_schema"]["schema"]
    assert schema["properties"]["quote_ids"]["items"]["enum"] == ["quote:1", "quote:2", "quote:3"]
    selected = parse_quote_selection('{"quote_ids":["quote:2","quote:2"]}', [source], case["lang"])
    assert selected.count("[quote:2]") == 1
    _assert_grounded(render_numerical_quotes(selected, [source]), case, requires_numbers=True)


@pytest.mark.parametrize("raw", [
    "17% at 1 mg", '{"quote_ids":["quote:999"]}', '{"quote_ids":[17]}',
    '{"quote_ids":"quote:2"}', '{"quote_ids":null}',
    '{"quote_ids":["quote:2"],"answer":"17% at 1 mg"}',
    '{"quote_ids":["quote:2","quote:2","quote:2","quote:2","quote:2","quote:2"]}',
    '[]', 'null',
])
def test_native_selection_rejects_untrusted_output(raw: str) -> None:
    with pytest.raises((LLMError, NumericalGroundingError)):
        parse_quote_selection(raw, [_source(CASES[0])], "en")


def test_native_selection_supports_qualitative_sources_and_abstention() -> None:
    source = _source(CASES[0])
    source.text = "Ask your pharmacist.\n\n"
    selected = parse_quote_selection('{"quote_ids":["quote:1"]}', [source], "en")
    rendered = render_numerical_quotes(selected, [source])
    assert '“Ask your pharmacist.” [Ozempic SmPC 4.8]' in rendered
    for lang in ("en", "el"):
        abstention = parse_quote_selection('{"quote_ids":[]}', [source], lang)
        assert "[quote:" not in abstention
        assert abstention


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("MEDLEAF_LIVE_FAITHFULNESS") != "1",
    reason="set MEDLEAF_LIVE_FAITHFULNESS=1 and configure an LLM to run live quote regressions",
)
@pytest.mark.parametrize(
    ("case", "question"),
    [(case, question) for case in CASES for question in case["questions"]],
    ids=[f'{case["id"]}-{i}' for case in CASES for i in range(len(case["questions"]))],
)
def test_live_numerical_quotes(case: dict[str, Any], question: dict[str, Any]) -> None:
    async def answer() -> str:
        settings = Settings()
        sources = [_source(case)]
        messages = build_messages(
            question["text"], sources, case["lang"],
            source_quotes_only=settings.llm_source_quotes_only,
        )
        citations = CitationFilter({"Ozempic SmPC 4.8"})
        # Use the same opt-in native schema and parser as the API.
        response_format = (
            quote_selection_format(sources) if settings.llm_source_quotes_only else None
        )
        parts = [piece async for piece in stream_chat(
            settings, messages, response_format=response_format,
        )]
        raw = "".join(parts)
        print(f'\nRaw {case["id"]}: {question["text"]}\n{raw}\n')
        if settings.llm_source_quotes_only:
            raw = parse_quote_selection(raw, sources, case["lang"])
        rendered = render_numerical_quotes(raw, sources)
        return citations.feed(rendered) + citations.flush()

    response = asyncio.run(asyncio.wait_for(answer(), timeout=60))
    print(f'\n{case["id"]}: {question["text"]}\n{response}\n')
    _assert_grounded(response, case, requires_numbers=question["requires_numbers"])


@pytest.mark.network
@pytest.mark.skipif(
    os.getenv("MEDLEAF_LIVE_FAITHFULNESS") != "1", reason="live provider test is opt-in",
)
@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_live_source_selection_abstains_without_nausea_evidence(case: dict[str, Any]) -> None:
    settings = Settings()
    if not settings.llm_source_quotes_only:
        pytest.skip("native source selection is not enabled")
    # Keep the real, attributed distractors but remove the only nausea-rate sentence.
    source = _source(case).model_copy(update={
        "text": "\n".join([case["source_sentences"][0], case["source_sentences"][2]]),
    })

    async def run() -> None:
        messages = build_messages(
            case["questions"][1]["text"], [source], case["lang"], source_quotes_only=True,
        )
        parts = [piece async for piece in stream_chat(
            settings, messages, response_format=quote_selection_format([source]),
        )]
        raw = "".join(parts)
        print(f'\nNo nausea evidence {case["id"]}: {raw}\n')
        answer = parse_quote_selection(raw, [source], case["lang"])
        assert json.loads(raw)["quote_ids"] == [], "A different endpoint is not nausea evidence"
        assert "[quote:" not in answer

    asyncio.run(asyncio.wait_for(run(), timeout=60))
