"""M4 slice 2A: safety labels before retrieval, and the medicine-name filter."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from medleaf.api.app import create_app
from medleaf.config import Settings
from medleaf.generation.answer import LLMError
from medleaf.retrieval.names import detect_medicines, medicine_names
from medleaf.retrieval.service import SearchHit, SearchResponse
from medleaf.safety.classifier import MESSAGES, SAFETY_PROMPT, classify, keyword_label

EMERGENCY = [
    "I took too many Eliquis tablets",
    "I took 20 tablets of Lyrica",
    "Πήρα 30 χάπια",
    "My son swallowed my Lyrica pills, what do I do?",
    "I can't breathe after my Ozempic injection",
    "My throat is swelling after Humira",
    "I want to kill myself with my tablets",
    "I'm suicidal",
    "Πήρα πολλά χάπια Xarelto",
    "Το παιδί μου κατάπιε δύο χάπια Eliquis",
    "Δεν μπορώ να αναπνεύσω μετά την ένεση",
    "Θέλω να αυτοκτονήσω",
    "ΘΕΛΩ ΝΑ ΠΕΘΑΝΩ",
]
PERSONAL = [
    "I'm on 5 mg and missed a dose, should I double it?",
    "I am pregnant, is Ozempic safe for me?",
    "Ξέχασα μια δόση Eliquis",
    "Είμαι έγκυος και παίρνω Lyrica",
]
INFO = [
    "What are the symptoms of an Eliquis overdose?",
    "Does Lyrica cause suicidal thoughts?",
    "Can Eliquis be taken with ibuprofen?",
    "Can the tablets be swallowed whole?",
    "How many tablets are in one pack of Eliquis?",
    "Τι κάνω σε περίπτωση υπερδοσολογίας με Xarelto;",
    "Προκαλεί το Lyrica αυτοκτονικό ιδεασμό;",
    "Πώς φυλάσσεται το Ozempic;",
]


@pytest.mark.parametrize("question", EMERGENCY)
def test_emergency_keywords(question: str) -> None:
    assert keyword_label(question) == "emergency"


@pytest.mark.parametrize("question", PERSONAL)
def test_personal_keywords(question: str) -> None:
    assert keyword_label(question) == "personal"


@pytest.mark.parametrize("question", INFO)
def test_info_questions_match_no_rule(question: str) -> None:
    assert keyword_label(question) is None


def _reply(*pieces: str, error: Exception | None = None):  # type: ignore[no-untyped-def]
    calls: list[list[dict[str, str]]] = []

    async def generate(messages: list[dict[str, str]]) -> AsyncIterator[str]:
        calls.append(messages)
        for piece in pieces:
            yield piece
        if error is not None:
            raise error
    generate.calls = calls  # type: ignore[attr-defined]
    return generate


def test_classify_combines_rules_and_llm() -> None:
    llm = _reply("Info")
    assert asyncio.run(classify("I took too much Eliquis", llm)) == ("emergency", "keywords")
    assert llm.calls == []  # emergencies skip the LLM
    assert asyncio.run(classify("What is Eliquis?", llm)) == ("info", "llm")
    assert asyncio.run(classify("Should I stop Eliquis?", llm)) == ("personal", "keywords")
    assert asyncio.run(classify("q", _reply("EMERGENCY."))) == ("emergency", "llm")


def test_classify_falls_back_to_personal() -> None:
    assert asyncio.run(classify("q", _reply("not sure"))) == ("personal", "fallback")
    failing = _reply(error=LLMError("timeout"))
    assert asyncio.run(classify("q", failing)) == ("personal", "fallback")
    assert len(failing.calls) == 2  # retried once
    assert asyncio.run(classify("Should I stop Eliquis?", failing)) == ("personal", "keywords")


def test_classify_recovers_after_one_timeout() -> None:
    attempts: list[int] = []

    async def flaky(messages: list[dict[str, str]]) -> AsyncIterator[str]:
        attempts.append(1)
        if len(attempts) == 1:
            raise LLMError("timeout")
        yield "info"

    assert asyncio.run(classify("What is Eliquis?", flaky)) == ("info", "llm")


NAMES = medicine_names(
    ["eliquis", "brilique", "ozempic", "rybelsus", "entresto"],
    {"eliquis": "apixaban", "brilique": "ticagrelor", "ozempic": "semaglutide",
     "rybelsus": "semaglutide", "entresto": "sacubitril;valsartan"},
)


def test_detect_medicines_brand_substance_and_greek() -> None:
    assert detect_medicines("Can ELIQUIS be crushed?", NAMES) == ["eliquis"]
    assert detect_medicines("apixaban and ticagrelor", NAMES) == ["eliquis", "brilique"]
    assert detect_medicines("Παρενέργειες της τικαγρελόρης", NAMES) == ["brilique"]
    assert detect_medicines("σεμαγλουτίδη", NAMES) == ["ozempic", "rybelsus"]
    assert detect_medicines("valsartan dose", NAMES) == ["entresto"]
    assert detect_medicines("What is paracetamol?", NAMES) == []


class FakeService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def manifest(self) -> Any:
        return SimpleNamespace(medicines=["eliquis", "xarelto"],
                               active_substances={"eliquis": "apixaban"})

    def search(self, query: str, **kwargs: Any) -> SearchResponse:
        self.calls.append(kwargs)
        hit = SearchHit(
            rank=1, score=0.5, point_id="p", chunk_id="eliquis:smpc:4.5", medicine="eliquis",
            lang="en", doc="smpc", section="4.5", title="T", text="t", source_url="u",
            revision_date=None,
        )
        return SearchResponse(
            query=query, corpus_version="v", prefetch_limit=30, limit=kwargs["limit"], count=1,
            latency_ms=1.0, embed_model="e5", embed_revision="r", results=[hit],
        )


def _events(body: str) -> list[tuple[str, Any]]:
    return [
        (block.split("\n")[0].removeprefix("event: "),
         json.loads(block.split("\n")[1].removeprefix("data: ")))
        for block in body.strip().split("\n\n")
    ]


def _llm(label: str):  # type: ignore[no-untyped-def]
    async def generate(messages: list[dict[str, str]]) -> AsyncIterator[str]:
        yield label if messages[0]["content"] == SAFETY_PROMPT else "Yes [Eliquis SmPC 4.5]."
    return generate


def test_ask_redirects_emergency_without_retrieval() -> None:
    service = FakeService()
    client = TestClient(create_app(settings=Settings(), service=service, generate=_llm("info")))  # type: ignore[arg-type]
    events = _events(client.post("/v1/ask", json={"question": "Πήρα πολλά χάπια"}).text)
    assert [name for name, _ in events] == ["safety", "token", "citations", "done"]
    assert events[0][1] == {"label": "emergency"}
    assert events[1][1]["text"] == MESSAGES["emergency"]["el"]
    assert events[3][1]["refused"] is True
    assert service.calls == []


def test_ask_redirects_personal_label_from_llm() -> None:
    service = FakeService()
    client = TestClient(create_app(settings=Settings(), service=service, generate=_llm("personal")))  # type: ignore[arg-type]
    events = _events(client.post("/v1/ask", json={"question": "Eliquis and me"}).text)
    assert events[0][1] == {"label": "personal"}
    assert events[1][1]["text"] == MESSAGES["personal"]["en"]
    assert service.calls == []


@pytest.mark.parametrize("reply", ["not sure", LLMError("provider unavailable")])
def test_ask_reports_classifier_failure_without_retrieval(reply: str | LLMError) -> None:
    service = FakeService()
    generate = _reply(error=reply) if isinstance(reply, LLMError) else _reply(reply)
    client = TestClient(create_app(settings=Settings(), service=service, generate=generate))  # type: ignore[arg-type]
    response = client.post("/v1/ask", json={"question": "What is Eliquis used for?"})
    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "safety_unavailable"
    assert service.calls == []


def test_ask_filters_search_to_named_medicines() -> None:
    service = FakeService()
    client = TestClient(create_app(settings=Settings(), service=service, generate=_llm("info")))  # type: ignore[arg-type]
    events = _events(client.post("/v1/ask", json={"question": "Can apixaban be crushed?"}).text)
    assert events[0][1] == {"label": "info"}
    assert service.calls[0]["medicine"] == ["eliquis"]
    client.post("/v1/ask", json={"question": "What is paracetamol?"})
    assert service.calls[1]["medicine"] is None
