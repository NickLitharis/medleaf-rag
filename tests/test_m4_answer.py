"""M4 slice 1: cited answers over ``/v1/ask`` with a fake retriever and a fake LLM."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from medleaf.api.app import create_app
from medleaf.config import Settings
from medleaf.generation.answer import (
    CitationFilter,
    LLMError,
    build_messages,
    detect_lang,
    select_sources,
    stream_chat,
)
from medleaf.retrieval.index import IndexMissingError, RetrievalUnavailableError
from medleaf.retrieval.service import SearchHit, SearchResponse
from medleaf.safety.classifier import SAFETY_PROMPT


def _hit(chunk_id: str, lang: str = "en", rank: int = 1) -> SearchHit:
    medicine, doc, section = chunk_id.split("#", 1)[0].split(":")
    return SearchHit(
        rank=rank, score=0.5, point_id=f"{chunk_id}-{lang}", chunk_id=chunk_id,
        medicine=medicine, lang=lang, doc=doc, section=section, title="Title",
        text=f"text of {chunk_id} ({lang})", source_url=f"https://ema.example/{medicine}_{lang}.pdf",
        revision_date="2026-04-01",
    )


class FakeService:
    """Stand-in for RetrievalService.search; records its calls."""

    def __init__(self, results: Sequence[SearchHit] = (), error: Exception | None = None) -> None:
        self.results = list(results) or [_hit("eliquis:smpc:4.5")]
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def manifest(self) -> Any:
        raise IndexMissingError("no manifest in this test")

    def search(self, query: str, **kwargs: Any) -> SearchResponse:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SearchResponse(
            query=query, corpus_version="v", prefetch_limit=30, limit=kwargs["limit"],
            count=len(self.results), latency_ms=1.0, embed_model="e5", embed_revision="rev",
            results=self.results,
        )


def _events(body: str) -> list[tuple[str, object]]:
    events = []
    for block in body.strip().split("\n\n"):
        name, data = block.split("\n", 1)
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


def _llm(*pieces: str, error: Exception | None = None):  # type: ignore[no-untyped-def]
    async def generate(messages: list[dict[str, str]]) -> AsyncIterator[str]:
        if messages[0]["content"] == SAFETY_PROMPT:
            yield "info"
            return
        for piece in pieces:
            yield piece
        if error is not None:
            raise error
    return generate


def test_detect_lang() -> None:
    assert detect_lang("Ποιες είναι οι παρενέργειες του Ozempic;") == "el"
    assert detect_lang("What are the side effects of Ozempic?") == "en"
    assert detect_lang("4.5?") == "en"


def test_citation_filter_drops_invented_keys_across_chunk_boundaries() -> None:
    citations = CitationFilter({"Eliquis SmPC 4.5", "Eliquis SmPC 4.3"})
    streamed = "".join(
        citations.feed(piece)
        for piece in ["Use with care [Eliq", "uis SmPC 4.5]. Made up [Eliquis SmPC 9.9]. ",
                      "Both [Eliquis SmPC 4.3; Fake 1]."]
    ) + citations.flush()
    assert streamed == "Use with care [Eliquis SmPC 4.5]. Made up . Both [Eliquis SmPC 4.3]."
    assert citations.used == ["Eliquis SmPC 4.5", "Eliquis SmPC 4.3"]
    assert citations.invented == ["Eliquis SmPC 9.9", "Fake 1"]


@pytest.mark.parametrize("space", ["\u00a0", "\u202f", "\u2009"])
def test_citation_filter_normalizes_unicode_spaces(space: str) -> None:
    citations = CitationFilter({"Eliquis Leaflet 1"})
    text = citations.feed(f"Uses [Eliquis Leaflet{space}1].") + citations.flush()
    assert text == "Uses [Eliquis Leaflet 1]."
    assert citations.used == ["Eliquis Leaflet 1"]
    assert citations.invented == []


def test_citation_filter_releases_a_long_unclosed_bracket() -> None:
    citations = CitationFilter(set())
    text = "[" + "x" * 100
    assert citations.feed(text) + citations.flush() == text


def test_select_sources_prefers_the_question_language_and_caps_k() -> None:
    hits = [_hit("a:smpc:4.5", "en"), _hit("b:smpc:4.3", "el"), _hit("a:smpc:4.5", "el"),
            _hit("c:smpc:6.4", "en")]
    chosen = select_sources(hits, "el", k=2)
    assert [(hit.chunk_id, hit.lang) for hit in chosen] == [("a:smpc:4.5", "el"),
                                                           ("b:smpc:4.3", "el")]
    prompt = build_messages("q", chosen, "el")
    assert "Greek" in prompt[0]["content"]
    assert "[A SmPC 4.5] Title\ntext of a:smpc:4.5 (el)" in prompt[1]["content"]


@pytest.mark.parametrize("space", [" ", "\u202f"])
def test_ask_streams_a_cited_answer(space: str) -> None:
    service = FakeService([_hit("eliquis:smpc:4.5"), _hit("eliquis:smpc:4.3", rank=2)])
    llm = _llm("Avoid with NSAIDs [Eliquis ", f"SmPC{space}4.5]. Fake [Xarelto SmPC 1].")
    client = TestClient(create_app(settings=Settings(), service=service, generate=llm))  # type: ignore[arg-type]
    response = client.post("/v1/ask", json={"question": "Can I take Eliquis with ibuprofen?"})
    assert response.status_code == 200
    events = _events(response.text)
    assert [name for name, _ in events][:2] == ["safety", "retrieval"]
    assert [name for name, _ in events][-2:] == ["citations", "done"]
    text = "".join(data["text"] for name, data in events if name == "token")  # type: ignore[index]
    assert text == "Avoid with NSAIDs [Eliquis SmPC 4.5]. Fake ."
    citations, done = events[-2][1], events[-1][1]
    assert citations == [{"key": "Eliquis SmPC 4.5", "medicine": "eliquis", "doc": "smpc",
                          "section": "4.5", "url": "https://ema.example/eliquis_en.pdf",
                          "revision_date": "2026-04-01"}]
    assert done["refused"] is False  # type: ignore[index]
    assert done["invented_citations"] == ["Xarelto SmPC 1"]  # type: ignore[index]
    assert service.calls[0]["lang"] is None  # both languages are searched


def test_ask_reports_an_llm_failure_as_an_error_event() -> None:
    llm = _llm("Partial", error=LLMError("model: HTTP 401: no key"))
    client = TestClient(create_app(settings=Settings(), service=FakeService(), generate=llm))  # type: ignore[arg-type]
    events = _events(client.post("/v1/ask", json={"question": "q"}).text)
    assert [name for name, _ in events] == ["safety", "retrieval", "error"]
    assert events[-1][1] == {"code": "llm_unavailable", "message": "model: HTTP 401: no key"}


@pytest.mark.parametrize(
    "pieces", [("A partial introduction. ", "17", "% at 1 mg. [Ozempic SmPC 4.8]"),
               ("[quote:", "999]"), ("[quote:1]", " and 19.9% at 0.5 mg.")],
)
def test_ask_never_emits_unverified_numerical_tokens(pieces: tuple[str, ...]) -> None:
    hit = _hit("ozempic:smpc:4.8")
    hit.text = "Nausea occurred in 17% and 19.9% at 0.5 mg and 1 mg, respectively."
    client = TestClient(create_app(
        settings=Settings(), service=FakeService([hit]), generate=_llm(*pieces),
    ))  # type: ignore[arg-type]
    events = _events(client.post("/v1/ask", json={"question": "What does the source say?"}).text)
    assert [name for name, _ in events] == ["safety", "retrieval", "error"]
    assert events[-1][1]["code"] == "numerical_grounding_failed"  # type: ignore[index]


def test_ask_renders_a_quote_reference_split_across_provider_tokens() -> None:
    hit = _hit("ozempic:smpc:4.8")
    hit.text = "Nausea occurred in 17% and 19.9% at 0.5 mg and 1 mg, respectively."
    client = TestClient(create_app(
        settings=Settings(), service=FakeService([hit]), generate=_llm("Reported: [quo", "te:1]"),
    ))  # type: ignore[arg-type]
    events = _events(client.post("/v1/ask", json={"question": "What does the source say?"}).text)
    assert [name for name, _ in events] == ["safety", "retrieval", "token", "citations", "done"]
    assert events[2][1]["text"] == f'Reported: “{hit.text}” [Ozempic SmPC 4.8]'  # type: ignore[index]
    assert events[-1][1]["refused"] is False  # type: ignore[index]


def test_ask_native_selection_renders_only_trusted_sources() -> None:
    hit = _hit("ozempic:smpc:4.8")
    hit.text = "Nausea occurred in 17% and 19.9% at 0.5 mg and 1 mg, respectively."
    client = TestClient(create_app(
        settings=Settings(llm_source_quotes_only=True), service=FakeService([hit]),
        generate=_llm('{"quote_', 'ids":["quote:1"]}'),
    ))  # type: ignore[arg-type]
    events = _events(client.post("/v1/ask", json={"question": "What does the source say?"}).text)
    text = "".join(data["text"] for name, data in events if name == "token")
    assert hit.text in text
    assert "[Ozempic SmPC 4.8]" in text
    assert 'quote_ids' not in text
    assert events[-1][0] == "done"


def test_native_schema_applies_to_answer_but_not_safety(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def provider(
        settings: Settings, messages: list[dict[str, str]], **kwargs: Any,
    ) -> AsyncIterator[str]:
        if messages[0]["content"] == SAFETY_PROMPT:
            assert "response_format" not in kwargs
            calls.append("safety")
            yield "info"
        else:
            schema = kwargs["response_format"]["json_schema"]["schema"]
            assert "quote:1" in schema["properties"]["quote_ids"]["items"]["enum"]
            calls.append("answer")
            yield '{"quote_ids":["quote:1"]}'

    monkeypatch.setattr("medleaf.api.app.stream_chat", provider)
    client = TestClient(create_app(
        settings=Settings(llm_source_quotes_only=True), service=FakeService(),
    ))  # type: ignore[arg-type]
    events = _events(client.post("/v1/ask", json={"question": "What does the source say?"}).text)
    assert calls == ["safety", "answer"]
    assert events[-1][0] == "done"


@pytest.mark.parametrize("raw", [
    '{"quote_ids":["quote:999"]}', '{"quote_ids":["quote:1"],"answer":"17% at 1 mg"}',
    '17% at 1 mg', '{"quote_ids":',
])
def test_ask_native_selection_never_leaks_invalid_output(raw: str) -> None:
    client = TestClient(create_app(
        settings=Settings(llm_source_quotes_only=True),
        service=FakeService([_hit("ozempic:smpc:4.8")]),
        generate=_llm(raw),
    ))  # type: ignore[arg-type]
    events = _events(client.post("/v1/ask", json={"question": "What does the source say?"}).text)
    assert any(name == "error" for name, _ in events)
    assert not any(name in {"token", "done"} for name, _ in events)


def test_ask_returns_503_when_retrieval_is_down() -> None:
    service = FakeService(error=RetrievalUnavailableError("qdrant down"))
    client = TestClient(create_app(settings=Settings(), service=service, generate=_llm("x")))  # type: ignore[arg-type]
    response = client.post("/v1/ask", json={"question": "q"})
    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "retrieval_unavailable"


def _collect(settings: Settings, handler) -> list[str]:  # type: ignore[no-untyped-def]
    async def run() -> list[str]:
        transport = httpx.MockTransport(handler)
        return [piece async for piece in stream_chat(settings, [], transport=transport)]
    return asyncio.run(run())


def test_stream_chat_parses_openai_style_sse() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = (": OPENROUTER PROCESSING\n\n"
                'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
                'data: {"choices":[{"delta":{}}]}\n\ndata: [DONE]\n\n')
        return httpx.Response(200, text=body)

    settings = Settings(MEDLEAF_LLM_API_KEY="test-key")  # type: ignore[call-arg]
    assert _collect(settings, handler) == ["Hel", "lo"]
    assert str(seen[0].url) == "https://openrouter.ai/api/v1/chat/completions"
    assert seen[0].headers["Authorization"] == "Bearer test-key"


def test_stream_chat_forwards_native_response_format() -> None:
    expected = {"type": "json_schema", "json_schema": {"name": "test"}}

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["response_format"] == expected
        return httpx.Response(
            200, text='data: {"choices":[{"delta":{"content":"{}"}}]}\n\ndata: [DONE]\n\n',
        )

    async def run() -> list[str]:
        return [piece async for piece in stream_chat(
            Settings(), [], transport=httpx.MockTransport(handler), response_format=expected,
        )]

    assert asyncio.run(run()) == ["{}"]


def test_stream_chat_raises_on_provider_errors() -> None:
    with pytest.raises(LLMError, match="HTTP 401"):
        _collect(Settings(), lambda request: httpx.Response(401, text="no auth"))
    with pytest.raises(LLMError, match="rate limited"):
        _collect(Settings(), lambda request: httpx.Response(
            200, text='data: {"error": {"message": "rate limited"}}\n\n'))
