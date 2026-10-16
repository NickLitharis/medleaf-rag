"""FastAPI surface: ``POST /v1/search`` (plus a GET mirror) and ``POST /v1/ask`` (SSE).

Every failure mode is explicit. A missing index, an unreachable Qdrant or a model that
cannot load all return HTTP 503 with ``{"error": ..., "detail": ...}``; the endpoint
never answers 200 with an empty result list to hide a broken index. Once ``/v1/ask`` has
started streaming, an LLM failure arrives as an ``error`` event instead.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from functools import partial
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from medleaf.config import Settings, get_settings
from medleaf.generation.answer import (
    DISCLAIMERS,
    CitationFilter,
    LLMError,
    build_messages,
    citation_key,
    detect_lang,
    select_sources,
    stream_chat,
)
from medleaf.retrieval.index import (
    IndexIncompleteError,
    IndexMissingError,
    RetrievalUnavailableError,
)
from medleaf.retrieval.names import detect_medicines, medicine_names
from medleaf.retrieval.service import ReadyState, RetrievalService, SearchHit, SearchResponse
from medleaf.safety.classifier import MESSAGES, Label, classify

DOC_KINDS = Literal["smpc", "leaflet"]
LANGS = Literal["en", "el"]
SEARCH_PAGE = Path(__file__).with_name("static") / "index.html"
Generate = Callable[[list[dict[str, str]]], AsyncIterator[str]]
log = logging.getLogger("medleaf.ask")
if not log.handlers:  # uvicorn only configures its own loggers; SPEC §9 logs every label
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)


class MedicineOption(BaseModel):
    """One indexed medicine, for the search page's filter."""

    slug: str
    active_substance: str | None = None


class SearchRequest(BaseModel):
    """Body of ``POST /v1/search``."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Natural-language question or keyword phrase.")
    lang: LANGS | None = Field(default=None, description="Restrict to one corpus language.")
    medicine: list[str] | None = Field(
        default=None, description="Restrict to these medicine slugs, e.g. ['eliquis']."
    )
    doc: DOC_KINDS | None = Field(
        default=None, description="Restrict to the SmPC or the package leaflet."
    )
    active_substance: str | None = Field(
        default=None, description="Restrict to chunks whose medicine has this substance."
    )
    corpus_version: str | None = Field(
        default=None,
        description="Exact corpus version filter; defaults to the indexed corpus version.",
    )
    limit: int = Field(default=10, ge=1, le=50)


class AskRequest(BaseModel):
    """Body of ``POST /v1/ask`` (SPEC section 10)."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=1000)
    lang: LANGS | None = Field(
        default=None, description="Answer language; detected from the question when null."
    )
    top_k: int = Field(default=6, ge=1, le=10, description="Source chunks given to the LLM.")


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _error(status: int, code: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": code, "detail": detail})


def _run_search(
    service: RetrievalService,
    *,
    query: str,
    lang: str | None,
    medicine: list[str] | None,
    doc: str | None,
    active_substance: str | None,
    corpus_version: str | None,
    limit: int,
) -> SearchResponse:
    try:
        return service.search(
            query,
            lang=lang,
            medicine=medicine,
            doc=doc,
            active_substance=active_substance,
            corpus_version=corpus_version,
            limit=limit,
        )
    except IndexIncompleteError as error:
        raise HTTPException(
            status_code=503, detail={"error": "index_incomplete", "detail": str(error)}
        ) from error
    except IndexMissingError as error:
        raise HTTPException(
            status_code=503, detail={"error": "index_missing", "detail": str(error)}
        ) from error
    except RetrievalUnavailableError as error:
        raise HTTPException(
            status_code=503, detail={"error": "retrieval_unavailable", "detail": str(error)}
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=422, detail={"error": "bad_request", "detail": str(error)}
        ) from error


def create_app(
    *,
    settings: Settings | None = None,
    service: RetrievalService | None = None,
    generate: Generate | None = None,
) -> FastAPI:
    """Build the API app; ``service`` and ``generate`` are injectable so tests use fakes."""
    resolved_settings = settings or get_settings()
    retriever = service or RetrievalService(resolved_settings)
    llm: Generate = generate or partial(stream_chat, resolved_settings)

    app = FastAPI(
        title="MedLeaf RAG API",
        version="0.3.0",
        description=(
            "Hybrid (dense e5 + Qdrant sparse IDF, RRF) retrieval and cited answers over "
            "EMA medicine information. Not medical advice; EMA is the source of the corpus."
        ),
    )
    app.state.settings = resolved_settings
    app.state.service = retriever

    @app.get("/", include_in_schema=False)
    def search_page() -> FileResponse:
        """A one-page search UI over ``/v1/search``."""
        return FileResponse(SEARCH_PAGE)

    @app.get("/v1/medicines", tags=["retrieval"], response_model=list[MedicineOption])
    def medicines() -> list[MedicineOption]:
        """Medicines in the current index, from the build manifest."""
        try:
            manifest = retriever.manifest()
        except IndexMissingError as error:
            raise HTTPException(
                status_code=503, detail={"error": "index_missing", "detail": str(error)}
            ) from error
        return [
            MedicineOption(slug=slug, active_substance=manifest.active_substances.get(slug))
            for slug in sorted(manifest.medicines)
        ]

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict[str, str]:
        """Liveness: the process answers."""
        return {"status": "ok"}

    @app.get("/readyz", tags=["ops"], response_model=ReadyState)
    def readyz() -> ReadyState:
        """Readiness: Qdrant reachable, collection populated, manifest present."""
        state = retriever.ready()
        if not state.ready:
            return JSONResponse(status_code=503, content=state.model_dump(mode="json"))  # type: ignore[return-value]
        return state

    @app.post("/v1/search", tags=["retrieval"], response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        """Hybrid search returning chunks with full source provenance."""
        return _run_search(
            retriever,
            query=request.query,
            lang=request.lang,
            medicine=request.medicine,
            doc=request.doc,
            active_substance=request.active_substance,
            corpus_version=request.corpus_version,
            limit=request.limit,
        )

    @app.get("/v1/search", tags=["retrieval"], response_model=SearchResponse)
    def search_get(
        query: Annotated[str, Query(min_length=1)],
        lang: LANGS | None = None,
        medicine: Annotated[list[str] | None, Query()] = None,
        doc: DOC_KINDS | None = None,
        active_substance: str | None = None,
        corpus_version: str | None = None,
        limit: Annotated[int, Query(ge=1, le=50)] = 10,
    ) -> SearchResponse:
        """GET mirror of ``POST /v1/search`` so ``curl`` acceptance is one line."""
        return _run_search(
            retriever,
            query=query,
            lang=lang,
            medicine=medicine,
            doc=doc,
            active_substance=active_substance,
            corpus_version=corpus_version,
            limit=limit,
        )

    @app.post("/v1/ask", tags=["generation"], response_class=StreamingResponse)
    async def ask(request: AskRequest) -> StreamingResponse:
        """Question → cited answer, streamed as Server-Sent Events.

        Events: ``safety``, ``retrieval``, ``token`` (repeated), ``citations``, ``done``;
        or ``error`` in place of the rest. A ``personal`` or ``emergency`` question gets a
        fixed message and never reaches retrieval.
        """
        started = time.perf_counter()
        lang = request.lang or detect_lang(request.question)
        label, source = await classify(request.question, llm)
        log.info("ask safety: label=%s source=%s", label, source)
        if label != "info":
            return StreamingResponse(
                _redirect_events(label, lang, started),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        named = detect_medicines(request.question, _names(retriever))
        # Search runs before the stream opens, so a broken index is still a plain 503.
        found = await run_in_threadpool(
            partial(
                _run_search,
                retriever,
                query=request.question,
                lang=None,  # both languages; the answer is written in ``lang``
                medicine=named or None,
                doc=None,
                active_substance=None,
                corpus_version=None,
                limit=resolved_settings.ask_search_limit,
            )
        )
        sources = select_sources(found.results, lang, request.top_k)
        return StreamingResponse(
            _answer_events(request.question, lang, sources, llm, started),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def _names(retriever: RetrievalService) -> dict[str, list[str]]:
    """Medicine names from the manifest; none when it is missing (the search then 503s)."""
    try:
        manifest = retriever.manifest()
    except IndexMissingError:
        return {}
    return medicine_names(manifest.medicines, manifest.active_substances)


async def _redirect_events(label: Label, lang: str, started: float) -> AsyncIterator[str]:
    yield _sse("safety", {"label": label})
    yield _sse("token", {"text": MESSAGES[label][lang]})
    yield _sse("citations", [])
    yield _sse(
        "done",
        {
            "trace_id": uuid.uuid4().hex,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "refused": True,
            "disclaimer": DISCLAIMERS[lang],
        },
    )


async def _answer_events(
    question: str,
    lang: str,
    sources: list[SearchHit],
    llm: Generate,
    started: float,
) -> AsyncIterator[str]:
    trace_id = uuid.uuid4().hex
    yield _sse("safety", {"label": "info"})
    yield _sse(
        "retrieval",
        [{"chunk_id": hit.chunk_id, "lang": hit.lang, "score": hit.score} for hit in sources],
    )
    by_key: dict[str, SearchHit] = {}
    for hit in sources:
        by_key.setdefault(citation_key(hit), hit)
    citations = CitationFilter(set(by_key))
    try:
        async for piece in llm(build_messages(question, sources, lang)):
            text = citations.feed(piece)
            if text:
                yield _sse("token", {"text": text})
        tail = citations.flush()
        if tail:
            yield _sse("token", {"text": tail})
    except LLMError as error:
        log.warning("ask %s: LLM failed after retrieval %s: %s", trace_id,
                    [hit.chunk_id for hit in sources], error)
        yield _sse("error", {"code": "llm_unavailable", "message": str(error)})
        return
    if citations.invented:
        log.warning("ask %s: dropped invented citations %s", trace_id, citations.invented)
    yield _sse(
        "citations",
        [
            {
                "key": key,
                "medicine": by_key[key].medicine,
                "doc": by_key[key].doc,
                "section": by_key[key].section,
                "url": by_key[key].source_url,
                "revision_date": by_key[key].revision_date,
            }
            for key in citations.used
        ],
    )
    yield _sse(
        "done",
        {
            "trace_id": trace_id,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            # ponytail: "no valid citation" stands in for refusal until the slice-2 rerank
            # threshold and refusal checks exist.
            "refused": not citations.used,
            "disclaimer": DISCLAIMERS[lang],
            "invented_citations": citations.invented,
        },
    )


__all__ = ["AskRequest", "SearchRequest", "create_app"]
