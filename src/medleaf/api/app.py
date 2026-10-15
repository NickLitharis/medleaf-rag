"""FastAPI surface for M2: ``POST /v1/search`` (and a GET mirror for ``curl``).

Every failure mode is explicit. A missing index, an unreachable Qdrant or a model that
cannot load all return HTTP 503 with ``{"error": ..., "detail": ...}``; the endpoint
never answers 200 with an empty result list to hide a broken index.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from medleaf.config import Settings, get_settings
from medleaf.retrieval.index import (
    IndexIncompleteError,
    IndexMissingError,
    RetrievalUnavailableError,
)
from medleaf.retrieval.service import ReadyState, RetrievalService, SearchResponse

DOC_KINDS = Literal["smpc", "leaflet"]
LANGS = Literal["en", "el"]
SEARCH_PAGE = Path(__file__).with_name("static") / "index.html"


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
) -> FastAPI:
    """Build the API app; ``service`` is injectable so tests can use fakes."""
    resolved_settings = settings or get_settings()
    retriever = service or RetrievalService(resolved_settings)

    app = FastAPI(
        title="MedLeaf RAG retrieval API",
        version="0.2.0",
        description=(
            "Hybrid (dense e5 + Qdrant sparse IDF, RRF) retrieval over EMA medicine "
            "information. Not medical advice; EMA is the source of the corpus."
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

    return app


__all__ = ["SearchRequest", "create_app"]
