"""API tests for ``/v1/search`` using an injected fake retriever.

The point of these tests is the *failure* behaviour as much as the happy path: a
missing index or an unreachable service must answer 503 with a machine-readable code,
never 200 with an empty list.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from medleaf.api.app import create_app
from medleaf.config import Settings
from medleaf.retrieval.index import (
    IndexIncompleteError,
    IndexMissingError,
    RetrievalUnavailableError,
)
from medleaf.retrieval.service import ReadyState, SearchHit, SearchResponse


def _hit(rank: int = 1, lang: str = "en") -> SearchHit:
    return SearchHit(
        rank=rank,
        score=0.5,
        point_id=f"point-{rank}",
        chunk_id="eliquis:smpc:4.5",
        medicine="eliquis",
        lang=lang,
        doc="smpc",
        section="4.5",
        title="Interaction with other medicinal products",
        text="Eliquis should be used with caution with NSAIDs.",
        source_url="https://www.ema.europa.eu/en/documents/product-information/eliquis-epar-product-information_en.pdf",
        revision_date="2026-04-01",
        corpus_version="medicines.yaml:abc",
    )


class FakeService:
    """Duck-typed stand-in for :class:`~medleaf.retrieval.service.RetrievalService`."""

    def __init__(
        self,
        *,
        results: Sequence[SearchHit] | None = None,
        ready: ReadyState | None = None,
        search_error: Exception | None = None,
    ) -> None:
        self._results = list(results if results is not None else [_hit()])
        self._ready = ready or ReadyState(
            ready=True,
            detail="10 points indexed",
            collection="medleaf_chunks",
            points=10,
            corpus_version="medicines.yaml:abc",
            embed_model="intfloat/multilingual-e5-base",
            embed_revision="rev",
        )
        self._search_error = search_error
        self.calls: list[dict[str, Any]] = []

    def ready(self) -> ReadyState:
        return self._ready

    def manifest(self) -> SimpleNamespace:
        if self._search_error is not None:
            raise self._search_error
        return SimpleNamespace(
            medicines=["xarelto", "eliquis"],
            active_substances={"eliquis": "apixaban", "xarelto": None},
        )

    def search(self, query: str, **kwargs: Any) -> SearchResponse:
        self.calls.append({"query": query, **kwargs})
        if self._search_error is not None:
            raise self._search_error
        return SearchResponse(
            query=query,
            lang=kwargs.get("lang"),
            medicine=kwargs.get("medicine"),
            doc=kwargs.get("doc"),
            corpus_version=kwargs.get("corpus_version") or "medicines.yaml:abc",
            prefetch_limit=30,
            limit=kwargs.get("limit") or 10,
            count=len(self._results),
            latency_ms=2.5,
            embed_model="intfloat/multilingual-e5-base",
            embed_revision="rev",
            results=self._results,
        )


def _client(service: FakeService) -> TestClient:
    app = create_app(settings=Settings(), service=service)  # type: ignore[arg-type]
    return TestClient(app)


def test_healthz_is_alive_even_when_retrieval_is_broken() -> None:
    service = FakeService(search_error=IndexMissingError("no collection"))
    with _client(service) as client:
        assert client.get("/healthz").status_code == 200


def test_readyz_ok_when_index_is_present() -> None:
    with _client(FakeService()) as client:
        response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert response.json()["points"] == 10


def test_readyz_503_explains_what_is_missing() -> None:
    service = FakeService(
        ready=ReadyState(
            ready=False, detail="collection 'x' does not exist; run `medleaf index build`"
        )
    )
    with _client(service) as client:
        response = client.get("/readyz")
    assert response.status_code == 503
    assert "medleaf index build" in response.json()["detail"]


def test_post_search_returns_chunks_with_provenance() -> None:
    with _client(FakeService()) as client:
        response = client.post(
            "/v1/search", json={"query": "interactions with NSAIDs?", "lang": "en", "limit": 5}
        )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["fusion"] == "rrf"
    assert body["results"][0]["chunk_id"] == "eliquis:smpc:4.5"
    assert body["results"][0]["source_url"].endswith("_en.pdf")
    assert body["results"][0]["corpus_version"] == "medicines.yaml:abc"


def test_get_search_mirror_passes_filters() -> None:
    service = FakeService()
    with _client(service) as client:
        response = client.get(
            "/v1/search",
            params={"query": "ναυτία", "lang": "el", "medicine": ["eliquis"], "doc": "smpc"},
        )
    assert response.status_code == 200
    assert service.calls[0]["lang"] == "el"
    assert service.calls[0]["medicine"] == ["eliquis"]
    assert service.calls[0]["doc"] == "smpc"


def test_search_503_when_the_index_is_missing() -> None:
    service = FakeService(search_error=IndexMissingError("collection missing"))
    with _client(service) as client:
        response = client.post("/v1/search", json={"query": "nausea"})
    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "index_missing"


def test_search_503_when_the_index_is_incomplete() -> None:
    service = FakeService(search_error=IndexIncompleteError("804 points but 8797 recorded"))
    with _client(service) as client:
        response = client.post("/v1/search", json={"query": "nausea"})
    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "index_incomplete"


def test_search_503_when_retrieval_is_unavailable() -> None:
    service = FakeService(search_error=RetrievalUnavailableError("qdrant unavailable: refused"))
    with _client(service) as client:
        response = client.get("/v1/search", params={"query": "nausea"})
    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "retrieval_unavailable"


@pytest.mark.parametrize("payload", [{}, {"query": ""}])
def test_search_rejects_an_empty_query(payload: dict[str, str]) -> None:
    with _client(FakeService()) as client:
        response = client.post("/v1/search", json=payload)
    assert response.status_code == 422


def test_search_rejects_an_unknown_filter_value() -> None:
    with _client(FakeService()) as client:
        response = client.post("/v1/search", json={"query": "nausea", "lang": "fr"})
    assert response.status_code == 422


def test_root_serves_the_search_page() -> None:
    with _client(FakeService()) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "/v1/search" in response.text


def test_medicines_lists_the_indexed_slugs_sorted() -> None:
    with _client(FakeService()) as client:
        response = client.get("/v1/medicines")
    assert response.status_code == 200
    assert response.json() == [
        {"slug": "eliquis", "active_substance": "apixaban"},
        {"slug": "xarelto", "active_substance": None},
    ]


def test_medicines_503_without_an_index() -> None:
    service = FakeService(search_error=IndexMissingError("index manifest missing"))
    with _client(service) as client:
        response = client.get("/v1/medicines")
    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "index_missing"
